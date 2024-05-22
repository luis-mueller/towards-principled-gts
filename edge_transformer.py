import math
import torch
import torch.nn.functional as F
import tqdm
from functools import partial
from torch_geometric.utils import to_dense_batch, to_dense_adj, scatter
from torch_geometric.nn.aggr import Set2Set


class CosineWithWarmupLR:
    """Adapted from https://github.com/karpathy/nanoGPT"""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, epoch: int):
        lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int):
        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        # 2) if epoch > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)


def transform_dataset(dataset, transform):
    data_list = []
    for data in tqdm.tqdm(dataset, miniters=len(dataset) / 50):
        data_list.append(transform(data))
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset._data, dataset.slices = dataset.collate(data_list)
    return dataset


def token_index_transform(data):
    token_index = torch.arange(data.num_nodes).unsqueeze(0)
    token_index = torch.cat(
        [
            token_index.repeat_interleave(data.num_nodes, 1),
            torch.arange(data.num_nodes).repeat(data.num_nodes).unsqueeze(0),
        ],
        dim=0,
    )
    data.token_index = token_index
    return data


class RRWPTransform:
    def __init__(self, num_iter: int = 0):
        self.num_iter = num_iter

    def __call__(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes

        adj: torch.Tensor = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1

        deg_inv = torch.nan_to_num(1 / adj.sum(1), posinf=0)
        P = torch.diag(deg_inv) @ adj

        probs = torch.empty((num_nodes, num_nodes, self.num_iter))
        for k in range(self.num_iter):
            probs[:, :, k] = P
            P = P @ P

        data.rrwp = probs[data.token_index[0], data.token_index[1]]
        return data


class RRWPEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_iter):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_iter, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
            torch.nn.ReLU(),
        )

    def forward(self, data):
        pos_enc: torch.Tensor = self.encoder(data.rrwp)
        data.token_attr = pos_enc
        return data


class EdgePositionalEncoder(torch.nn.Module):
    def __init__(
        self,
        positional_encoder: str = None,
        positional_dim: int = 0,
        **positional_encoder_kwargs,
    ):
        super().__init__()
        if positional_encoder == "rrwp":
            self.positional_encoder = RRWPEncoder(
                positional_dim, **positional_encoder_kwargs
            )
        elif positional_encoder is not None:
            raise ValueError(
                f"Positional encoder {positional_encoder} is not supported for edges"
            )

    def forward(self, data):
        if hasattr(self, "positional_encoder"):
            data = self.positional_encoder(data)
        return data


class FeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        node_encoder,
        edge_encoder,
        node_dim=None,
        edge_dim=0,
        edge_positional_encoder: str = None,
        edge_positional_dim: int = 0,
        edge_positional_encoder_kwargs={},
    ):
        super().__init__()
        node_embed_dim = embed_dim
        edge_embed_dim = embed_dim - edge_positional_dim

        if node_encoder == "embedding":
            self.node_encoder = torch.nn.Embedding(node_dim, node_embed_dim)
        elif node_encoder == "linear":
            self.node_encoder = torch.nn.Linear(node_dim, node_embed_dim)
        else:
            self.node_encoder = node_encoder

        if edge_dim == 0:
            edge_encoder = "embedding"
            edge_dim = 2

        if edge_encoder == "embedding":
            self.edge_encoder = torch.nn.Embedding(edge_dim, edge_embed_dim)
        elif edge_encoder == "linear":
            self.edge_encoder = torch.nn.Linear(edge_dim, edge_embed_dim)
        else:
            self.edge_encoder = edge_encoder

        self.edge_positional_encoder = EdgePositionalEncoder(
            edge_positional_encoder,
            edge_positional_dim,
            **edge_positional_encoder_kwargs,
        )

    def forward(self, data):
        data.x = self.node_encoder(data.x)

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.ones_like(data.edge_index[0]).to(torch.long)

        data.edge_attr = self.edge_encoder(data.edge_attr)
        data = self.edge_positional_encoder(data)
        return data


class MLP(torch.nn.Sequential):
    def __init__(
        self, input_dim, output_dim, dropout: float = 0.0, linear: bool = False
    ):
        if not linear:
            hidden_dim = output_dim

            layers = [
                torch.nn.BatchNorm1d(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.Dropout(dropout),
            ]
            super().__init__(*layers)
        else:
            super().__init__(
                torch.nn.Linear(input_dim, output_dim),
            )


class Composer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        has_edge_attr: bool = True,
        linear: bool = True,
    ):
        super().__init__()
        concat_dim = 2 * embed_dim
        self.node_proj = MLP(concat_dim, embed_dim, linear=linear)

        if not has_edge_attr:
            self.edge_encoder = torch.nn.Embedding(2, embed_dim)

    def encode_edge_attributes(self, edge_index, edge_attr):
        if edge_attr is not None:
            return edge_attr
        e = torch.ones_like(edge_index[0])
        return self.edge_encoder(e.to(torch.long))

    def forward(self, x, edge_index, edge_attr, batch, token_index, token_attr=None):
        edge_attr = self.encode_edge_attributes(edge_index, edge_attr)
        edge_features = to_dense_adj(edge_index, batch, edge_attr)

        if token_attr is not None:
            token_attr = to_dense_adj(token_index, batch, token_attr)
            edge_features = torch.cat([edge_features, token_attr], -1)

        x = x[token_index.T].flatten(1, 2)
        x = self.node_proj(x)
        x = to_dense_adj(token_index, batch, x)
        x = x + edge_features
        return x


class Head(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        pooling: str = "set2set",
        activation: str = "relu",
        project_down=True,
    ):
        super().__init__()

        if pooling in ["sum", "mean"]:
            self.pooling = partial(scatter, reduce=pooling)
            in_dim = embed_dim
        elif pooling == "set2set":
            self.pooling = Set2Set(embed_dim, processing_steps=6)
            in_dim = embed_dim * 2
        elif pooling is None:
            self.pooling = None
            in_dim = embed_dim
        else:
            raise ValueError(f"Pooling {pooling} is not supported")

        if activation == "relu":
            act_fn = torch.nn.ReLU
        elif activation == "gelu":
            act_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} is not supported")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, embed_dim // 2 if project_down else embed_dim),
            act_fn(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(
                embed_dim // 2 if project_down else embed_dim,
                embed_dim // 4 if project_down else embed_dim,
            ),
            act_fn(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(embed_dim // 4 if project_down else embed_dim, output_dim),
        )

    def forward(self, x, data):
        if self.pooling is not None:
            x = self.pooling(x, data.batch)
        return self.mlp(x)


class Decomposer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        reduce_fn="sum",
    ):
        super().__init__()
        self.node_dim = embed_dim
        self.reduce_fn = reduce_fn

        self.out_proj = MLP(embed_dim, 2 * embed_dim)
        self.node_mlp = MLP(self.node_dim, embed_dim)

    def forward(self, x, node_features, node_batch, token_index):
        x = self.out_proj(x)

        dim_size = node_batch.size(0)
        node_features = torch.zeros_like(node_features)

        for i in range(2):
            features_order_i = x[:, i * self.node_dim : (i + 1) * self.node_dim]
            features_order_i = scatter(
                features_order_i,
                token_index[i],
                0,
                dim_size=dim_size,
                reduce=self.reduce_fn,
            )
            node_features = node_features + features_order_i

        return self.node_mlp(node_features)


def apply_mask_2d(node_features, node_batch):
    _, mask = to_dense_batch(node_features, node_batch)
    unbatch = mask.unsqueeze(2) * mask.unsqueeze(1)  # B x N x N
    mask = unbatch.unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)  # B x N x N x N
    return unbatch, mask


class FFN(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0,
        activation: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} is not supported")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 2),
            activation_fn(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

        if norm == "batch":
            self.norm = torch.nn.BatchNorm1d(embed_dim)
            self.norm_aggregate = torch.nn.BatchNorm1d(embed_dim)
        elif norm == "layer":
            self.norm = torch.nn.LayerNorm(embed_dim)
            self.norm_aggregate = torch.nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Norm {norm} is not supported")

        self.dropout_aggregate = torch.nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(self, x_prior, x):
        x = self.dropout_aggregate(x)
        x = x_prior + x
        x = self.norm_aggregate(x)
        x = self.mlp(x) + x
        return self.norm(x)


class EdgeAttention(torch.nn.Module):
    """Adapted from https://github.com/bergen/EdgeTransformer"""

    def __init__(self, embed_dim, num_heads, dropout, qkv_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(5)]
        )
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

        if qkv_norm:
            self.norms = torch.nn.ModuleList(
                [torch.nn.LayerNorm(self.d_k) for _ in range(3)]
            )

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k, right_k, left_v, right_v = [
            l(x) for l, x in zip(self.linears, (query, key, value, value))
        ]
        left_k = left_k.view(
            num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k
        )
        right_k = right_k.view(
            num_batches, key.size(1), key.size(2), self.num_heads, self.d_k
        )
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        if hasattr(self, "norms"):
            left_k = self.norms[0](left_k)
            right_k = self.norms[1](right_k)

        scores = torch.einsum("bxahd,bayhd->bxayh", left_k, right_k) / math.sqrt(
            self.d_k
        )

        if mask is not None:
            scores_dtype = scores.dtype
            scores = (
                scores.to(torch.float32)
                .masked_fill(mask.unsqueeze(4), -1e9)
                .to(scores_dtype)
            )

        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        val = torch.einsum("bxahd,bayhd->bxayhd", left_v, right_v)

        if hasattr(self, "norms"):
            val = self.norms[2](val)

        x = torch.einsum("bxayh,bxayhd->bxyhd", att, val)
        x = x.view(num_batches, num_nodes_q, num_nodes_k, self.embed_dim)

        return self.linears[-1](x)


class EdgeTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        activation: str = "relu",
        norm: str = "batch",
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attention = EdgeAttention(embed_dim, num_heads, attention_dropout)

        if norm_first:
            self.norm = torch.nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, dropout, activation, norm)

    def forward(self, x_in, mask=None):
        x = x_in

        if self.norm_first:
            x = self.norm(x)

        if mask is not None:
            x_upd = self.attention(x, x, x, ~mask)
        else:
            x_upd = self.attention(x, x, x)
        x = self.ffn(x_in, x_upd)
        return x


class EdgeTransformer(torch.nn.Module):
    def __init__(
        self,
        feature_encoder,
        num_layers,
        embed_dim,
        out_dim,
        num_heads,
        activation,
        pooling=None,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        head_project_down=True,
        has_edge_attr=False,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.composer = Composer(embed_dim, has_edge_attr)
        self.layers = torch.nn.ModuleList(
            [
                EdgeTransformerLayer(
                    embed_dim,
                    num_heads,
                    ffn_dropout,
                    attention_dropout,
                    norm="layer",
                    norm_first=True,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.decomposer = Decomposer(embed_dim)
        self.head = Head(
            embed_dim,
            out_dim,
            pooling=pooling,
            activation=activation,
            project_down=head_project_down,
        )

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data):
        data = self.feature_encoder(data)
        token_attr = data.token_attr if hasattr(data, "token_attr") else None
        token_index = data.token_index

        x = self.composer(
            data.x, data.edge_index, data.edge_attr, data.batch, token_index, token_attr
        )
        unbatch, mask = apply_mask_2d(data.x, data.batch)

        for layer in self.layers:
            x = layer(x, mask)

        x = x[unbatch]
        x = self.decomposer(x, data.x, data.batch, token_index)
        return self.head(x, data)
