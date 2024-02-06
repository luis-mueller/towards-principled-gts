import hydra
import torch
import math
import torch.nn.functional as F
from brec.dataset import BRECDataset
from brec.evaluator import evaluate
from torch_geometric.utils import to_dense_adj


def diag_offdiag_maxpool(input):
    N = input.shape[-1]
    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]

    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)

    val = torch.abs(torch.add(max_val, min_val))
    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)
    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]

    return torch.cat((max_diag, max_offdiag), dim=1)


class EdgeTransformerLayer(torch.nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads=1):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_dim, embed_dim)] * 4)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.head_dim)] * 3)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(in_dim + embed_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        B, N, _, _ = x.shape

        q, k, v1, v2 = (
            self.linears[i](x).view(B, N, N, -1, self.head_dim) for i in range(4)
        )
        q = self.norms[0](q)
        k = self.norms[1](k)

        scores = torch.einsum("bxahd,bayhd->bxayh", q, k) / math.sqrt(self.head_dim)
        att = F.softmax(scores, dim=2)

        x_upd = torch.einsum("bxahd,bayhd->bxayhd", v1, v2)
        x_upd = self.norms[2](x_upd)
        x_upd = torch.einsum("bxayh,bxayhd->bxyhd", att, x_upd).view(B, N, N, -1)

        x = torch.cat([x, x_upd], -1)
        return self.ffn(x)


class EdgeTransformer(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads=1):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                EdgeTransformerLayer(embed_dim if i > 0 else 1, embed_dim, num_heads)
                for i in range(num_layers)
            ]
        )
        self.out_projections = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim * 2, embed_dim // 2) for _ in range(num_layers)]
        )
        self.bn = torch.nn.BatchNorm1d(embed_dim // 2, momentum=1.0, affine=False)
        self.reset_parameters()

    def forward(self, data):
        edge_attr = torch.ones(
            data.edge_index.size(1), device=data.edge_index.device, dtype=torch.float
        )
        x = to_dense_adj(data.edge_index, data.batch, edge_attr)
        x[:, range(x.size(1)), range(x.size(2))] = 2.0
        x = x.unsqueeze(-1)

        scores = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            scores = (
                self.out_projections[i](diag_offdiag_maxpool(x.permute(0, 3, 1, 2)))
                + scores
            )

        scores = self.bn(scores)
        return scores

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


@hydra.main(version_base=None, config_path=".", config_name="brec")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BRECDataset(root=cfg.root)
    model = EdgeTransformer(5, 32, 4).to(device)
    evaluate(dataset, model, device, cfg.log_file)


if __name__ == "__main__":
    main()
