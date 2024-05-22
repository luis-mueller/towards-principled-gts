import hydra
import torch
import math
import wandb
import time
import torch_geometric
from torch_geometric.seed import seed_everything
import torch.nn.functional as F
from brec.dataset import BRECDataset
from torch_geometric.utils import to_dense_adj
from loguru import logger
from tqdm import tqdm
from torch.nn import CosineEmbeddingLoss

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023


def get_model(cfg, device):
    model = EdgeTransformer(cfg.num_layers, cfg.embed_dim, cfg.num_heads).to(device)
    return model


def evaluate(dataset, cfg, device):
    def T2_calculation(dataset, model, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    part_dict = {
        "Basic": (0, 60),
        "Regular": (60, 110),
        "Extension": (160, 260),
        "CFI": (260, 320),
        # "4-Vertex_Condition": (360, 380),
        # "Distance_Regular": (380, 400),
    }

    global BATCH_SIZE
    BATCH_SIZE = cfg.batch_size

    time_start = time.process_time()

    # Do something
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            model = get_model(cfg, device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM)
                * NUM_RELABEL
                * 2 : (id + SAMPLE_NUM + 1)
                * NUM_RELABEL
                * 2
            ]
            model.train()
            for _ in range(EPOCH):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=BATCH_SIZE
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                # logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    # logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, model, True)
            T_square_reliability = T2_calculation(dataset_reliability, model, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

            if cfg.wandb_project is not None:
                wandb.log(
                    {
                        f"{part_name}/correct": cnt_part,
                        "total_correct": cnt,
                    }
                )

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)


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

    if cfg.wandb_project is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BRECDataset(root=cfg.root)
    evaluate(dataset, cfg, device)


if __name__ == "__main__":
    main()
