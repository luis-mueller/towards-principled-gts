import time
import os
import hydra
import torch
import pandas as pd
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Data, Batch
from loguru import logger
from edge_transformer import (
    token_index_transform,
    EdgeAttention,
)


class EdgeTransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.lin = torch.nn.Linear(2 * embed_dim, embed_dim)
        self.attention = EdgeAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.lin(x)
        x = x + self.attention(x, x, x, None)
        return self.norm(x + self.ffn(x))


@hydra.main(version_base=None, config_path=".", config_name="timing")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    torch.set_float32_matmul_precision("medium")
    logger.info(f"Setting float32 matmul precision to medium")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if cfg.model_name == "ET":
        model = EdgeTransformerLayer(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(device)
    elif cfg.model_name == "ET (+Triton)":
        model = EdgeTransformerLayer(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(device)
        model = torch.compile(model)
    else:
        raise ValueError(f"Model {cfg.model_name} is not supported")
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()

    results = None
    for it in range(cfg.repeats + 1):
        batch = Batch.from_data_list(
            [
                token_index_transform(
                    Data(
                        torch.randn((cfg.num_nodes, cfg.embed_dim)),
                        erdos_renyi_graph(cfg.num_nodes, cfg.edge_prob),
                    )
                )
                for _ in range(cfg.batch_size)
            ]
        ).to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                x = batch.x[batch.token_index.T].flatten(1, 2)
                x = to_dense_adj(batch.token_index, batch.batch, x)
                out = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

        # NOTE: Ignore first iteration (warm-up)
        if it > 0:
            timing_results = pd.DataFrame([{"elapsed": elapsed, **dict(cfg)}])
            if results is None:
                results = timing_results
            else:
                results = pd.concat([results, timing_results])

    if cfg.results_path is not None:
        if not os.path.exists(cfg.results_path):
            os.makedirs(cfg.results_path)

        if not os.path.exists(results_file := f"{cfg.results_path}/results.csv"):
            results.to_csv(results_file)
        else:
            results.to_csv(results_file, mode="a", header=False)


if __name__ == "__main__":
    main()
