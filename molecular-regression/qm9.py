import hydra
import torch
import wandb
from torch_geometric.datasets import QM9
from torch_geometric.utils import remove_self_loops
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from edge_transformer import (
    EdgeTransformer,
    token_index_transform,
    FeatureEncoder,
    RRWPTransform,
    CosineWithWarmupLR,
    transform_dataset,
)
from loguru import logger


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def load_dataset(root: str):
    return QM9(f"{root}/qm9")


@hydra.main(version_base=None, config_path=".", config_name="qm9")
def main(cfg):
    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config={"dataset": "qm9", **dict(cfg)},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    dataset = load_dataset(cfg.root)
    dataset.data.y = dataset.data.y[:, 0:12]
    dataset = dataset.shuffle()

    transform = [Complete(), T.Distance(norm=False), token_index_transform]
    pe_kwargs = {}

    if cfg.rrwp:
        pe_kwargs = dict(num_iter=21)
        transform.append(RRWPTransform(**pe_kwargs))

    transform = T.Compose(transform)

    logger.info(f"Pre-transforming dataset")
    dataset = transform_dataset(dataset, transform)

    tenpercent = int(len(dataset) * 0.1)
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent : 2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent :].shuffle()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    if cfg.rrwp:
        feature_encoder = FeatureEncoder(
            node_encoder="linear",
            edge_encoder="linear",
            node_dim=11,
            edge_dim=5,
            embed_dim=cfg.embed_dim,
            edge_positional_encoder="rrwp",
            edge_positional_dim=32,
            edge_positional_encoder_kwargs=pe_kwargs,
        )
    else:
        feature_encoder = FeatureEncoder(
            node_encoder="linear",
            edge_encoder="linear",
            node_dim=11,
            edge_dim=5,
            embed_dim=cfg.embed_dim,
        )

    model = EdgeTransformer(
        feature_encoder=feature_encoder,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        out_dim=12,
        num_heads=cfg.num_heads,
        activation=cfg.activation,
        pooling=cfg.pooling,
        attention_dropout=cfg.attention_dropout,
        ffn_dropout=cfg.ffn_dropout,
        has_edge_attr=True,
    ).to(device)
    logger.info(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineWithWarmupLR(
        optimizer,
        5,
        lr=cfg.lr,
        lr_decay_iters=200,
        min_lr=0,
    )

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            if cfg.gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        return error.mean().item()

    best_val_error = None
    logger.info(f"Starting training for 200 epochs 🚀")
    for epoch in range(200):
        scheduler(epoch)
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train()
        val_error = test(val_loader)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        if cfg.wandb_project:
            wandb.log(
                {
                    "lr": lr,
                    "loss": loss,
                    "val_error": val_error,
                    "test_error": test_error,
                }
            )

        logger.info(
            f"Epoch: {epoch} LR: {lr:.5f} Loss: {loss:.5f} Val. error {val_error:.5f} Test error {test_error:.5f}"
        )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
