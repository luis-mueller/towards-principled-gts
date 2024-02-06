import hydra
import torch
import wandb
from torch_geometric.datasets import TUDataset
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from edge_transformer import (
    EdgeTransformer,
    token_index_transform,
    FeatureEncoder,
    RRWPTransform,
    CosineWithWarmupLR,
    transform_dataset,
)
from loguru import logger


def load_dataset(root: str):
    infile = open("molecular-regression/train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("molecular-regression/val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("molecular-regression/test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    return TUDataset(f"{root}/alchemy", name="alchemy_full")[indices]


@hydra.main(version_base=None, config_path=".", config_name="alchemy")
def main(cfg):
    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config={"dataset": "alchemy", **dict(cfg)},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    dataset = load_dataset(cfg.root)

    transform = [token_index_transform]
    pe_kwargs = {}

    if cfg.rrwp:
        pe_kwargs = dict(num_iter=20)
        transform.append(RRWPTransform(**pe_kwargs))

    transform = Compose(transform)

    logger.info(f"Pre-transforming dataset")
    dataset = transform_dataset(dataset, transform)

    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    train_dataset = dataset[0:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    if cfg.rrwp:
        feature_encoder = FeatureEncoder(
            node_encoder="linear",
            edge_encoder="linear",
            node_dim=6,
            edge_dim=4,
            embed_dim=cfg.embed_dim,
            edge_positional_encoder="rrwp",
            edge_positional_dim=16,
            edge_positional_encoder_kwargs=pe_kwargs,
        )
    else:
        feature_encoder = FeatureEncoder(
            node_encoder="linear",
            edge_encoder="linear",
            node_dim=6,
            edge_dim=4,
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineWithWarmupLR(
        optimizer,
        50,
        lr=cfg.lr,
        lr_decay_iters=2000,
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
    logger.info(f"Starting training for 2000 epochs 🚀")
    for epoch in range(2000):
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
