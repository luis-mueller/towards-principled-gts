import hydra
import torch
import wandb
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.seed import seed_everything
from edge_transformer import (
    EdgeTransformer,
    token_index_transform,
    FeatureEncoder,
    RRWPTransform,
    CosineWithWarmupLR,
    transform_dataset,
)
from loguru import logger


@hydra.main(version_base=None, config_path=".", config_name="zinc")
def main(cfg):
    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config={"dataset": "zinc", **dict(cfg)},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    if cfg.rrwp:
        pe_kwargs = dict(num_iter=21)
        transform = Compose([token_index_transform, RRWPTransform(**pe_kwargs)])
    else:
        transform = token_index_transform

    train_dataset = ZINC(cfg.root, subset=True, split="train").shuffle()
    val_dataset = ZINC(cfg.root, subset=True, split="val").shuffle()
    test_dataset = ZINC(cfg.root, subset=True, split="test").shuffle()

    train_dataset = transform_dataset(train_dataset, transform)
    val_dataset = transform_dataset(val_dataset, transform)
    test_dataset = transform_dataset(test_dataset, transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if cfg.rrwp:
        feature_encoder = FeatureEncoder(
            node_encoder="embedding",
            edge_encoder="embedding",
            node_dim=28,
            edge_dim=4,
            embed_dim=cfg.embed_dim,
            edge_positional_encoder="rrwp",
            edge_positional_dim=32,
            edge_positional_encoder_kwargs=pe_kwargs,
        )
    else:
        feature_encoder = FeatureEncoder(
            node_encoder="embedding",
            edge_encoder="embedding",
            node_dim=28,
            edge_dim=4,
            embed_dim=cfg.embed_dim,
        )

    model = EdgeTransformer(
        feature_encoder=feature_encoder,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        out_dim=1,
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
        optimizer, 50, lr=cfg.lr, lr_decay_iters=2000, min_lr=0
    )

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.squeeze()
            optimizer.zero_grad()
            loss = lf(model(data).squeeze(), data.y)
            loss.backward()
            if cfg.gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            data.x = data.x.squeeze()
            error += (model(data).squeeze() - data.y).abs().sum().item()
        return error / len(loader.dataset)

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
            "Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, "
            "Test MAE: {:.7f}".format(epoch, lr, loss, val_error, test_error)
        )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
