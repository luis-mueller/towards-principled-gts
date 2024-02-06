import hydra
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.datasets import WebKB
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from edge_transformer import (
    EdgeTransformer,
    token_index_transform,
    FeatureEncoder,
    CosineWithWarmupLR,
    transform_dataset,
)
from loguru import logger


@hydra.main(version_base=None, config_path=".", config_name="webkb")
def main(cfg):
    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config={"dataset": cfg.dataset, **dict(cfg)},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    torch.set_float32_matmul_precision("medium")
    logger.info(f"Setting float32 matmul precision to medium")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"
    )
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float32 if dtype == "float32" else torch.bfloat16

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    dataset = WebKB(cfg.root, cfg.dataset)

    logger.info(f"Pre-transforming dataset")
    dataset = transform_dataset(dataset, token_index_transform)

    loader = DataLoader(dataset)

    feature_encoder = FeatureEncoder(
        node_encoder="linear",
        edge_encoder=None,
        node_dim=dataset.num_features,
        edge_dim=0,
        embed_dim=cfg.embed_dim,
    )

    model = EdgeTransformer(
        feature_encoder=feature_encoder,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        out_dim=dataset.num_classes,
        num_heads=cfg.num_heads,
        activation=cfg.activation,
        attention_dropout=cfg.attention_dropout,
        ffn_dropout=cfg.ffn_dropout,
        head_project_down=False,
        has_edge_attr=False,
    ).to(device)
    logger.info(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineWithWarmupLR(
        optimizer,
        10,
        lr=cfg.lr,
        lr_decay_iters=200,
        min_lr=0,
    )

    def train(fold):
        model.train()
        optimizer.zero_grad()
        data = next(iter(loader)).to(device)
        with torch.autocast(device, tdtype):
            logits = model(data)
        preds = logits[data.train_mask[:, fold]]
        targets = data.y[data.train_mask[:, fold]]
        loss = F.cross_entropy(preds, targets)
        loss.backward()
        if cfg.gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.gradient_norm
            )
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(fold, split):
        model.eval()
        data = next(iter(loader)).to(device)
        with torch.autocast(device, tdtype):
            logits = model(data)
        mask = data[f"{split}_mask"]
        preds = logits[mask[:, fold]].softmax(-1).argmax(-1)
        targets = data.y[mask[:, fold]]
        return (preds == targets).sum() / len(targets)

    best_val_acc = None
    logger.info(f"Starting training on fold {cfg.fold} for 200 epochs 🚀")
    for epoch in range(200):
        scheduler(epoch)
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train(cfg.fold)
        val_acc = test(cfg.fold, "val")

        if best_val_acc is None or val_acc >= best_val_acc:
            test_acc = test(cfg.fold, "test")
            best_val_acc = val_acc

        if cfg.wandb_project:
            wandb.log(
                {
                    "lr": lr,
                    "loss": loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "test_acc": test_acc,
                }
            )

        logger.info(
            f"Epoch: {epoch} LR: {lr:.5f} Loss: {loss:.5f} Val. acc. {val_acc:.5f} Test acc. {test_acc:.5f}"
        )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
