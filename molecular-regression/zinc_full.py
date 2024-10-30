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
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


def save_checkpoint(checkpoint_file, step, model, optimizer, scaler, best_val_score):
    logger.info(f"Creating and saving checkpoint to {checkpoint_file}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "best_val_score": best_val_score,
        },
        checkpoint_file,
    )


def continue_from_checkpoint(checkpoint_file, model, optimizer, scaler, device_id=None):
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_file}")
        load_args = (
            dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}
        )
        checkpoint = torch.load(checkpoint_file, **load_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint["step"] + 1, checkpoint["best_val_score"]
    else:
        logger.info(
            f"Could not find checkpoint {checkpoint_file}, starting training from scratch"
        )
        return 0, None


def ddp_setup():
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            master_process = device_id == 0
        else:
            device_id = 0
            master_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        master_process = True

    return device, device_id, device_count, master_process


@hydra.main(version_base=None, config_path=".", config_name="zinc_full")
def main(cfg):
    ddp_setup()

    device, device_id, device_count, master_process = accelerator_setup()

    if cfg.wandb_project and master_process:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config={"dataset": "zinc", **dict(cfg)},
        )

    logger.info(f"Accelerator: {device}")

    dtype = cfg.dtype
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float32 if dtype == "float32" else torch.bfloat16
    ctx = torch.autocast(device_type=device, dtype=tdtype, enabled=True)

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    if cfg.rrwp:
        pe_kwargs = dict(num_iter=21)
        transform = Compose([token_index_transform, RRWPTransform(**pe_kwargs)])
    else:
        transform = token_index_transform

    train_dataset = ZINC(cfg.root, subset=False, split="train").shuffle()
    val_dataset = ZINC(cfg.root, subset=False, split="val").shuffle()
    test_dataset = ZINC(cfg.root, subset=False, split="test").shuffle()

    train_dataset = transform_dataset(train_dataset, transform)
    val_dataset = transform_dataset(val_dataset, transform)
    test_dataset = transform_dataset(test_dataset, transform)

    if device_count > 1:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size // device_count,
            shuffle=False,
            num_workers=cfg.num_workers,
            sampler=DistributedSampler(train_dataset),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

    if master_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

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
        compiled=cfg.compiled,
    ).to(device)
    logger.info(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    if cfg.checkpoint is not None:
        checkpoint_file = f"{cfg.root}/{cfg.checkpoint}.pt"
        logger.info(f"Trying to continue from checkpoint {checkpoint_file}")
        start_epoch, best_val_error = continue_from_checkpoint(
            checkpoint_file, model, optimizer, scaler, device_id
        )
    else:
        start_epoch, best_val_error = 0, None

    scheduler = CosineWithWarmupLR(
        optimizer, 50, lr=cfg.lr, lr_decay_iters=cfg.num_epochs, min_lr=0
    )

    if device_count > 1:
        logger.info("Creating DDP module")
        model = DDP(model, device_ids=[device_id])

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.squeeze()
            optimizer.zero_grad()
            with ctx:
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
            with ctx:
                error += (model(data).squeeze() - data.y).abs().sum().item()
        return error / len(loader.dataset)

    logger.info(f"Starting training for {cfg.num_epochs - start_epoch} epochs 🚀")
    for epoch in range(start_epoch, cfg.num_epochs):
        if device_count > 1:
            train_loader.sampler.set_epoch(epoch)
        scheduler(epoch)
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = train() * device_count

        if master_process:
            val_error = test(val_loader)

            if best_val_error is None or val_error <= best_val_error:
                test_error = test(test_loader)
                best_val_error = val_error

                if cfg.checkpoint is not None:
                    module = model.module if device_count > 1 else model
                    save_checkpoint(
                        checkpoint_file,
                        epoch,
                        module,
                        optimizer,
                        scaler,
                        best_val_error,
                    )

            if cfg.wandb_project:
                wandb.log(
                    {
                        "lr": lr,
                        "loss": loss,
                        "val_error": val_error,
                        "test_error": test_error,
                    },
                    step=epoch,
                )

            logger.info(
                "Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, "
                "Test MAE: {:.7f}".format(epoch, lr, loss, val_error, test_error)
            )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project and master_process:
        wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
