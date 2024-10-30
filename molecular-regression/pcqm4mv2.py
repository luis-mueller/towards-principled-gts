import os
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4MEvaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import hydra
import torch
import torch.nn.functional as F
from loguru import logger
import wandb
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from prettytable import PrettyTable
import tqdm
import inspect
import torch_geometric.transforms as T
from edge_transformer import (
    EdgeTransformer,
    token_index_transform,
    FeatureEncoder,
    RRWPTransform,
)
import math


class CosineWithWarmupLR:
    """Adapted from https://github.com/karpathy/nanoGPT"""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
        epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.epoch = epoch
        self.step()

    def step(self):
        self.epoch += 1
        lr = self._get_lr(self.epoch)
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


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay,
    learning_rate,
    betas,
    device_type,
):
    """Adapted from https://github.com/karpathy/nanoGPT"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() >= 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    nodecay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() < 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {use_fused}")

    return optimizer


def count_parameters(model: torch.nn.Module):
    """Source: https://stackoverflow.com/a/62508086"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.info(f"\n{str(table)}")
    return total_params


def ensure_root_folder(root, master_process=True):
    if not os.path.exists(root) and master_process:
        logger.info(f"Creating root directory {root}")
        os.makedirs(root)

    if not os.path.exists(data_dir := f"{root}/data") and master_process:
        logger.info(f"Creating data directory {data_dir}")
        os.makedirs(data_dir)

    if not os.path.exists(ckpt_dir := f"{root}/ckpt") and master_process:
        logger.info(f"Creating ckpt directory {ckpt_dir}")
        os.makedirs(ckpt_dir)

    return data_dir, ckpt_dir


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
        checkpoint = torch.load(checkpoint_file, weights_only=True, **load_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint["step"] + 1, checkpoint["best_val_score"]
    else:
        logger.info(
            f"Could not find checkpoint {checkpoint_file}, starting training from scratch"
        )
        return 1, None


@torch.no_grad()
def evaluate(evaluator, model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(
                -1,
            )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


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


@hydra.main(version_base=None, config_path="./", config_name="pcqm4mv2")
def main(cfg):
    ddp_setup()

    device, device_id, device_count, master_process = accelerator_setup()
    logger.info(f"Accelerator: {device}, num. devices {device_count}")

    data_dir, ckpt_dir = ensure_root_folder(cfg.root, master_process)

    if cfg.wandb_project is not None and master_process:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    torch.set_float32_matmul_precision("medium")
    logger.info(f"Setting float32 matmul precision to medium")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float16 if dtype == "float16" else torch.bfloat16

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    transform = [token_index_transform]
    pe_kwargs = {}

    if cfg.rrwp:
        pe_kwargs = dict(num_iter=cfg.rrwp_iter)
        transform.append(RRWPTransform(**pe_kwargs, self_loops=True))

    transform = T.Compose(transform)

    logger.info(f"Loading dataset from {data_dir}")
    dataset = PygPCQM4Mv2Dataset(root=data_dir, transform=transform)
    logger.info("Dataset loaded")

    split_idx = dataset.get_idx_split()

    if device_count > 1:
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=cfg.batch_size // device_count,
            num_workers=cfg.num_workers,
            shuffle=False,
            sampler=DistributedSampler(dataset[split_idx["train"]]),
        )
    else:
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )

    if master_process:
        val_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=64,
            num_workers=cfg.num_workers,
        )
        evaluator = PCQM4MEvaluator()

    logger.info("Creating transformer")
    if cfg.rrwp:
        edge_positional_dim = cfg.edge_positional_dim
        atom_encoder = AtomEncoder(emb_dim=cfg.embed_dim)
        bond_encoder = BondEncoder(emb_dim=cfg.embed_dim - edge_positional_dim)
        feature_encoder = FeatureEncoder(
            embed_dim=cfg.embed_dim,
            node_encoder=atom_encoder,
            edge_encoder=bond_encoder,
            edge_positional_encoder="rrwp",
            edge_positional_dim=edge_positional_dim,
            edge_positional_encoder_kwargs=pe_kwargs,
        )
    else:
        atom_encoder = AtomEncoder(emb_dim=cfg.embed_dim)
        bond_encoder = BondEncoder(emb_dim=cfg.embed_dim)
        feature_encoder = FeatureEncoder(
            embed_dim=cfg.embed_dim,
            node_encoder=atom_encoder,
            edge_encoder=bond_encoder,
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

    if master_process:
        num_params = count_parameters(model)

        if cfg.wandb_project is not None and master_process:
            wandb.log(dict(num_params=num_params))

        logger.info(model)
        logger.info(f"Number of parameters: {num_params}")

    optimizer = configure_optimizers(
        model,
        cfg.weight_decay,
        cfg.lr,
        (0.9, 0.95),
        device,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    if cfg.checkpoint is not None:
        checkpoint_file = f"{ckpt_dir}/{cfg.checkpoint}.pt"
        logger.info(f"Trying to continue from checkpoint {checkpoint_file}")
        step, best_val_score = continue_from_checkpoint(
            checkpoint_file, model, optimizer, scaler, device_id
        )
    else:
        step, best_val_score = 1, None

    scheduler = CosineWithWarmupLR(
        optimizer, cfg.num_warmup_steps, cfg.lr, cfg.num_steps, 0.0, step - 1
    )

    if master_process:
        logger.info(f"Optimizer + scheduler with lr {cfg.lr} ready")

    if device_count > 1:
        logger.info("Creating DDP module")
        model = DDP(model, device_ids=[device_id])

    logger.info(
        f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀"
    )

    epoch_after = len(dataset[split_idx["train"]]) // cfg.batch_size
    logger.info(f"Epoch after: {epoch_after}")

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)
    while step <= cfg.num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device_id)
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                logits = model(batch)
                loss = F.l1_loss(logits.squeeze(), batch.y)
            scaler.scale(loss).backward()

            if cfg.gradient_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if master_process:
                loss_window.append(float(loss.detach().cpu()))

            if step % int(cfg.log_after) == 0 and master_process:
                if cfg.wandb_project is not None and master_process:
                    wandb.log(
                        dict(
                            train_loss=sum(loss_window) / len(loss_window),
                            lr=optimizer.param_groups[0]["lr"],
                            time=time.time() - start_time,
                        ),
                        step=step,
                    )
                loss_window = []
                start_time = time.time()

            if step % int(epoch_after) == 0:
                logger.info(f"Completed epoch [{device_id}]")
                if device_count > 1:
                    epoch += 1
                    train_loader.sampler.set_epoch(epoch)

            if step % int(cfg.val_after) == 0:
                if master_process:
                    logger.info("Evaluating model")
                    model.eval()
                    val_score = evaluate(evaluator, model, val_loader, device_id)
                    if best_val_score is None or val_score < best_val_score:
                        best_val_score = val_score

                        if cfg.checkpoint is not None:
                            module = model.module if device_count > 1 else model
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                module,
                                optimizer,
                                scaler,
                                best_val_score,
                            )

                    logger.info(
                        dict(
                            step=step,
                            val_score=val_score,
                            best_val_score=best_val_score,
                        )
                    )
                    if cfg.wandb_project is not None and master_process:
                        wandb.log(
                            dict(val_score=val_score, best_val_score=best_val_score),
                            step=step,
                        )
                    model.train()

            step += 1

    logger.info(f"Training complete 🥳")
    if cfg.wandb_project is not None and master_process:
        wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
