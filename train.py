import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@dataclass
class TrainConfig:
    emb_size: int = 384
    head_size: int = 8
    n_layer: int = 6
    sequence_len: int = 128
    learning_rate: float = 3e-4
    eval_iters: int = 3
    batch_size: int = 100
    dropout: float = 0.4
    seed: int = 12046
    dataset_path: str = "./my_novel_dataset"
    tokenizer_path: str = "./my_novel_tokenizer"
    test_size: float = 0.1
    split_seed: int = 1024
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    input_corrupt_prob: float = 0.0
    sample_stride: int = 1
    prefix_trunc_prob: float = 0.0
    min_prefix_len: int = 0
    # 学习率调度：none 表示不变；cosine 为每 epoch 一次的余弦退火（T_max 见下）
    lr_scheduler: str = "none"
    lr_cosine_t_max_epochs: int = 0
    lr_cosine_eta_min: float = 1e-6


def get_config_by_mode(mode_size: int) -> TrainConfig:
    configs: Dict[int, TrainConfig] = {
        1: TrainConfig(),
        # 13万字语料优先降模型容量，并开启轻量正则防过拟合
        2: TrainConfig(
            emb_size=192,
            head_size=32,
            n_layer=4,
            sequence_len=128,
            learning_rate=3e-4,
            batch_size=128,
            dropout=0.2,
            weight_decay=0.1,
            label_smoothing=0.05,
            grad_clip=1.0,
            input_corrupt_prob=0.08,
            sample_stride=4,
            prefix_trunc_prob=0.6,
            min_prefix_len=32,
        ),
        3: TrainConfig(
            emb_size=256,
            head_size=32,
            n_layer=4,
            learning_rate=3e-4,
            batch_size=96,
            dropout=0.2,
            weight_decay=0.08,
            label_smoothing=0.05,
            grad_clip=1.0,
            input_corrupt_prob=0.06,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
        ),
        4: TrainConfig(
            emb_size=256,
            head_size=32,
            n_layer=4,
            learning_rate=3e-4,
            batch_size=160,
            dropout=0.25,
            weight_decay=0.1,
            label_smoothing=0.08,
            grad_clip=1.0,
            input_corrupt_prob=0.08,
            sample_stride=4,
            prefix_trunc_prob=0.6,
            min_prefix_len=32,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
        ),
        5: TrainConfig(
            emb_size=256,
            head_size=32,
            n_layer=4,
            learning_rate=3e-4,
            batch_size=192,
            dropout=0.28,
            weight_decay=0.12,
            label_smoothing=0.1,
            grad_clip=1.0,
            input_corrupt_prob=0.1,
            sample_stride=6,
            prefix_trunc_prob=0.7,
            min_prefix_len=40,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
        ),
        # 略增容量 + 略降峰值 LR + 略松前缀截断 + 余弦退火（按本次 --epochs 为周期）
        6: TrainConfig(
            emb_size=288,
            head_size=32,
            n_layer=5,
            learning_rate=2.5e-4,
            batch_size=176,
            dropout=0.26,
            weight_decay=0.11,
            label_smoothing=0.09,
            grad_clip=1.0,
            input_corrupt_prob=0.09,
            sample_stride=6,
            prefix_trunc_prob=0.62,
            min_prefix_len=36,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
            lr_scheduler="cosine",
            lr_cosine_t_max_epochs=0,
            lr_cosine_eta_min=1e-6,
        ),
        # 基于参数组6的温和回调版：略放松防背诵强度，尝试压低 best_test
        7: TrainConfig(
            emb_size=288,
            head_size=32,
            n_layer=5,
            learning_rate=2.5e-4,
            batch_size=176,
            dropout=0.24,
            weight_decay=0.1,
            label_smoothing=0.08,
            grad_clip=1.0,
            input_corrupt_prob=0.08,
            sample_stride=4,
            prefix_trunc_prob=0.52,
            min_prefix_len=32,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
            lr_scheduler="cosine",
            lr_cosine_t_max_epochs=0,
            lr_cosine_eta_min=1e-6,
        ),
        # 基于参数组6思路：降低模型容量 + 提高评估稳定性（eval_iters=5）
        8: TrainConfig(
            emb_size=256,
            head_size=32,
            n_layer=4,
            sequence_len=128,
            learning_rate=2.5e-4,
            eval_iters=5,
            batch_size=192,
            dropout=0.26,
            weight_decay=0.11,
            label_smoothing=0.09,
            grad_clip=1.0,
            input_corrupt_prob=0.09,
            sample_stride=6,
            prefix_trunc_prob=0.65,
            min_prefix_len=36,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
            lr_scheduler="cosine",
            lr_cosine_t_max_epochs=0,
            lr_cosine_eta_min=1e-6,
        ),
        # 基于参数组6的微调版：目标冲击 test_loss < 6.0
        9: TrainConfig(
            emb_size=288,
            head_size=32,
            n_layer=5,
            sequence_len=128,
            learning_rate=2.3e-4,
            eval_iters=5,
            batch_size=176,
            dropout=0.26,
            weight_decay=0.11,
            label_smoothing=0.09,
            grad_clip=1.0,
            input_corrupt_prob=0.09,
            sample_stride=6,
            prefix_trunc_prob=0.66,
            min_prefix_len=36,
            dataset_path="./my_novel_dataset_v2",
            tokenizer_path="./my_novel_tokenizer_v2",
            lr_scheduler="cosine",
            lr_cosine_t_max_epochs=0,
            lr_cosine_eta_min=1e-6,
        ),
    }
    if mode_size not in configs:
        raise ValueError(f"未配置 mode_size={mode_size}，请先在 get_config_by_mode 中添加对应参数组。")
    return configs[mode_size]


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def process(data, tokenizer, sequence_len: int, sample_stride: int = 1):
    text = data["text"]
    inputs, labels = [], []
    stride = max(1, int(sample_stride))
    for t in text:
        enc = tokenizer.encode(t)
        enc += [tokenizer.end_ind]
        if len(enc) <= sequence_len:
            continue
        for i in range(0, len(enc) - sequence_len, stride):
            inputs.append(enc[i : i + sequence_len])
            labels.append(enc[i + 1 : i + 1 + sequence_len])
    return {"inputs": inputs, "labels": labels}


def corrupt_inputs(inputs: torch.Tensor, tokenizer, prob: float) -> torch.Tensor:
    if prob <= 0:
        return inputs
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return inputs
    mask = torch.rand_like(inputs, dtype=torch.float) < prob
    corrupted = inputs.clone()
    corrupted[mask] = unk_id
    return corrupted


def random_prefix_truncate_inputs(
    inputs: torch.Tensor,
    tokenizer,
    trunc_prob: float,
    min_prefix_len: int,
) -> torch.Tensor:
    """
    随机前缀截断训练：按样本随机抹去前缀，迫使模型在不完整上下文下学习。
    通过将前缀替换为 unk token 保持张量形状不变。
    """
    if trunc_prob <= 0:
        return inputs
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return inputs

    bsz, t = inputs.shape
    min_len = max(1, min(int(min_prefix_len), t))
    out = inputs.clone()
    apply_mask = torch.rand(bsz, device=inputs.device) < trunc_prob
    if not torch.any(apply_mask):
        return out

    keep_lens = torch.randint(low=min_len, high=t + 1, size=(bsz,), device=inputs.device)
    for i in range(bsz):
        if apply_mask[i]:
            drop_len = t - int(keep_lens[i].item())
            if drop_len > 0:
                out[i, :drop_len] = unk_id
    return out


def attention(query, key, value, mask=None):
    _, _, h = query.shape
    scores = query @ key.transpose(-2, -1) / (h**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    w_att = F.softmax(scores, dim=-1)
    out = w_att @ value
    return out


class MaskedAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, sequence_len: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        mask = self.tril[:t, :t]
        out = attention(q, k, v, mask)
        return self.dp(out)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, sequence_len: int, dropout: float):
        super().__init__()
        n_head = emb_size // head_size
        self.heads = nn.ModuleList(
            [MaskedAttention(emb_size, head_size, sequence_len, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(emb_size, emb_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        return self.dp(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.Linear(emb_size, 4 * emb_size)
        self.ln2 = nn.Linear(4 * emb_size, emb_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        out = F.gelu(self.ln1(x))
        return self.dp(self.ln2(out))


class Block(nn.Module):
    def __init__(self, emb_size: int, head_size: int, sequence_len: int, dropout: float):
        super().__init__()
        self.l1 = nn.LayerNorm(emb_size)
        self.mha = MaskedMultiHeadAttention(emb_size, head_size, sequence_len, dropout)
        self.l2 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size, dropout)

    def forward(self, x):
        x = x + self.mha(self.l1(x))
        x = x + self.ff(self.l2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vs: int, config: TrainConfig):
        super().__init__()
        self.token_emb = nn.Embedding(vs, config.emb_size)
        self.pos_emb = nn.Embedding(config.sequence_len, config.emb_size)
        block = [
            Block(config.emb_size, config.head_size, config.sequence_len, config.dropout)
            for _ in range(config.n_layer)
        ]
        self.blocks = nn.Sequential(*block)
        self.l = nn.LayerNorm(config.emb_size)
        self.lm = nn.Linear(config.emb_size, vs)

    def forward(self, x):
        _, t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(pos)
        h = token_embeddings + position_embeddings
        h = self.blocks(h)
        logits = self.lm(self.l(h))
        return logits


@torch.no_grad()
def estimate_loss(model, train_loader, test_loader, eval_iters: int) -> Dict[str, float]:
    model.eval()
    out = {
        "train": _loss(model, train_loader, eval_iters),
        "test": _loss(model, test_loader, eval_iters),
    }
    model.train()
    return out


@torch.no_grad()
def _loss(model, data_loader, eval_iters: int) -> float:
    losses = []
    data_iter = iter(data_loader)
    for _ in range(eval_iters):
        data = next(data_iter, None)
        if data is None:
            data_iter = iter(data_loader)
            data = next(data_iter, None)
        inputs, labels = data["inputs"], data["labels"]
        logits = model(inputs)
        loss = F.cross_entropy(logits.transpose(-2, -1), labels).item()
        losses.append(loss)
    return float(torch.tensor(losses).mean().item())


def plot_curve(values: List[float], title: str, y_label: str, save_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step" if "Train" in title else "Epoch")
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def get_next_run_id(base_dir: Path, mode_size: int) -> int:
    pattern = f"train_{mode_size}_"
    ids = []
    if base_dir.exists():
        for p in base_dir.iterdir():
            if p.is_dir() and p.name.startswith(pattern):
                try:
                    ids.append(int(p.name.split("_")[-1]))
                except ValueError:
                    continue
    return max(ids, default=0) + 1


def get_resume_dir(base_dir: Path, mode_size: int, remuse: int) -> Path:
    run_dir = base_dir / f"train_{mode_size}_{remuse}"
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到续训目录: {run_dir}")
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"续训目录缺少 meta.json: {run_dir}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    old_mode_size = int(meta.get("mode_size", -1))
    if old_mode_size != mode_size:
        raise ValueError(f"参数组不一致：当前 mode_size={mode_size}，待续训目录 mode_size={old_mode_size}")
    return run_dir


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    epochs_this_run: int,
):
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "cosine":
        t_max = config.lr_cosine_t_max_epochs if config.lr_cosine_t_max_epochs > 0 else max(1, epochs_this_run)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config.lr_cosine_eta_min,
        )
    raise ValueError(f"未知 lr_scheduler: {config.lr_scheduler}")


def main():
    parser = argparse.ArgumentParser(description="训练 CharGPT 并保存训练产物")
    parser.add_argument("--mode_size", type=int, default=1, help="参数组编号，默认1")
    parser.add_argument("--epochs", type=int, default=10, help="本次训练轮数，默认10")
    parser.add_argument("--remuse", "--resume", dest="remuse", type=int, default=0, help="0重训；非0表示从 train_modeSize_remuse 的best权重续训")
    parser.add_argument(
        "--resume_from",
        type=str,
        default="test",
        choices=["test", "train"],
        help="续训时从哪类最优权重恢复：test=best_test_model.pth, train=best_train_model.pth",
    )
    args = parser.parse_args()

    config = get_config_by_mode(args.mode_size)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = Path("my_model")
    root_dir.mkdir(parents=True, exist_ok=True)
    run_id = get_next_run_id(root_dir, args.mode_size)
    run_dir = root_dir / f"train_{args.mode_size}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(run_dir)
    logger.info("device: %s", device)
    logger.info("mode_size: %s", args.mode_size)
    logger.info("epochs: %s", args.epochs)
    logger.info("resume_from_type: %s", args.resume_from)
    logger.info("config: %s", asdict(config))
    logger.info("run_dir: %s", run_dir.as_posix())

    datasets = Dataset.load_from_disk(config.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.add_tokens("<|e|>")
    tokenizer.end_ind = tokenizer.encode("<|e|>")[0]

    tokenized = datasets.train_test_split(test_size=config.test_size, seed=config.split_seed, shuffle=True)
    tokenized = tokenized.map(
        lambda x: process(x, tokenizer, config.sequence_len, config.sample_stride),
        batched=True,
        remove_columns=datasets.column_names,
    )
    tokenized.set_format(type="torch", device=device)

    train_loader = DataLoader(tokenized["train"], batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(tokenized["test"], batch_size=config.batch_size, shuffle=True)

    model = CharGPT(len(tokenizer), config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("model:\n%s", model)
    logger.info("total_params: %s", total_params)

    start_epoch = 0
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    train_step_losses: List[float] = []
    eval_train_losses: List[float] = []
    eval_test_losses: List[float] = []
    ckpt = None

    if args.remuse != 0:
        resume_dir = get_resume_dir(root_dir, args.mode_size, args.remuse)
        resume_ckpt_name = "best_test_model.pth" if args.resume_from == "test" else "best_train_model.pth"
        resume_ckpt_path = resume_dir / resume_ckpt_name
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"续训权重不存在: {resume_ckpt_path.as_posix()}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_test_loss = float(ckpt.get("best_test_loss", float("inf")))
        best_train_loss = float(ckpt.get("best_train_loss", float("inf")))
        logger.info("resume_from: %s", resume_dir.as_posix())
        logger.info("resume_ckpt: %s", resume_ckpt_path.name)
        logger.info("resume_start_epoch: %s", start_epoch)
        logger.info("resume_best_test_loss: %.4f", best_test_loss)
        logger.info("resume_best_train_loss: %.4f", best_train_loss)

    scheduler = build_lr_scheduler(optimizer, config, args.epochs)
    if scheduler is not None:
        logger.info(
            "lr_scheduler: %s (cosine T_max=%d, eta_min=%s)",
            config.lr_scheduler,
            config.lr_cosine_t_max_epochs if config.lr_cosine_t_max_epochs > 0 else max(1, args.epochs),
            config.lr_cosine_eta_min,
        )
    if args.remuse != 0 and scheduler is not None and ckpt is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("loaded scheduler state from checkpoint")

    meta = {
        "mode_size": args.mode_size,
        "epochs_this_run": args.epochs,
        "device": device,
        "config": asdict(config),
        "resume_from": args.remuse,
        "resume_from_type": args.resume_from,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        for data in train_loader:
            inputs, labels = data["inputs"], data["labels"]
            inputs = corrupt_inputs(inputs, tokenizer, config.input_corrupt_prob)
            inputs = random_prefix_truncate_inputs(
                inputs,
                tokenizer,
                trunc_prob=config.prefix_trunc_prob,
                min_prefix_len=config.min_prefix_len,
            )
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.transpose(-2, -1),
                labels,
                label_smoothing=config.label_smoothing,
            )
            train_step_losses.append(loss.item())
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        stats = estimate_loss(model, train_loader, test_loader, config.eval_iters)
        eval_train_losses.append(stats["train"])
        eval_test_losses.append(stats["test"])
        logger.info(
            "epoch %3d | train loss %.4f | test loss %.4f | lr %.2e",
            epoch,
            stats["train"],
            stats["test"],
            optimizer.param_groups[0]["lr"],
        )

        if stats["test"] < best_test_loss:
            best_test_loss = stats["test"]
            payload = {
                "epoch": epoch,
                "mode_size": args.mode_size,
                "best_test_loss": best_test_loss,
                "best_train_loss": best_train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(payload, run_dir / "best_test_model.pth")
            logger.info("save new best_test model at epoch %d, test loss %.4f", epoch, best_test_loss)

        if stats["train"] < best_train_loss:
            best_train_loss = stats["train"]
            payload = {
                "epoch": epoch,
                "mode_size": args.mode_size,
                "best_test_loss": best_test_loss,
                "best_train_loss": best_train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(payload, run_dir / "best_train_model.pth")
            logger.info("save new best_train model at epoch %d, train loss %.4f", epoch, best_train_loss)

        if scheduler is not None:
            scheduler.step()

    last_payload = {
        "epoch": start_epoch + args.epochs - 1,
        "mode_size": args.mode_size,
        "best_test_loss": best_test_loss,
        "best_train_loss": best_train_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        last_payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(last_payload, run_dir / "last_model.pth")

    plot_curve(
        train_step_losses,
        title="Train Loss Curve",
        y_label="Cross Entropy Loss",
        save_path=run_dir / "train_loss_curve.png",
    )
    plot_curve(
        eval_test_losses,
        title="Eval Loss Curve",
        y_label="Cross Entropy Loss",
        save_path=run_dir / "eval_loss_curve.png",
    )

    hist = {
        "epoch_start": start_epoch,
        "epoch_end": start_epoch + args.epochs - 1,
        "eval_train_losses": eval_train_losses,
        "eval_test_losses": eval_test_losses,
        "best_test_loss": best_test_loss,
        "best_train_loss": best_train_loss,
    }
    (run_dir / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("training done. artifacts saved in: %s", run_dir.as_posix())


if __name__ == "__main__":
    main()
