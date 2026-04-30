import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from train import CharGPT, get_config_by_mode


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "test.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


@torch.no_grad()
def generate(
    model: CharGPT,
    context: torch.Tensor,
    end_token_id: int,
    sequence_len: int,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
) -> List[int]:
    out = context.tolist()[0]
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(context[:, -sequence_len:])
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(next_token_logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        context = torch.concat((context, ix), dim=-1)
        out.append(ix.item())
        if out[-1] == end_token_id:
            break
    return out


def get_checkpoint_path(run_dir: Path, weight_type: str) -> Path:
    file_map = {
        "last": "last_model.pth",
        "best_test": "best_test_model.pth",
        "best_train": "best_train_model.pth",
    }
    ckpt_path = run_dir / file_map[weight_type]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {ckpt_path.as_posix()}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="加载训练权重并生成文本")
    parser.add_argument("--mode_size", type=int, required=True, help="参数组编号（需和训练目录一致）")
    parser.add_argument("--train_id", type=int, required=True, help="训练编号（即 train_modeSize_trainId 的最后一个数字）")
    parser.add_argument(
        "--weight_type",
        type=str,
        default="best_test",
        choices=["last", "best_test", "best_train"],
        help="加载哪个权重：last / best_test / best_train",
    )
    parser.add_argument("--prompt", type=str, required=True, help="生成起始文本")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="最大新生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--num_samples", type=int, default=1, help="生成条数")
    parser.add_argument("--seed", type=int, default=12046, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config_by_mode(args.mode_size)

    root_dir = Path("my_model")
    run_dir = root_dir / f"train_{args.mode_size}_{args.train_id}"
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到训练目录: {run_dir.as_posix()}")

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"训练目录缺少 meta.json: {run_dir.as_posix()}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_mode_size = int(meta.get("mode_size", -1))
    if trained_mode_size != args.mode_size:
        raise ValueError(
            f"参数组不一致：输入 mode_size={args.mode_size}，训练目录中 mode_size={trained_mode_size}"
        )

    ckpt_path = get_checkpoint_path(run_dir, args.weight_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / f"test_{args.weight_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(output_dir)

    logger.info("device: %s", device)
    logger.info("run_dir: %s", run_dir.as_posix())
    logger.info("checkpoint: %s", ckpt_path.as_posix())
    logger.info("mode_size: %s", args.mode_size)
    logger.info("weight_type: %s", args.weight_type)
    logger.info("num_samples: %s", args.num_samples)
    logger.info("max_new_tokens: %s", args.max_new_tokens)
    logger.info("temperature: %.4f", args.temperature)
    logger.info("seed: %s", args.seed)
    logger.info("config: %s", config)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.add_tokens("<|e|>")
    tokenizer.end_ind = tokenizer.encode("<|e|>")[0]

    model = CharGPT(len(tokenizer), config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("model loaded. epoch in ckpt: %s", ckpt.get("epoch", "unknown"))

    all_results = []
    for idx in range(1, args.num_samples + 1):
        context = torch.tensor(tokenizer.encode(args.prompt), device=device).unsqueeze(0)
        token_ids = generate(
            model=model,
            context=context,
            end_token_id=tokenizer.end_ind,
            sequence_len=config.sequence_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        text = "".join(tokenizer.decode(token_ids))
        all_results.append({"sample_id": idx, "text": text})
        logger.info("sample %d generated, text_length=%d", idx, len(text))

    out_txt = output_dir / "generated.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"prompt: {args.prompt}\n")
        f.write(f"mode_size: {args.mode_size}, train_id: {args.train_id}, weight_type: {args.weight_type}\n")
        f.write(f"max_new_tokens: {args.max_new_tokens}, temperature: {args.temperature}\n")
        f.write("=" * 80 + "\n")
        for item in all_results:
            f.write(f"[sample {item['sample_id']}]\n")
            f.write(item["text"] + "\n")
            f.write("-" * 80 + "\n")

    out_json = output_dir / "generated.json"
    out_json.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "mode_size": args.mode_size,
                "train_id": args.train_id,
                "weight_type": args.weight_type,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "checkpoint": ckpt_path.as_posix(),
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("generation done. outputs: %s, %s", out_txt.as_posix(), out_json.as_posix())


if __name__ == "__main__":
    main()
