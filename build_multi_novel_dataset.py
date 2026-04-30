import argparse
import re
from pathlib import Path
from typing import List

from datasets import Dataset


def clean_novel(text: str) -> str:
    """
    清洗文本：
    1) 删除章节标题（兼容：第一章 / 第01慕 / 第44幕 等）
    2) 删除连续等号分隔线
    3) 清理多余空行
    """
    text = re.sub(r"第[零一二三四五六七八九十百千万两\d０-９]+[章节慕幕]\s*", "\n", text)
    text = re.sub(r"\n=+\n", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def split_to_blocks(clean_text: str, max_words: int) -> List[str]:
    """
    按自然段累积，达到阈值后换新块。
    """
    blocks = []
    current_block = []
    total_words = 0

    for paragraph in clean_text.split("\n"):
        para_length = len(paragraph)
        if total_words + para_length > max_words and current_block:
            blocks.append("\n".join(current_block))
            current_block = [paragraph]
            total_words = para_length
        else:
            current_block.append(paragraph)
            total_words += para_length

    if current_block:
        blocks.append("\n".join(current_block))
    return blocks


def read_and_process_one_book(file_path: Path, block_words: int) -> List[str]:
    raw_text = file_path.read_text(encoding="utf-8")
    cleaned = clean_novel(raw_text)
    return split_to_blocks(cleaned, block_words)


def main():
    parser = argparse.ArgumentParser(description="构建三本小说联合训练数据集")
    parser.add_argument("--data_dir", type=str, default=".", help="小说文件所在目录")
    parser.add_argument("--block_words", type=int, default=1200, help="分块字数阈值")
    parser.add_argument("--out_dataset_dir", type=str, default="./my_novel_dataset_v2", help="输出Dataset目录")
    parser.add_argument(
        "--out_list_txt",
        type=str,
        default="./训练数据_list_v2.txt",
        help="输出文本列表文件（调试用）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    novel_files = ["我这一辈子.txt", "骆驼祥子.txt", "四世同堂.txt"]

    all_blocks = []
    for name in novel_files:
        file_path = data_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"未找到小说文件: {file_path.as_posix()}")
        blocks = read_and_process_one_book(file_path, args.block_words)
        print(f"{name}: {len(blocks)} 个文本块")
        all_blocks.extend(blocks)

    print(f"总文本块数: {len(all_blocks)}")

    Path(args.out_list_txt).write_text(str(all_blocks), encoding="utf-8")

    dataset = Dataset.from_dict({"text": all_blocks})
    dataset.save_to_disk(args.out_dataset_dir)
    print(f"已保存Dataset到: {args.out_dataset_dir}")


if __name__ == "__main__":
    main()
