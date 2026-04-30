import argparse

from datasets import Dataset
from transformers import AutoTokenizer


def get_training_corpus(data: Dataset, batch_size: int):
    texts = data["text"]
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser(description="基于新数据集训练并保存新分词器")
    parser.add_argument("--dataset_dir", type=str, default="./my_novel_dataset_v2", help="输入Dataset目录")
    parser.add_argument("--base_tokenizer", type=str, default="gpt2", help="基础分词器")
    parser.add_argument("--vocab_size", type=int, default=9000, help="词表大小")
    parser.add_argument("--min_frequency", type=int, default=2, help="最小词频")
    parser.add_argument("--batch_size", type=int, default=10, help="训练分词器时的迭代批大小")
    parser.add_argument("--out_tokenizer_dir", type=str, default="./my_novel_tokenizer_v2", help="输出分词器目录")
    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.dataset_dir)
    print(f"加载数据集: {args.dataset_dir}, 样本数: {len(dataset)}")

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    new_tok = base_tok.train_new_from_iterator(
        get_training_corpus(dataset, args.batch_size),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    new_tok.save_pretrained(args.out_tokenizer_dir)
    print(f"已保存新分词器到: {args.out_tokenizer_dir}")


if __name__ == "__main__":
    main()
