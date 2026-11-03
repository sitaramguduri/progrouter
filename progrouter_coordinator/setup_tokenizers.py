"""Explicitly provision local tokenizers for context admission; no model weights."""
import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    for size in ('0.5B', '1.5B', '3B', '7B'):
        name = f'Qwen/Qwen2.5-Coder-{size}-Instruct'
        AutoTokenizer.from_pretrained(name, local_files_only=args.offline, trust_remote_code=False)
        print('Tokenizer ready:', name)


if __name__ == '__main__':
    main()
