"""Local tokenizer admission; never downloads or truncates workflow state."""
from functools import lru_cache


@lru_cache(maxsize=8)
def tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=False)


def input_tokens(model, messages):
    return len(tokenizer(model.tokenizer_name).apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True))


def check_context(model, messages):
    from workflow import LimitReached
    count = input_tokens(model, messages)
    # Small margin for server template differences. Pin server/tokenizer revisions
    # and compare server prompt_tokens during calibration.
    if count + model.max_tokens + 64 > model.context_tokens:
        raise LimitReached(f'context_tokens: input={count}, output_reserve={model.max_tokens}, '
                           f'margin=64, capacity={model.context_tokens}; no silent truncation')
    return count
