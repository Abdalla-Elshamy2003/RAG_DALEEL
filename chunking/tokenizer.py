from __future__ import annotations

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore


class HFTokenizerWrapper:
    def __init__(self, model_name: str) -> None:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required. Install: pip install transformers sentencepiece protobuf"
            )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        try:
            self.tokenizer.model_max_length = 10**9
        except Exception:
            pass

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))