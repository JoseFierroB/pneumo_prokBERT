#!/usr/bin/env python3
"""
Runtime patches for the vendored upstream ProkBERT code (prokbert/ is kept untouched).

Upstream issues worked around here:
- prokbert_tokenizer.py:352 writes token + "\\n" (literal backslash-n) -> corrupt vocab.txt
- ProkBERTTokenizer.vocab_size is not defined, Trainer.save_pretrained raises NotImplementedError
- __init__ appends a synthetic full-unk token ('N'*kmer) to self.vocab -> 4102 vs 4101 canonical
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def apply_tokenizer_patches():
    from prokbert.prokbert_tokenizer import ProkBERTTokenizer

    if getattr(ProkBERTTokenizer, "_pneumo_patched", False):
        return ProkBERTTokenizer

    ProkBERTTokenizer.vocab_size = property(
        lambda self: len(self.get_vocab()) if hasattr(self, "get_vocab") else 4101
    )

    def _fixed_save_vocabulary(self, save_directory, filename_prefix=None, **kwargs):
        prefix = filename_prefix or ""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        path = Path(save_directory) / f"{prefix}vocab.txt"
        skip = getattr(self, "full_unk_token", None)
        with open(path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                if token == skip:
                    continue
                f.write(token + "\n")
        return (str(path),)

    ProkBERTTokenizer.save_vocabulary = _fixed_save_vocabulary

    _orig_save = ProkBERTTokenizer.save_pretrained

    def _patched_save(self, save_directory, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        try:
            return _orig_save(self, save_directory, **kwargs)
        except NotImplementedError:
            (Path(save_directory) / "tokenizer_config.json").write_text(
                '{"tokenizer_class":"ProkBERTTokenizer","kmer":6,"shift":1}'
            )
            return ()

    ProkBERTTokenizer.save_pretrained = _patched_save
    ProkBERTTokenizer._pneumo_patched = True
    return ProkBERTTokenizer


def write_run_info(output_dir, info):
    info = dict(info)
    info.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    return str(path)
