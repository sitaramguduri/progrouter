"""Download/cache the draft's MiniLM encoder and verify CPU inference.

Run in qwen-local: python setup_minilm.py
For a subsequent offline check: python setup_minilm.py --offline
"""
from __future__ import annotations

import argparse
import sys


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="Use only cached weights; do not download model files.",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        print(f"Python: {sys.executable}")
        print(f"Model: {MODEL_NAME}")
        print("Loading on CPU (cached files only)." if args.offline
              else "Loading on CPU; missing model files will be downloaded.")
        model = SentenceTransformer(
            MODEL_NAME, device="cpu", local_files_only=args.offline,
        )
        sentences = [
            "The implementation passes its public examples.",
            "The implementation fails a public example.",
        ]
        embeddings = model.encode(sentences, normalize_embeddings=True)
        if embeddings.shape != (2, 384) or not np.isfinite(embeddings).all():
            raise RuntimeError(f"Unexpected embeddings: shape={embeddings.shape}")
        print(f"Device: {model.device}")
        print(f"Embedding shape: {embeddings.shape}")
        del model

        cached = SentenceTransformer(
            MODEL_NAME, device="cpu", local_files_only=True,
        )
        cached.encode([sentences[0]], normalize_embeddings=True)
        print("Success: CPU inference and local-cache reload both passed.")
        print("This script does not change the workflow's encoder device setting.")
        return 0
    except ImportError as error:
        print(f"Missing dependency: {error}", file=sys.stderr)
        print("Install: python -m pip install -r requirements-progress.txt", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"MiniLM setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
