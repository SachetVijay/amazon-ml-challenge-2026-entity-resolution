"""Text embeddings with any sentence-transformers compatible model.

  python -m amlc.embed_text --model Qwen/Qwen3-Embedding-0.6B --name qwen3e06 --batch 64
  python -m amlc.embed_text --model Qwen/Qwen3-Embedding-4B --name qwen3e4b --batch 16 --max-len 384

Check the model licence against the competition rules before using it.
"""
import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .data import save_matrix, test_df, texts, train_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--prompt", default=None, help="instruction prefix for instruct-style embedders")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}
    m = SentenceTransformer(a.model, device=dev, model_kwargs=kw, trust_remote_code=True)
    m.max_seq_length = a.max_len

    def enc(s):
        # sort by length so batches pad less -> big speedup
        order = np.argsort(s.str.len().to_numpy())
        e = m.encode(s.iloc[order].tolist(), batch_size=a.batch, prompt=a.prompt, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
        out = np.empty_like(e)
        out[order] = e
        return out.astype(np.float16)

    save_matrix("emb", a.name, enc(texts(train_df())), enc(texts(test_df())))
    print("done", a.name)


if __name__ == "__main__":
    main()
