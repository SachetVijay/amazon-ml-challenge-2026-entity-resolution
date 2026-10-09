"""Image embeddings with CLIP / SigLIP-family models (anything exposing get_image_features).

  python -m amlc.embed_image --model google/siglip2-base-patch16-224 --name siglip2b --batch 128

Rows without an image get a zero vector; the last column is a has_image flag.
Run download_images first.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

from .data import ART, save_matrix


class Imgs(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        ok = isinstance(p, str) and p != ""
        try:
            im = Image.open(p).convert("RGB") if ok else Image.new("RGB", (224, 224))
        except Exception:
            im, ok = Image.new("RGB", (224, 224)), False
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.model)

    def run(split):
        paths = pd.read_csv(ART / "cache" / f"images_{split}.csv").path.fillna("").tolist()
        dl = DataLoader(Imgs(paths, proc), batch_size=a.batch, num_workers=a.workers)
        outs, flags = [], []
        with torch.no_grad():
            for px, ok in dl:
                e = model.get_image_features(pixel_values=px.to(dev, dt))
                e = torch.nn.functional.normalize(e.float(), dim=-1) * ok[:, None].to(dev)
                outs.append(e.cpu().numpy()), flags.append(ok.numpy())
        return np.hstack([np.vstack(outs), np.concatenate(flags)[:, None]]).astype(np.float16)

    save_matrix("emb", a.name, run("train"), run("test"))
    print("done", a.name)


if __name__ == "__main__":
    main()
