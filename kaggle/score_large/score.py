"""Score test band pairs with the e5-large cross-encoder trained in kernel amlc-ce-e5large (2x T4, fp16)."""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

mdir = os.path.dirname(glob.glob("/kaggle/input/**/model/config.json", recursive=True)[0])
data = glob.glob("/kaggle/input/**/band.parquet", recursive=True)[0]
print("model", mdir, "data", data, flush=True)
df = pd.read_parquet(data)
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForSequenceClassification.from_pretrained(mdir, torch_dtype=torch.float16).cuda().eval()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
order = np.argsort((df.text_a.str.len() + df.text_b.str.len()).to_numpy())
ta, tb = df.text_a.to_numpy()[order], df.text_b.to_numpy()[order]
out, t, bs = [], time.time(), 1024
with torch.no_grad():
    for i in range(0, len(df), bs):
        enc = tok(list(ta[i:i + bs]), list(tb[i:i + bs]), truncation=True, max_length=128, padding=True, return_tensors="pt")
        out.append(model(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy())
        if i % (bs * 200) == 0:
            print(f"{i}/{len(df)} {time.time() - t:.0f}s", flush=True)
s = np.empty(len(df), np.float32)
s[order] = np.concatenate(out)
df[["a", "b", "country"]].assign(ce=s).to_parquet("/kaggle/working/ce_large_test.parquet")
print("done", len(df), f"{time.time() - t:.0f}s", flush=True)
