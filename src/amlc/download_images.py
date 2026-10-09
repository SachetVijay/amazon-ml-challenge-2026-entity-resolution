"""Parallel, resumable image downloader. Start this in tmux the minute data lands.

  python -m amlc.download_images --split train --workers 64 --max-side 512

Files are named by sha1(url) so duplicate URLs download once. Writes
artifacts/cache/images_<split>.csv mapping row -> local path ('' if failed).
"""
import argparse
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from .data import ART, ROOT, cfg, test_df, train_df

IMG_DIR = ROOT / "images"


def path_for(url: str) -> Path:
    return IMG_DIR / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".jpg")


def fetch(sess, url, max_side, retries=4):
    p = path_for(url)
    if p.exists() and p.stat().st_size > 0:
        return url, str(p)
    for i in range(retries):
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 404:
                return url, ""
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.thumbnail((max_side, max_side))
            tmp = p.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=92)
            tmp.rename(p)
            return url, str(p)
        except Exception:
            time.sleep(1.5 * 2 ** i)
    return url, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-side", type=int, default=512)
    a = ap.parse_args()
    IMG_DIR.mkdir(exist_ok=True)
    df = train_df() if a.split == "train" else test_df()
    urls = df[cfg()["image_col"]].fillna("").astype(str)
    uniq = [u for u in urls.unique() if u.startswith("http")]
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=a.workers, pool_maxsize=a.workers))
    got = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, sess, u, a.max_side) for u in uniq]
        for fu in tqdm(as_completed(futs), total=len(futs), desc=a.split):
            u, p = fu.result()
            got[u] = p
    out = pd.DataFrame({"row": range(len(df)), "path": [got.get(u, "") for u in urls]})
    (ART / "cache").mkdir(parents=True, exist_ok=True)
    out.to_csv(ART / "cache" / f"images_{a.split}.csv", index=False)
    print(f"{a.split}: {(out.path != '').mean():.2%} rows have images ({len(uniq)} unique urls)")


if __name__ == "__main__":
    main()
