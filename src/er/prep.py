"""Load a split (3 sources [+ ground truth]) and attach normalised views. Cached as parquet.

  python -m er.prep --split-dir student_resource/dataset/train --prefix train --out artifacts/train
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .normalize import normalize_record


def _norm_chunk(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([normalize_record(n, a) for n, a in zip(df.business_name, df.business_address)],
                        index=df.index)


def read_tsv(p) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(split_dir: str, prefix: str, out: str, workers: int = 8, arrow: bool = False) -> pd.DataFrame:
    """Streams TSV -> normalise -> parquet in 500k-row chunks (bounded memory at 12M rows).
    arrow=True reads back with pyarrow-backed strings (~3x smaller than object dtype)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_p = Path(out)
    cache = out_p / "records.parquet"
    if not cache.exists():
        out_p.mkdir(parents=True, exist_ok=True)
        writer, offset = None, 0
        with Pool(workers) as p:
            for s in (1, 2, 3):
                f = Path(split_dir) / (f"{prefix}_source{s}.tsv" if prefix else f"source{s}.tsv")
                for df in pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=500_000):
                    df["src"] = s
                    df.index = pd.RangeIndex(offset, offset + len(df))
                    offset += len(df)
                    chunks = [df.iloc[i:i + 20_000] for i in range(0, len(df), 20_000)]
                    df = pd.concat([df, pd.concat(p.map(_norm_chunk, chunks))], axis=1)
                    df["country"] = df["country"].str.strip()
                    t = pa.Table.from_pandas(df, preserve_index=False)
                    writer = writer or pq.ParquetWriter(cache, t.schema)
                    writer.write_table(t)
        writer.close()
    return pd.read_parquet(cache, dtype_backend="pyarrow") if arrow else pd.read_parquet(cache)


def load_gt(path) -> pd.DataFrame:
    """Ground truth as a long pair table (s1, other). Singletons are absent here."""
    gt = read_tsv(path)
    gt = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    gt = gt[gt.m.notna() & (gt.m != "")]
    return gt.rename(columns={"source1_entity_id": "s1", "m": "other"})[["s1", "other"]].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    r = load_split(a.split_dir, a.prefix, a.out, a.workers)
    print(r.shape, r.src.value_counts().to_dict())
