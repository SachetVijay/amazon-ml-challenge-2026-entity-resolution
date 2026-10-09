"""Config-driven data access shared by every script.

Everything the pipeline needs to know about the problem lives in config.json,
so when the real problem statement drops only that file should need editing.
Artifacts convention:
  artifacts/folds.npy            fold id per train row
  artifacts/oof/<name>.npy       out-of-fold preds (original target space, NaN = not predicted)
  artifacts/test/<name>.npy      test preds (original target space)
  artifacts/feats/<name>_{train,test}.npy   feature matrices
  artifacts/emb/<name>_{train,test}.npy     embeddings
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AMLC_ROOT", Path(__file__).resolve().parents[2]))
ART = ROOT / "artifacts"


@lru_cache
def cfg() -> dict:
    return json.loads((ROOT / os.environ.get("AMLC_CONFIG", "config.json")).read_text())


def _read(path: str) -> pd.DataFrame:
    p = ROOT / path
    cache = ART / "cache" / (p.stem + ".parquet")
    if cache.exists() and cache.stat().st_mtime > p.stat().st_mtime:
        return pd.read_parquet(cache)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def train_df() -> pd.DataFrame:
    return _read(cfg()["train"])


def test_df() -> pd.DataFrame:
    return _read(cfg()["test"])


def texts(df: pd.DataFrame) -> pd.Series:
    cols = cfg()["text_cols"]
    return df[cols].fillna("").astype(str).agg("\n".join, axis=1)


def y_raw() -> np.ndarray:
    return train_df()[cfg()["target_col"]].to_numpy(dtype=np.float64)


def fwd(y: np.ndarray) -> np.ndarray:
    t = cfg().get("target_transform", "none")
    if t == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if t == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inv(z: np.ndarray) -> np.ndarray:
    t = cfg().get("target_transform", "none")
    if t == "log1p":
        return np.expm1(z)
    if t == "sqrt":
        return np.square(np.clip(z, 0, None))
    return z


def folds() -> np.ndarray:
    return np.load(ART / "folds.npy")


def save_preds(name: str, oof: np.ndarray, test: np.ndarray) -> None:
    (ART / "oof").mkdir(parents=True, exist_ok=True)
    (ART / "test").mkdir(parents=True, exist_ok=True)
    np.save(ART / "oof" / f"{name}.npy", oof)
    np.save(ART / "test" / f"{name}.npy", test)


def load_matrix(kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """kind is 'feats' or 'emb'."""
    d = ART / kind
    return np.load(d / f"{name}_train.npy"), np.load(d / f"{name}_test.npy")


def save_matrix(kind: str, name: str, tr: np.ndarray, te: np.ndarray) -> None:
    d = ART / kind
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}_train.npy", tr)
    np.save(d / f"{name}_test.npy", te)
