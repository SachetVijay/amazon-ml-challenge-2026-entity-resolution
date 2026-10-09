"""Handcrafted catalog features + TF-IDF/SVD.

  python -m amlc.features            -> artifacts/feats/hand_*.npy, tfidf_svd_*.npy, tfidf sparse (.npz)

Quantity/unit parsing matters a lot for anything size/price related: '12 oz x 6 pack'
is 72 oz of product. Extend UNIT_MAP / patterns after looking at the real data.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import ART, save_matrix, test_df, texts, train_df

# canonical: mass -> grams, volume -> ml, length -> cm
UNIT_MAP = {
    "g": ("mass", 1), "gram": ("mass", 1), "grams": ("mass", 1), "gm": ("mass", 1),
    "kg": ("mass", 1000), "kilogram": ("mass", 1000), "mg": ("mass", 1e-3),
    "oz": ("mass", 28.3495), "ounce": ("mass", 28.3495), "ounces": ("mass", 28.3495),
    "lb": ("mass", 453.592), "lbs": ("mass", 453.592), "pound": ("mass", 453.592), "pounds": ("mass", 453.592),
    "ml": ("vol", 1), "milliliter": ("vol", 1), "l": ("vol", 1000), "liter": ("vol", 1000), "litre": ("vol", 1000),
    "fl oz": ("vol", 29.5735), "fluid ounce": ("vol", 29.5735), "fluid ounces": ("vol", 29.5735),
    "gallon": ("vol", 3785.41), "gal": ("vol", 3785.41), "quart": ("vol", 946.353), "pint": ("vol", 473.176),
    "cm": ("len", 1), "mm": ("len", 0.1), "m": ("len", 100), "inch": ("len", 2.54), "inches": ("len", 2.54),
    "in": ("len", 2.54), "ft": ("len", 30.48), "foot": ("len", 30.48), "feet": ("len", 30.48),
    "count": ("count", 1), "ct": ("count", 1), "pcs": ("count", 1), "pieces": ("count", 1),
    "piece": ("count", 1), "pack": ("count", 1), "sheets": ("count", 1), "bags": ("count", 1),
}
_units = sorted(UNIT_MAP, key=len, reverse=True)
QTY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(map(re.escape, _units)) + r")\b", re.I)
PACK_RE = re.compile(r"(?:pack|set|case|box)\s*of\s*(\d+)|(\d+)\s*[- ]?(?:pack|pk|count|ct)\b", re.I)
VALUE_RE = re.compile(r"Value:\s*([\d.]+)", re.I)
UNIT_RE = re.compile(r"Unit:\s*([A-Za-z ]+)", re.I)


def parse_row(t: str) -> dict:
    out = {"len_chars": len(t), "n_words": len(t.split()), "n_digits": sum(ch.isdigit() for ch in t),
           "upper_ratio": sum(ch.isupper() for ch in t) / max(len(t), 1), "n_lines": t.count("\n") + 1,
           "n_bullets": len(re.findall(r"bullet|•|^\s*[-*]", t, re.I | re.M))}
    agg = {"mass": [], "vol": [], "len": [], "count": []}
    for num, unit in QTY_RE.findall(t):
        kind, mult = UNIT_MAP[unit.lower()]
        try:
            agg[kind].append(float(num.replace(",", ".")) * mult)
        except ValueError:
            pass
    for k, v in agg.items():
        out[f"{k}_max"] = max(v) if v else np.nan
        out[f"{k}_first"] = v[0] if v else np.nan
        out[f"{k}_n"] = len(v)
    packs = [int(a or b) for a, b in PACK_RE.findall(t) if (a or b)]
    out["pack"] = max(packs) if packs else np.nan
    m = VALUE_RE.search(t)
    out["value_field"] = float(m.group(1)) if m else np.nan
    m = UNIT_RE.search(t)
    unit = m.group(1).strip().lower() if m else ""
    out["unit_field_kind"] = ["", "mass", "vol", "len", "count"].index(UNIT_MAP.get(unit, ("",))[0]) if unit else -1
    total = out["value_field"] if not np.isnan(out["value_field"]) else out["mass_max"]
    out["total_qty"] = total * (out["pack"] if not np.isnan(out["pack"]) else 1) if not np.isnan(total) else np.nan
    for w in ["organic", "premium", "gluten free", "kosher", "bulk", "gift", "set", "refill", "sample", "travel"]:
        out[f"kw_{w.replace(' ', '_')}"] = int(w in t.lower())
    return out


def hand_features(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([parse_row(t) for t in s]).astype(np.float32)


def main():
    tr_t, te_t = texts(train_df()), texts(test_df())
    h_tr, h_te = hand_features(tr_t), hand_features(te_t)
    save_matrix("feats", "hand", h_tr.to_numpy(), h_te.to_numpy())
    (ART / "feats" / "hand_cols.txt").write_text("\n".join(h_tr.columns))
    print("hand feats", h_tr.shape)

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000, sublinear_tf=True)
    all_t = pd.concat([tr_t, te_t])  # unsupervised fit on train+test text is standard and leak-free w.r.t. target
    word.fit(all_t)
    char.fit(all_t)
    X_tr = sp.hstack([word.transform(tr_t), char.transform(tr_t)]).tocsr()
    X_te = sp.hstack([word.transform(te_t), char.transform(te_t)]).tocsr()
    sp.save_npz(ART / "feats" / "tfidf_train.npz", X_tr)
    sp.save_npz(ART / "feats" / "tfidf_test.npz", X_te)
    svd = TruncatedSVD(256, random_state=0).fit(sp.vstack([X_tr, X_te]))
    save_matrix("feats", "tfidf_svd", svd.transform(X_tr).astype(np.float32), svd.transform(X_te).astype(np.float32))
    print("tfidf", X_tr.shape)


if __name__ == "__main__":
    main()
