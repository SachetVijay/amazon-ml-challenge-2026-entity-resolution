# Business Entity Resolution — v2 reproduction

v2 = TF-IDF blocking (top-5 both directions) + LightGBM pair classifier trained at **full density**
(the whole training split is blocked exactly like test; features/labels for a deterministic 12.5 % hash
sample of Source-1 entities) + exclusive assignment + threshold 0.75.
Validation (out-of-fold, official macro F0.5 incl. singletons): 0.927 overall (US 0.940, India 0.908).
Uses only the provided data. Model: LightGBM (MIT). Hardware: 24 CPU threads, 10 GB RAM, no GPU.

## Setup
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```
Data expected at `dataset/train/*.tsv` and `dataset/test/*.tsv`.

## Reproduce with the shipped model (`models/model.txt`)
```bash
python -m er.predict --split-dir dataset/test --prefix test --art artifacts/test \
    --model models/model.txt --out output --rule thr --thr 0.75
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (~40 min on 24 threads).

## Retrain the model
```bash
python -m er.train_full --split-dir dataset/train --prefix train --art artifacts/train \
    --gt dataset/train/train_ground_truth.tsv --sample-mod 8        # -> artifacts/train/model.txt
```

## Pipeline (src/er)
| module | role |
|---|---|
| `normalize.py` | transliteration (anyascii), legal-form canonicalisation, abbreviation expansion, consonant-skeleton keys |
| `prep.py` | streaming TSV → normalised parquet (bounded memory) |
| `blocking.py` | TF-IDF retrieval (name char-3-grams + skeleton tokens + address tokens + numbers), top-5 both directions, per country |
| `features.py` | ~45 pair similarity features (rapidfuzz, set overlaps, house-number agreement, legal forms, name frequency) |
| `predict.py` | per-country chunked inference + competition context features, writes both TSVs |
| `candidates.py` | (hybrid sparse+dense candidates — not used by v2, imported by predict) |
| `decide.py` | exclusivity (each S2/S3 record → ≤ 1 S1), threshold / expected-F0.5 rules, official macro F0.5 |
| `train_full.py` | full-density training + entity-grouped CV |
