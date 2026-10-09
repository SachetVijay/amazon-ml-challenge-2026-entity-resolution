# Business Entity Resolution — v1 reproduction

v1 = TF-IDF blocking + LightGBM pair classifier trained on a 10 % development slice + exclusive
assignment + probability threshold 0.75. Uses only the provided data (no external lookups).
Model: LightGBM (MIT). Hardware used: 24 CPU threads, 10 GB RAM, no GPU.

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

## Retrain the model from scratch
```bash
python -m er.make_dev_slice --train-dir dataset/train --out data/dev10
python -m er.prep --split-dir data/dev10 --out artifacts/dev10
python -m er.train_dev --art artifacts/dev10 --gt data/dev10/ground_truth.tsv   # -> artifacts/dev10/model.txt
```

## Pipeline (src/er)
| module | role |
|---|---|
| `normalize.py` | transliteration (anyascii), legal-form canonicalisation, abbreviation expansion, consonant-skeleton keys |
| `prep.py` | streaming TSV → normalised parquet |
| `blocking.py` | TF-IDF retrieval (name char-3-grams + skeleton tokens + address tokens + numbers), top-5 both directions, per country |
| `features.py` | ~45 pair similarity features (rapidfuzz, set overlaps, house-number agreement, legal forms, name frequency) |
| `predict.py` | chunked per-country inference + competition context features, writes both TSVs |
| `decide.py` | exclusivity (each S2/S3 record → ≤ 1 S1), threshold / expected-F0.5 rules, official macro F0.5 |
| `train_dev.py`, `make_dev_slice.py` | training on the development slice |
