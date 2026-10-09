# Business Entity Resolution — reproduction guide

Pipeline: **normalise → block → pair features → LightGBM → cluster-aware decision → TSVs**.
Everything uses only the provided training/test files (no external data, APIs or lookups).

## Environment
- Python 3.12, Linux, 24 CPU threads, ~10 GB RAM (the pipeline is chunked to fit this)
- Optional NVIDIA GPU (only for the bi-encoder stage)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```
Place the challenge data at `dataset/train/*.tsv` and `dataset/test/*.tsv` (as distributed).

## 1. Train the pair classifier (full-density training)
Blocks the entire training split exactly as at test time, featurises a deterministic 12.5 % sample of
Source-1 entities (hash bucket 0), trains LightGBM with entity-grouped CV and saves the model.
```bash
python -m er.train_full --split-dir dataset/train --prefix train --art artifacts/train \
    --gt dataset/train/train_ground_truth.tsv --sample-mod 8
```
Outputs `artifacts/train/model.txt` and prints out-of-fold macro F0.5 (overall and per country).

## 2. Predict the test set
```bash
python -m er.predict --split-dir dataset/test --prefix test --art artifacts/test \
    --model artifacts/train/model.txt --out output
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (the exact candidate set the
model scores). Countries are processed as an open label set, one at a time.

## Source layout (`src/er/`)
| module | role |
|---|---|
| `normalize.py` | script-agnostic normalisation: transliteration (anyascii), legal-form canonicalisation, abbreviation expansion, consonant-skeleton phonetic keys |
| `prep.py` | streaming TSV → normalised parquet (bounded memory) |
| `blocking.py` | TF-IDF retrieval over name char-3-grams + skeleton tokens + address tokens + numbers, both directions, posting-list capped |
| `features.py` | ~45 pair features (rapidfuzz similarities, set overlaps, house-number agreement, legal forms, name frequency) |
| `predict.py` | chunked full-scale inference + candidate/competition context features |
| `decide.py` | exclusivity (each S2/S3 record → at most one S1), threshold / expected-F0.5 selection, official metric |
| `stage2.py` | cluster-aware re-scoring (sibling agreement, margins) |
| `biencoder.py` | fine-tuned multilingual-e5-small (MIT) dense retriever |
| `train_full.py`, `train_dev.py` | training / validation drivers |

## Licences
Models: LightGBM (MIT), intfloat/multilingual-e5-small (MIT, 118M parameters). Libraries: see requirements.txt.
