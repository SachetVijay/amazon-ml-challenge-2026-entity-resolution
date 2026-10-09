# Business Entity Resolution — v3 reproduction

v3 = **hybrid blocking** (TF-IDF top-5 ∪ fine-tuned multilingual bi-encoder top-5, both directions, per
country) + LightGBM pair classifier trained at **full density** with dense retrieval features +
exclusive assignment + per-entity **expected-F0.5** selection.
Validation (out-of-fold, official macro F0.5 incl. singletons): **0.978** (US 0.979, India 0.977);
blocking recall 99.2 %. Uses only the provided data.
Models: LightGBM (MIT); `intfloat/multilingual-e5-small` fine-tuned (MIT, 118M parameters).
Hardware used: 24 CPU threads, 10 GB RAM, NVIDIA RTX 3050 8 GB.

## Setup
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```
Data expected at `dataset/train/*.tsv` and `dataset/test/*.tsv`.

## Reproduce the test predictions
The fine-tuned bi-encoder weights (fp16) are distributed separately: `unzip v3_encoder_weights.zip -d models/`
creates `models/e5er_fp16`. Alternatively retrain them with step B1 (then use `models/e5er`).
```bash
# A. normalise + sparse candidates are built on demand by predict; dense retrieval first:
python -m er.prep --split-dir dataset/test --prefix test --out artifacts/test
python -m er.biencoder encode --art artifacts/test --model models/e5er_fp16   # GPU, ~45 min
python -m er.biencoder search --art artifacts/test --k-other 10 --k-s1 10    # GPU, ~30 min
python -m er.predict --split-dir dataset/test --prefix test --art artifacts/test \
    --model models/model.txt --hybrid --dense-k 5 --rule expected_f --out output
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Retrain from scratch
```bash
# B0. sparse full-density training features (also caches sparse candidates used as hard negatives)
python -m er.train_full --split-dir dataset/train --prefix train --art artifacts/train \
    --gt dataset/train/train_ground_truth.tsv
# B1. bi-encoder: MNRL with mined hard negatives on entity hash-bucket 1 (disjoint from classifier bucket 0)
python -m er.biencoder train --art artifacts/train --gt dataset/train/train_ground_truth.tsv --out models/e5er
python -m er.biencoder encode --art artifacts/train --model models/e5er
python -m er.biencoder search --art artifacts/train --k-other 10 --k-s1 10
# B2. hybrid classifier -> artifacts/train/model_hyb5.txt
python -m er.train_full --split-dir dataset/train --prefix train --art artifacts/train \
    --gt dataset/train/train_ground_truth.tsv --hybrid --dense-k 5
```

## Pipeline (src/er)
| module | role |
|---|---|
| `normalize.py` | transliteration (anyascii), legal-form canonicalisation, abbreviation expansion, consonant-skeleton keys |
| `prep.py` | streaming TSV → normalised parquet (bounded memory) |
| `blocking.py` | sparse TF-IDF retriever (name char-3-grams + skeleton tokens + address tokens + numbers) |
| `biencoder.py` | bi-encoder fine-tuning (hard negatives), disk-memmap encoding, exact tiled GPU top-k |
| `candidates.py` | hybrid sparse ∪ dense candidates + unbiased dense context features |
| `features.py` | ~45 string/number similarity features |
| `train_full.py` | full-density training, entity-grouped CV, OOF macro F0.5 |
| `predict.py` | chunked per-country inference, writes both TSVs |
| `decide.py` | exclusivity, threshold / expected-F0.5 rules, official metric |
