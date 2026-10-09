# Business Entity Resolution — v5 reproduction

v5 = v4 with (a) an **e5-base cross-encoder** (278M, MIT; trained on entity buckets 2–3, 1.2M pairs) and (b) **French-robust normalisation** (`N°` numbering marker, French filler words) + an **acronym feature**. v4 = v3 hybrid pipeline (TF-IDF top-5 ∪ fine-tuned bi-encoder top-5, full-density LightGBM, now with
name-ambiguity features) + **cross-encoder re-ranking of uncertain pairs** + **stage-2 LightGBM**
(stage-1 probability, cross-encoder logit, cluster context) + exclusive assignment + expected-F0.5.
Validation (out-of-fold, official macro F0.5 incl. singletons): **0.9865** (v4: 0.985, v3: 0.978).
Uses only the provided data. Models (all MIT): LightGBM ×2; `intfloat/multilingual-e5-small`
e5-small (118M) fine-tuned as a bi-encoder (retrieval) and `intfloat/multilingual-e5-base` (278M) fine-tuned as a cross-encoder (re-ranking).
Hardware: 24 CPU threads, 10 GB RAM, NVIDIA RTX 3050 8 GB.

## Setup
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```
Data expected at `dataset/train/*.tsv` and `dataset/test/*.tsv`.
Fine-tuned transformer weights (fp16) ship separately in `v5_weights.zip`:
`unzip v5_weights.zip -d models/` → `models/e5er_fp16` (bi-encoder), `models/ce_base_fp16` (cross-encoder).

## Reproduce the test predictions (shipped models)
```bash
python -m er.prep --split-dir dataset/test --prefix test --out artifacts/test
python -m er.biencoder encode --art artifacts/test --model models/e5er_fp16           # GPU ~45 min
python -m er.biencoder search --art artifacts/test --k-other 10 --k-s1 10             # GPU ~30 min
# stage 1 (sparse candidates are built on the fly) -> artifacts/test/pred_hyb_<country>.parquet
python -m er.predict --split-dir dataset/test --prefix test --art artifacts/test \
    --model models/model.txt --hybrid --dense-k 5 --rule expected_f --out output_stage1   # CPU ~90 min
# stage 2: cross-encoder on 0.005<p<0.995, stage-2 model, decision -> final TSVs
python -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base_fp16 \
    --stage2 models/stage2.txt --out output
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (candidates = every pair the
stage-1 model scored; all final matches are a subset).

## Retrain from scratch
Entity hash buckets (hash(S1 id) % 8) keep every learned component on disjoint entities:
bucket 0 = pair classifier (+ stage 2, via its out-of-fold predictions), 1 = bi-encoder, 2 = cross-encoder.
```bash
T=dataset/train; GT=$T/train_ground_truth.tsv
python -m er.train_full --split-dir $T --prefix train --art artifacts/train --gt $GT          # sparse cands + v2 model
python -m er.biencoder train  --art artifacts/train --gt $GT --out models/e5er                 # bucket 1
python -m er.biencoder encode --art artifacts/train --model models/e5er
python -m er.biencoder search --art artifacts/train --k-other 10 --k-s1 10
python -m er.train_full --split-dir $T --prefix train --art artifacts/train --gt $GT --hybrid --dense-k 5
python -m er.add_ambiguity --art artifacts/train --tag _hyb5        # (features.py computes these natively)
python -m er.train_full --split-dir $T --prefix train --art artifacts/train --gt $GT --hybrid --dense-k 5
python -m er.cross_encoder train --art artifacts/train --gt $GT --base intfloat/multilingual-e5-base --buckets 2 3 --max-pairs 1200000 --bs 32 --lr 2e-5 --out models/ce_base   # buckets 2-3
python -m er.cross_encoder score --art artifacts/train --pairs artifacts/train/oof_hyb5.parquet \
    --band 0.005 0.995 --model models/ce_base --out artifacts/train/ce_oof.parquet
python -m er.exp_stage2 --art artifacts/train --gt $GT --tag _hyb5 --ce artifacts/train/ce_oof.parquet
# -> artifacts/train/model_hyb5.txt (stage 1), artifacts/train/stage2_hyb5_ce.txt (stage 2)
```

## Pipeline (src/er)
| module | role |
|---|---|
| `normalize.py`, `prep.py` | script-agnostic normalisation, streaming TSV → parquet |
| `blocking.py` | sparse TF-IDF retriever |
| `biencoder.py` | bi-encoder fine-tuning (hard negatives), memmap encoding, exact tiled GPU top-k |
| `candidates.py` | hybrid sparse ∪ dense candidates + unbiased dense context features |
| `features.py`, `add_ambiguity.py` | string / number / name-ambiguity pair features |
| `train_full.py` | full-density stage-1 training, entity-grouped CV |
| `predict.py` | chunked per-country stage-1 inference |
| `cross_encoder.py` | cross-encoder training and scoring |
| `stage2.py`, `exp_stage2.py`, `predict_stage2.py` | cluster context, stage-2 training, test-time stage 2 + TSVs |
| `decide.py` | exclusivity, expected-F0.5 rule, official metric |
