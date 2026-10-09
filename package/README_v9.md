# Business Entity Resolution — v9 reproduction

v9 = v7 + (c) **native-script Indian state names** (16 forms, e.g. `ಕರ್ನಾಟಕ`, `महाराष्ट्र` → canonical state) and **mojibake repair** (`Â`, `Ï¿½`); (d) a 4th cross-encoder reading **normalised text** (core name | canonical address). v7 = v6 + (a) **French region ↔ department canonicalisation** (all 13 metropolitan regions / 96 departments →
one region token), French street abbreviations (`R.`, `All.`, `Ch.`, `Q.`, `Crs`) and `street/saint/st`
unification; (b) a **band specialist**: LightGBM trained only on pairs with 0.005 < p₁ < 0.995, using the full
stage-1 feature vector + cluster context + logits of **three cross-encoders** (e5-base, e5-base continued on fresh
entities, mDeBERTa-v3-base).
Validation (out-of-fold, official macro F0.5 incl. singletons, US+India): **0.9886** (v7: 0.9885, v6: 0.9868).
Models (all MIT): LightGBM ×3 (stage 1, stage 2, band specialist); `intfloat/multilingual-e5-small` bi-encoder;
cross-encoders `intfloat/multilingual-e5-base` ×3 (raw text ×2, normalised text ×1) and `microsoft/mdeberta-v3-base` (≤ 278M parameters each).
Entity hash buckets (hash(S1 id) % 8) keep every learned component disjoint: 0 = pair classifier / stage 2 / band
specialist (out-of-fold), 1 = bi-encoder, 2–3 = e5-base cross-encoder, 4–5 = its continuation, 6–7 = mDeBERTa; the normalised-text cross-encoder also uses buckets 2–3.

## Setup
```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && export PYTHONPATH=src
```
Transformer weights ship separately (`models/e5er_fp16`, `models/ce_base`, `models/ce_base2`, `models/ce_mdeberta`, `models/ce_norm`).

## Reproduce the test predictions
```bash
python -m er.prep --split-dir dataset/test --prefix test --out artifacts/test
python -m er.biencoder encode --art artifacts/test --model models/e5er_fp16
python -m er.biencoder search --art artifacts/test --k-other 10 --k-s1 10
python -m er.predict --split-dir dataset/test --prefix test --art artifacts/test --model models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_stage1            # stage 1 (+ caches features)
python -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm \
    --stage2 models/stage2.txt --band-model models/band.txt --out output  # cross-encoders + stage 2 + band specialist
```

## Retrain
```bash
T=dataset/train; GT=$T/train_ground_truth.tsv; A=artifacts/train
python -m er.train_full --split-dir $T --prefix train --art $A --gt $GT                        # sparse candidates
python -m er.biencoder train --art $A --gt $GT --out models/e5er                               # bucket 1
python -m er.biencoder encode --art $A --model models/e5er && python -m er.biencoder search --art $A --k-other 10 --k-s1 10
python -m er.train_full --split-dir $T --prefix train --art $A --gt $GT --hybrid --dense-k 5    # stage 1 (bucket 0)
python -m er.cross_encoder train --art $A --gt $GT --base intfloat/multilingual-e5-base --buckets 2 3 --max-pairs 1200000 --bs 32 --lr 2e-5 --out models/ce_base
python -m er.cross_encoder train --art $A --gt $GT --base models/ce_base --buckets 4 5 --max-pairs 1200000 --bs 32 --lr 1e-5 --out models/ce_base2
python -m er.export_ce_pairs --art $A --gt $GT --buckets 6 7 --out pairs67.parquet             # mDeBERTa data
python src/er/kaggle_train_ce.py --pairs pairs67.parquet --base microsoft/mdeberta-v3-base --out models/ce_mdeberta
for m in ce_base ce_base2 ce_mdeberta; do python -m er.cross_encoder score --art $A --pairs $A/oof_hyb5.parquet --band 0.005 0.995 --model models/$m --out $A/$m.oof.parquet; done
python -m er.exp_stage2 --art $A --gt $GT --tag _hyb5 --ce $A/ce_base.oof.parquet --save-tag _v7
python -m er.export_ce_pairs --art $A --gt $GT --buckets 2 3 --view norm --out pairs_norm.parquet
python src/er/kaggle_train_ce.py --pairs pairs_norm.parquet --base intfloat/multilingual-e5-base --out models/ce_norm
python -m er.cross_encoder score --art $A --pairs $A/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_norm --view norm --out $A/ce_norm.oof.parquet
python -m er.exp_band --art $A --gt $GT --tag _hyb5 --s2-tag _v7 --ce $A/ce_base.oof.parquet $A/ce_base2.oof.parquet $A/ce_mdeberta.oof.parquet $A/ce_norm.oof.parquet
```
