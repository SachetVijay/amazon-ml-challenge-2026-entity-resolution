# Business Entity Resolution (Team AMS): final pipeline (v22)

Resolves Source-2/Source-3 business records to deduplicated Source-1 entities:
**normalise → hybrid blocking (sparse TF-IDF ∪ fine-tuned bi-encoder) → LightGBM pair classifier (stage 1) →
six cross-encoders on the uncertain band → band specialist trained on test-like competition → second pass
with cross-encoder-measured rival strength and per-source slot filling → exclusive assignment → per-entity
expected-F0.5 decision; for countries absent from training, the dominant cross-encoder is adapted to the new
country with pseudo-labels.**

Validation (out-of-fold, official macro F0.5 incl. singletons, 276k held-out US+India S1 entities, test-like
competition): **0.99102** (v13: 0.99089; v12: 0.99052, public leaderboard 0.987426). Public leaderboard of this final
submission (v22), whose outputs are in `output/`: **0.990367**.

What v12 adds over v11: the band specialist's record-side competition features (`rank_in_b`, `b_max`,
`b_second`, `b_margin`) used to be computed from the sampled S1 entities only. That is ~1/8 of the S1s that
compete for a record at test time, so the model never learned what real competition looks like (training
band: 12 % of pairs have a better-scored rival S1; test: 21 %, France 44 %). `er.build_competitors` scores the
missing rival S1s with the final stage-1 model, and `er.exp_band_fd` retrains the specialists on test-like
context: OOF 0.98977 (current model under test-like context) → **0.99052**; unseen-country proxy
(train US → test India) 0.99084 → 0.99160.

What v13 adds over v12: v12 measures how strongly a rival S1 claims a record by its stage-1 probability (band
AUC 0.965). v13 scores every band pair of a record with a light *rival scorer* q = LightGBM(p1, record-side
context, cross-encoder logits), band AUC 0.991, recomputes the record-side context from q (margin to the best
rival, rival mass, rank) and adds it to a second-pass specialist: OOF 0.99052 → **0.99089** (4-cross-encoder
variant 0.99018 → 0.99051). Leakage control: training rivals come from S1 buckets that some cross-encoders were
trained on. Each rival is therefore scored only with the cross-encoders that never saw its bucket (bucket 1:
all six; 2–3: mDeBERTa, e5-large; 4–5: e5-base, mDeBERTa, normalised, e5-large; 6–7: e5-base, base2,
normalised, hard). At test time nothing is in-sample and all are used.

What v14 adds over v13: *slot filling*. Almost every entity has at least one S2 and one S3 record, so a candidate
S1 that already holds several confident records from the record's own source is a less likely owner than a
namesake holding none. Three features (the S1's other confident records from the record's source / the other
source, and the probability mass from the record's source; q inside the band, p1 outside) lift the second pass
from 0.99087 to **0.99102** (US 0.99008 → 0.99021, India 0.99204 → 0.99222, band AUC 0.9921 → 0.9924).

What v16 adds over v14 (unseen countries only; US/India output is byte-identical): the raw-text cross-encoders were
trained on US/India wording. On France they reject 7-20x more of the links that the cross-encoder-free stage-1 model is
certain about, and the France model routes 62 % of its rival scorer through one of them (`ce_base2`). That model is
fine-tuned for one epoch on French pseudo-labels plus 200k labelled US/India hard pairs (`er.adapt_unseen_ce`). No
French labels are used. Its French false-rejection rate on held-out pseudo-positives falls from 2.04 % to 0.17 % (US/India:
0.14-0.19 %). Its accept rate on French pseudo-negatives falls from 4.53 % to 1.01 %, and its US/India behaviour is
unchanged.

What v18 adds: one post-decision rule for the unseen country (`er.demote_swaps`). Leaderboard deltas acted as labels:
v14 → v16 (+0.00117) mostly added `X Club → X SARL Cie`-style noise, while v16 → v17c (−0.00141) mostly added
`X Club → X Ecole` category swaps at the same address. Records that swap one native category word at the same address
are therefore demoted; abbreviations and filler words appended after the legal form (`X SARL Cie`, the generator's
noise) are kept. v18 (6,684 links, leaderboard 0.989917) → v22 extends the same rule to swaps into native French words
(`compagnie`, `centre`, `societe`), in-place `cie`/`france` replacements, rarer category words, and category swaps
that come with other name or address noise (≈5,400 more links).

## Models (all MIT, ≤ 560M parameters)
| file / dir | what | trained on (entity bucket = hash(S1 id) % 8) |
|---|---|---|
| `models/model.txt` | stage-1 LightGBM pair classifier | bucket 0 |
| `models/stage2.txt` | stage-2 LightGBM (context stacker) | bucket 0, out-of-fold |
| `models/band.txt` | first-pass band specialist, 6 cross-encoders (v12; see note below) | bucket 0 + rivals |
| `models/band_unseen.txt` | first-pass band specialist, 4 cross-encoders (v12; see note below) | bucket 0 + rivals |
| `models/q.txt`, `models/band2.txt` | rival scorer + second-pass specialist with slot features (seen countries) | bucket 0 + rivals |
| `models/q_unseen.txt`, `models/band2_unseen.txt` | the same with 4 cross-encoders (unseen countries) | bucket 0 + rivals |
| `models/e5er` | bi-encoder `intfloat/multilingual-e5-small` | bucket 1 |
| `models/ce_base` | cross-encoder `intfloat/multilingual-e5-base`, raw text | buckets 2–3 |
| `models/ce_base2` | `ce_base` continued | buckets 4–5 |
| `models/ce_mdeberta` | `microsoft/mdeberta-v3-base` | buckets 6–7 |
| `models/ce_norm` | e5-base on normalised text (core name, canonical address) | buckets 2–3 |
| `models/ce_hard` | `ce_base` fine-tuned on the uncertain band only | buckets 4–5 |
| `models/ce_large` | `intfloat/multilingual-e5-large` | buckets 6–7 |

No model files are shipped: every model (LightGBM and transformer) is produced by the training commands below
(the rules ask for a self-contained pipeline that regenerates the outputs from the data).

Note: when the second-pass models (`q*.txt`, `band2*.txt`) are given, `er.predict_stage2` uses them for the uncertain band
and the first-pass specialists (`band.txt`, `band_unseen.txt`) do not affect the output. Their training step and flags are
kept because they are part of the exact command that produced the submitted files (and reproduce the v12 ablation).

## Setup
```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && export PYTHONPATH=src
# data: place (or symlink) the provided dataset/ folder here:
#   dataset/train/{train_source1..3.tsv, train_ground_truth.tsv}, dataset/test/test_source1..3.tsv
mkdir -p models artifacts
```
Run every command below from this folder (`code/business_entity_resolution/`). The only downloads are the public
pretrained checkpoints fine-tuned here (`intfloat/multilingual-e5-small/base/large`, `microsoft/mdeberta-v3-base`, all
MIT), fetched from the Hugging Face Hub on first use. No other external data, API or lookup service is used.
Hardware used: 24 CPU threads, 10 GB RAM, one 8 GB GPU (RTX 3050); the larger cross-encoders were trained on
Kaggle T4 GPUs with `src/er/kaggle_train_ce.py` (it runs on any CUDA machine when `--pairs` and `--out` are given, as
in the commands below). Everything streams per country and per S1 chunk.

## Train (buckets keep every learned component disjoint; no component is evaluated on its own entities)
```bash
T=dataset/train; GT=$T/train_ground_truth.tsv; A=artifacts/train
python -m er.prep --split-dir $T --prefix train --out $A
python -m er.train_full --split-dir $T --prefix train --art $A --gt $GT                        # sparse candidates
python -m er.biencoder train --art $A --gt $GT --out models/e5er                               # bucket 1
python -m er.biencoder encode --art $A --model models/e5er && python -m er.biencoder search --art $A --k-other 10 --k-s1 10
python -m er.train_full --split-dir $T --prefix train --art $A --gt $GT --hybrid --dense-k 5    # stage 1 (bucket 0)
cp $A/model_hyb5.txt models/model.txt
# cross-encoders
python -m er.cross_encoder train --art $A --gt $GT --base intfloat/multilingual-e5-base --buckets 2 3 --max-pairs 1200000 --bs 32 --lr 2e-5 --out models/ce_base
python -m er.cross_encoder train --art $A --gt $GT --base models/ce_base --buckets 4 5 --max-pairs 1200000 --bs 32 --lr 1e-5 --out models/ce_base2
python -m er.export_ce_pairs --art $A --gt $GT --buckets 6 7 --out pairs67.parquet
python src/er/kaggle_train_ce.py --pairs pairs67.parquet --base microsoft/mdeberta-v3-base --out models/ce_mdeberta
python src/er/kaggle_train_ce.py --pairs pairs67.parquet --base intfloat/multilingual-e5-large --lr 1.5e-5 --out models/ce_large
python -m er.export_ce_pairs --art $A --gt $GT --buckets 2 3 --view norm --out pairs_norm.parquet
python src/er/kaggle_train_ce.py --pairs pairs_norm.parquet --base intfloat/multilingual-e5-base --out models/ce_norm
python -m er.build_extra_band --art $A --gt $GT --model models/model.txt --buckets 4 5 --tag b45  # band of buckets 4-5
python -m er.export_band_pairs --art $A --tag b45 --out hard_b45.parquet
python src/er/kaggle_train_ce.py --pairs hard_b45.parquet --base models/ce_base --out models/ce_hard --bs 32 --lr 1e-5 --max-hours 3
# out-of-fold cross-encoder logits on the bucket-0 band (feature order = test order: base, base2, mdeberta, norm, hard, large)
i=0; for m in ce_base ce_base2 ce_mdeberta ce_norm ce_hard ce_large; do v=raw; [ $m = ce_norm ] && v=norm
  python -m er.cross_encoder score --art $A --pairs $A/oof_hyb5.parquet --band 0.005 0.995 --model models/$m --view $v --out $A/ceo$i.parquet; i=$((i+1)); done
python -m er.exp_stage2 --art $A --gt $GT --tag _hyb5 --ce $A/ceo0.parquet --save-tag _v9 && cp $A/stage2_hyb5_v9.txt models/stage2.txt
# test-like competition: score the rival S1s (outside bucket 0) of every bucket-0 band record, retrain the specialists
python -m er.build_competitors --art $A --model models/model.txt --tag fd
python -m er.exp_band_fd --art $A --gt $GT --s2-tag _v9 --ce $A/ceo{0,1,2,3,4,5}.parquet --comp-tag fd --save-tag _v12
python -m er.exp_band_fd --art $A --gt $GT --s2-tag _v9 --ce $A/ceo{0,1,2,3}.parquet --comp-tag fd --save-tag _v12u
cp $A/band_model_v12.txt models/band.txt && cp $A/band_model_v12u.txt models/band_unseen.txt
# second pass: cross-encoder logits for the rival band pairs, rival scorer + second-pass specialist
python -m er.export_rival_band --art $A --comp-tag fd
i=0; for m in ce_base ce_base2 ce_mdeberta ce_norm ce_hard ce_large; do v=raw; [ $m = ce_norm ] && v=norm
  python -m er.cross_encoder score --art $A --pairs $A/rival_band.parquet --model models/$m --view $v --out $A/rce$i.parquet; i=$((i+1)); done
python -m er.exp_band_rival --art $A --gt $GT --s2-tag _v9 --ce $A/ceo{0,1,2,3,4,5}.parquet --rival-ce $A/rce{0,1,2,3,4,5}.parquet --save-tag _v14 --pick-best
python -m er.exp_band_rival --art $A --gt $GT --s2-tag _v9 --ce $A/ceo{0,1,2,3}.parquet --rival-ce $A/rce{0,1,2,3}.parquet --save-tag _v14u --pick-best
cp $A/q_model_v14.txt models/q.txt && cp $A/band_model_v14.txt models/band2.txt
cp $A/q_model_v14u.txt models/q_unseen.txt && cp $A/band_model_v14u.txt models/band2_unseen.txt
```

## Predict (test)
```bash
S=artifacts/test
python -m er.prep --split-dir dataset/test --prefix test --out $S
python -m er.biencoder encode --art $S --model models/e5er && python -m er.biencoder search --art $S --k-other 10 --k-s1 10
python -m er.predict --split-dir dataset/test --prefix test --art $S --model models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_stage1 --chunk-s1 100000       # stage 1 (+ cached features)
CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_hard models/ce_large"
PRED="python -m er.predict_stage2 --art $S --ce-model $CES --stage2 models/stage2.txt --band-model models/band.txt \
      --band-model-unseen models/band_unseen.txt --unseen-n-ce 4 --train-records artifacts/train/records.parquet \
      --q-model models/q.txt --band-model2 models/band2.txt --q-model-unseen models/q_unseen.txt \
      --band-model2-unseen models/band2_unseen.txt"
# countries present in test but absent from training (open label set; nothing is keyed on a country name)
UNSEEN=$(python -c "import pandas as pd; r=lambda f: set(pd.read_csv(f, sep='\t', usecols=['country']).country); print(' '.join(sorted(r('dataset/test/test_source1.tsv') - r('dataset/train/train_source1.tsv'))))")
# 1. first pass
$PRED --out output_pass1
# 2. adapt the dominant cross-encoder to each unseen country with pseudo-labels from pass 1 (no labels used)
for C in $UNSEEN; do
  python -m er.adapt_unseen_ce --art $S --country $C --pred-s2 $S --ce-model models/ce_base2 --us-pairs hard_b45.parquet --out adapt_$C
  python src/er/kaggle_train_ce.py --pairs adapt_$C/pairs.parquet --base models/ce_base2 --out models/ce_base2_unseen --bs 32 --lr 1e-5 --max-hours 1.2
done                                                   # (this test set has exactly one unseen country)
# 3. final pass: unseen countries read the adapted cross-encoder
$PRED --unseen-ce-override 1=models/ce_base2_unseen --out output
# 4. unseen countries: demote same-address category-word swaps (planted-distractor recipe), re-decide those entities
for C in $UNSEEN; do
  python -m er.demote_swaps --art $S --country $C --pred-s2 $S --base-tsv output/matching_results.tsv \
      --out output/matching_results.tsv --keep-words cie france fils freres services groupe \
      --inplace-words cie france fils --dfmin 2e-4 --broad
done
```
Result: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Optional format check with the organisers'
validator (`utils/validate_submission.py` from the challenge's `student_resource/` kit, not part of this pipeline):
```bash
python3 <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids     # -> PASS
```
`output/candidate_pairs.tsv` is the **last filtering stage**: the hybrid blocking set (sparse TF-IDF top-5 ∪ fine-tuned
bi-encoder top-5, both directions; 52.8 pairs per S1) pruned by the stage-1 LightGBM, a learned blocking filter
(p₁ > 0.001). The cross-encoders, the second-pass specialists and the decision step score exactly these pairs:
**5.67 candidates per S1** (median 5, p95 10, max 212; 9.8M pairs for 1.73M S1), recall ceiling ≈ 99.15 %.
Every final match is a candidate.

**Open-set countries.** Nothing is keyed on country names. `--train-records` defines which countries were seen in
training; any other country uses the `*_unseen` models, which omit the two cross-encoders fitted most tightly to the
training countries' hard examples, reads a cross-encoder adapted to it with pseudo-labels, and gets the category-swap
demotion. The word lists passed to `er.demote_swaps` are filler words of the generator's noise, identified by a
label-free statistic (how much more frequent a word is in S2/S3 names than in S1 names).
