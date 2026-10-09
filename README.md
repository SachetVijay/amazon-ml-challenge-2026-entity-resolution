# Amazon ML Challenge 2026 — Business Entity Resolution (Team AMS)

## 🏆 Final rank: 21 (out of 89,000+ participants)
**Public leaderboard score: 0.990367 macro F0.5**

A 72-hour hackathon run by Amazon, open to engineering students across India, on resolving noisy
business records across three independent sources with no shared ID. Full write-up of the approach,
every ablation, and the reasoning behind each design choice: [`package/Documentation_final.md`](package/Documentation_final.md).

---

Given three independent, noisy sources of business records (Source 1 = deduplicated reference entities;
Source 2/3 = noisy records to resolve against them), find every Source 2/3 record that refers to each
Source 1 entity. Test includes a country (France) with zero training labels.

## Start here

- **`package/README_final_submit.md`** — exact, runnable commands to reproduce the final submission
  end to end (normalize → block → classify → re-rank → decide), with pinned dependencies.
- **`package/Documentation_final.md`** — the full methodology write-up: problem analysis, architecture,
  every ablation, error analysis, and the negative results (what we tried that didn't work).
- **`src/er/`** — the pipeline's source code, organized by stage (`normalize`, `blocking`, `biencoder`,
  `features`, `train_full`, `cross_encoder`, `stage2`, `exp_band*`, `decide`, `predict*`).
- **`scripts/`** — the ad-hoc orchestration scripts actually run during the 72-hour sprint (not a clean
  API; useful mainly as a record of the exact commands and sequencing behind each dated version).

## Architecture, in one picture

```
normalize (transliterate, canonicalize legal forms/abbreviations, skeleton keys)
   │
   ├─ sparse TF-IDF top-5 (both directions) ─┐
   └─ fine-tuned bi-encoder top-5 (both dir.) ┴─ hybrid candidate set (~53/S1, 99.2% recall)
                                                       │
                     69 pair features ──► LightGBM stage-1 classifier ──► p₁
                                                       │
                     learned filter p₁ > 0.001 ──► candidate_pairs.tsv (5.67/S1)
                                                       │
                0.005 < p₁ < 0.995 ──► 6 cross-encoders (disjoint entity buckets) ──► logits
                                                       │
         stage-2 LightGBM (outside band)   second-pass band specialist (inside band:
                                            all features + 6 logits + rival-strength context)
                                                       │
         exclusive assignment ──► per-S1 expected-F0.5 rule
                                                       │
         [country unseen in training] 4-logit fallback + pseudo-label-adapted cross-encoder
         + post-decision demotion of generator-pattern category swaps
                                                       │
                                        matching_results.tsv + candidate_pairs.tsv
```

Full reasoning for every box above — why hybrid blocking, why the entity-bucket leakage control, why
exclusive assignment + expected-F0.5 instead of a threshold, how the unseen-country (France) problem was
solved with zero labels — is in `package/Documentation_final.md`.

## Compute

Trained on one personal workstation (24 CPU threads, 10 GB RAM, one 8 GB consumer GPU) plus free
Kaggle T4 GPU kernels for the two largest cross-encoders (`src/er/kaggle_train_ce.py`, runs on any CUDA
machine). No cloud infrastructure details are specific to this repo — swap in whatever machine you have.

## Data

The competition dataset is not redistributed here (`student_resource/` is gitignored) — get it from the
official challenge page. Trained model weights are also not included (organizers confirmed they aren't
required for the final submission); every model is reproducible from the training commands in
`package/README_final_submit.md`.

## Compliance

All models MIT-licensed (largest 560M params, well under the 8B limit); no external data or lookups —
see the compliance notes in `package/Documentation_final.md`.
