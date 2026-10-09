# Forensic diagnosis of the current solution (v11l validation baseline)

The diagnosis is finished and nothing was modified. No repo or pipeline file was edited, and nothing under `~/amlc` on the compute machine was written. On the compute machine I only ran read-only, streaming extractions at `nice 19` with a memory watchdog, writing to a separate `~/amlc_diag/`. All heavy analysis ran on the Mac. The v12 chain ran undisturbed: its band model finished validating, and its test prediction step was still running when I finished.

**Population.** 276,044 validation S1 entities (US 165,294, India 110,750), 956,669 true pairs, 5.43% singletons. The baseline reproduces the logged v11l score exactly (0.98902), and labels agree with the pipeline's own with 0 mismatches.

## Phase 1 — what the pipeline does

**Matching path:**
- **Normalization:** `anyascii` transliteration, legal-form canonicalization, a generic-word stop list, abbreviations, a consonant "skeleton" key, and US/India/native-script state and French region↔department canonicalization. Output goes to `records.parquet`.
- **Blocking:** runs per country label.
  - Sparse: TF-IDF "all" retriever, top-5 in both directions (S1→record and record→S1), cosine ≥ 0.05, absolute document-frequency cap of 12,000.
  - Dense: fine-tuned e5-small, top-5 in both directions.
  - The two are unioned into the hybrid candidate set.
- **Stage 1:** ~70 features and LightGBM give p1 for every hybrid pair.
- **Stage 2:** pairs with p1 ≤ 1e-3 are dropped. Six cross-encoders score the band 0.005 < p1 < 0.995, then the stage-2 GBDT runs, then the band specialist. France gets the separate unseen-country band model.
- **Decision:** exclusive assignment (each record goes to its best S1), then per-S1 expected-F0.5 top-m selection → `matching_results.tsv`.

**`candidate_pairs.tsv`** is the full hybrid set, written by `predict_stage2` from `pred_hyb_*.cand`.

**Places where candidates can be lost:**
- country grouping;
- normalization;
- the sparse document-frequency cap, k=5 and the 0.05 cosine threshold;
- the dense k=5 cap;
- the p1 > 1e-3 drop;
- exclusivity;
- the expected-F cut.

## Phase 2 — the blocking ceiling

**A. Recall: 99.158% overall (8,059 true pairs missing).**

| Slice | Recall |
|---|---|
| US | 99.139% |
| India | 99.186% |
| US S2 / S3 | 99.06% / 99.21% |
| India S2 / S3 | 99.28% / 99.09% |
| S1 with 1 / 2 / 3 / 4–5 / 6+ true matches | 99.28 / 99.39 / 99.32 / 99.10 / 99.00% |

- 97.03% of matched S1 entities have all their true matches retrieved; 0.044% have none retrieved.
- Entity completeness falls with the number of true matches: 99.3% (1) → 94.0% (6+).
- Singletons have no true pairs, so recall doesn't apply; their candidate lists are the same size as everyone else's (44.2 vs 43.8).

**B. Candidate-set size per S1.**

| Set | Mean | Median | p90 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| Train (validation sample) | 43.8 | 28 | 71 | 107 | 386 | 3,561 |
| Test `candidate_pairs.tsv` | 52.8 | 35 | ~90 | 129–140 | 299–441 | 6,742 |

Candidate count is uncorrelated with the number of true matches (r = 0.006): the size is driven by hubs, not by entities with many matches.

**C. Reduction ratio.** On test, 91.5M candidate pairs against 6.7×10¹² same-country pairs gives RR 0.9999864. Against all cross-country pairs (1.7×10¹³) it is 0.9999947.

**D. Why the 8,059 true pairs were lost.**

| Mechanism | Count | % |
|---|---|---|
| Dense top-5 cap (true S1 sits at dense rank 6–10) | 3,566 | 44.2 |
| Same-name S1s fill the record's top-5 | 2,403 | 29.8 |
| Address-less record, name OK but outranked | 1,069 | 13.3 |
| Name rewritten, only the address agrees | 664 | 8.2 |
| Address-less record with a corrupted name | 227 | 2.8 |
| Moderate similarity on both fields / both corrupted | 92 | 1.1 |
| Native script (India only) | 35 | 0.4 |
| Normalization failure | 1 | 0 |
| Country filter | 0 | 0 |
| Post-blocking filter / source-specific bug | 0 | 0 |

- Cross-country true pairs: 0. No record matches more than one S1.
- **84% of the missed pairs are address-less records.**
- Sparse retrieval uniquely contributes only 953 true pairs (0.1%). The dense record→S1 direction uniquely contributes 21,643.

## Phase 3 — is blocking too loose?

End-to-end F0.5 below was measured by restricting the current v11l probabilities to each pruned set.

| Strategy | Recall | Candidates/S1 | Median | p95 | p99 | Macro F0.5 |
|---|---|---|---|---|---|---|
| **Current (sparse top-5 ∪ dense top-5)** | 99.158% | 43.8 | 29 | 111 | 411 | 0.98902 |
| Drop the sparse record→S1 direction | 99.079% | 25.7 | 21 | 59 | 93 | 0.98898 |
| Sparse top-1 ∪ dense top-5 | 99.075% | 25.1 | 20 | 60 | 100 | 0.98900 |
| Dense top-5 only | 99.007% | 23.8 | 19 | 57 | 91 | 0.98893 |
| Dense top-3 only | 98.561% | 14.1 | 12 | 32 | 51 | 0.98879 |
| Sparse top-5 only | 87.806% | 24.5 | 9 | 77 | 390 | 0.93815 |
| **Current ∩ p1 > 1e-3 (= the actual stage-2 input)** | **99.146%** | **4.8** | 5 | 8 | 11 | **0.98902** |
| Current ∩ top-5 by p1 | 94.419% | 5.0 | 5 | 5 | 5 | 0.98363 |

**Verdict:** recall is close to the frontier, but the set is **far too loose in size**. The sparse record→S1 direction is 53% of all pairs and contributes 0.08% unique recall. On test, the p1 > 1e-3 set is 9.8M pairs (5.7 per S1; France 7.3), about 9× smaller, with the same score.

## Phase 4 — the matcher on the fixed candidate set

**Pair level (final decision):** precision 0.99736, recall 0.97166 against all true pairs. Probabilities are well calibrated; for example, exclusive winners with p in 0.5–0.7 are 57.7% positive.

**Threshold sweep, entity-level macro F0.5 (exclusive + fixed threshold vs expected-F):**

| Rule | Macro F0.5 |
|---|---|
| Threshold 0.5 | 0.98765 |
| Threshold 0.6 | 0.98861 |
| Threshold 0.7 | 0.98903 |
| Threshold 0.8 | 0.98891 |
| Expected-F (current) | 0.98902 |
| Expected-F, empty_weight 1.5 | 0.98908 |

**The decision rule is not the bottleneck.**

**Entity categories (total loss 0.01098):**

| Category | S1 entities | Share of loss |
|---|---|---|
| 4. One or more correct, missed others | 24,402 (8.8%) | **60.1%** |
| 1. Predicted empty but has matches | 462 | 15.2% |
| 2. Correct plus extra false matches | 1,906 (+252 that also missed some) | 17.7% |
| 3. Singleton falsely matched | 203 (1.35% of singletons) | 6.7% |
| 5. All predictions wrong | 10 | 0.3% |

- **6. Small margins:** 2,206 S1 have a decision margin under 0.3 (5.2% of loss); 11,942 have 0.3–0.6 (23.1%).
- **7. Abstention:** of 392 S1 predicted empty despite a candidate with p > 0.2, 67% really are singletons, so those abstentions are correct.

**Oracle on the current candidates is 0.99771.** Blocking costs 0.00229 of F0.5; the matcher and decision cost 0.00869.

**False negatives inside the candidate set: 19,052.** 16,121 won exclusivity and were then cut by expected-F (median p about 0.3); 2,459 lost exclusivity to another S1.

**False positives: 2,456.** 65% are records whose true owner is an S1 outside the validation bucket; 33% are planted distractors that belong to no S1; 2% belong to another S1 in the sample.

## Phase 5 — systematic error patterns

- **Address-less records (G):** 3.7% of true pairs but **74% of in-candidate false negatives**. Recall on them is 0.604, vs 0.9945 for records with an address.
  - With an exact core-name match, the positive rate is **1/k when k S1s share the name**: 0.976 (k=1), 0.495 (2), 0.30 (3–4), 0.18 (5–10), 0.04 (>10).
  - Mean p equals the positive rate, and AUC within each group is only 0.79–0.82. This is label-level ambiguity, not a broken boundary.
- **Near-identical names but different entities (D):** 966 false positives (39% of all). The top false positives are exact name+address copies at p = 1.0 with no owner, e.g. `Foot & Ankle Group | 145 Wildwood Way`. These are planted and irreducible.
- **Chain/common names (M), ≥5 S1 sharing the name:** 44% of false negatives and 28% of false positives.
- **House-number conflicts:** 25% of false positives and 15% of false negatives.
- **Weak name + strong address (E):** precision 0.990, recall 0.968.
- **Solved patterns:**
  - typos: precision 0.9975, recall 0.987;
  - non-Latin scripts: precision 0.9991, recall 0.9975;
  - word order: precision 0.998, recall 0.982;
  - acronyms and legal suffixes: fine.
- **Multi-match S1s (N):** entities with ≥3 true matches score F 0.9913. **Single-match entities are the worst group** (F 0.964, 18% of loss). Singletons (O) score 0.9865.
- **Category-word swaps at the same address** are **3.4% true** in training (877 pairs). The data generator uses them to plant distractors.

## Phase 6 — field ablation

Stage 1 only, on a 1/8 subsample (34,471 S1, 2-fold entity-grouped CV). Baseline 0.97255; the full pipeline on the same entities scores 0.98921.

| Removed | ΔF0.5 |
|---|---|
| All name features | −0.0268 |
| All address features | −0.0201 |
| Token overlap | −0.0095 |
| House numbers | −0.0082 |
| Dense bi-encoder | −0.0075 |
| Competition/context | −0.0035 |
| Character similarity | −0.0029 |
| Name frequency | −0.0024 |
| Exact indicators | −0.0005 |
| Sparse TF-IDF scores | −0.0002 |
| Source flag | −0.0001 |
| Keep dense only | −0.163 |
| Keep strings only | −0.009 |

Country is deliberately not a feature, so it can't be ablated. Every major signal is used. Together with calibrated probabilities and a flat decision curve, **the residual is information-limited, not a poor decision boundary.**

## Phase 7 — validation audit

- **No pair leakage.** Splits are entity-grouped (hash mod 8 buckets; the bi-encoder and cross-encoders are trained on other buckets). No feature is unavailable at test time.
- **Validation is pessimistic, not optimistic.** Validation contexts contain only one-eighth of the competing S1s: the share of band records with a better-scored rival goes from 12% to 31% at test density. Measured effect on the current model: 0.98902 → 0.98977 on test-like context (v12 log). My exclusivity proxy gives about +0.0002 for the US.
- **Selection on a reused validation set.** Many versions were chosen on the same bucket-0 set, so treat deltas under about 0.0003 as noise.
- **France is unvalidated.** The label-free estimate overshoots the US by 0.0045, so France's 0.981 estimate is realistically about 0.975. The public leaderboard implies France is about 0.967–0.97.

## Phase 8 — source and country robustness

- **S2 vs S3:** balanced on recall and on false positives.
- **US vs India:** India scores better (0.9902 vs 0.9883).
- **France is harder, and the model is behaving correctly:**
  - it has 2.7× the uncertain winners per S1 (0.43 vs 0.16 US, 0.13 India);
  - its names are low-entropy: the top token "club" appears in 8.8% of S1 names, vs 3.2% for the top US token;
  - 26% of its uncertain pairs differ only by a category word (2.6% US, 5.3% India);
  - since training says such swaps are 97% distractors, the low France probabilities are most likely right.
- **No accidental hard-coding.** The pipeline iterates over whatever country labels exist; test labels are clean (3 values); the unseen-country branch keys on "not in training"; no country feature exists.
- **One review risk.** The state, native-script and French region dictionaries in `normalize.py` are static domain knowledge applied to every record. The French entries were written after inspecting unlabelled test France records. Document this plainly as knowledge-based rules, not an external lookup.

## Phase 9 — where the plateau is

**A. Blocking ceiling**
- Candidate recall: 99.158% (97.0% of entities complete).
- Reduction ratio: 0.9999864 on test, at 52.8 candidates per S1 (max 6,742).
- Main failure modes: address-less records (84%), the dense top-5 cap (44%), same-name crowding (30%).

**B. Matching ceiling**
- Candidate-conditioned F0.5: 0.98902, against an oracle of 0.99771.
- False-positive modes: planted duplicates with no owner, records belonging to same-name S1s, house-number conflicts.
- False-negative modes: address-less same-name records (74%), renamed or coined names, shifted house numbers.

**C. Entity-level ceiling**
- Singleton errors: 203 entities (0.0007 of F).
- Multi-match errors: missed extra links (0.0066).
- Abstention errors: empty predictions for entities with matches (0.0017).
- Top problem groups: address-less chain names; single-match entities.

**D. Validation reliability**
- Trustworthy for US/India (slightly pessimistic).
- Questionable for France and for deltas under 0.0003.

**E. Top 3 bottlenecks**

| # | Bottleneck | Evidence | Importance / impact | Difficulty and hidden-test generalization |
|---|---|---|---|---|
| 1 | **France** (15% of S1) | ~0.02 below US/India; 2.7× uncertain mass | Largest leaderboard lever: +0.01 on France is about +0.0015 overall. Recall-side (3.31 predicted vs 3.40 expected matches per S1). No candidate-size impact. | Hard: no labels, and no mechanical bug was found. Uncertain generalization. |
| 2 | **Address-less same-name ambiguity** | 74% of in-candidate false negatives, 84% of misses | Upper bound +0.0046, but realistic gain is small because the evidence is calibrated and weak. Mostly recall. | Hard. |
| 3 | **Candidate-set compactness** (reviewed) | 52.8 per S1, max 6,742 | No precision or recall change; about 9× smaller candidate set. | Trivial; exact on hidden test. |

## Phase 10 — experiments, ranked by expected value

**1. Write `candidate_pairs.tsv` from the p1 > 1e-3 set (the stage-2 input).**
- **Hypothesis:** the reviewed candidate set can shrink about 9× with no change to predictions.
- **Change:** in `predict_stage2`, write the filtered stage-1 set instead of the raw hybrid set.
- **Why:** Phase 3 shows the same F0.5, recall down 0.01 points, and every final match is still a subset of the candidates.
- **Expected effect:** test goes from 52.8 to 5.7 per S1 (91.5M → 9.8M pairs); precision, recall and F0.5 are identical.
- **Offline check:** the validator, plus a subset assertion.
- **Leaderboard submission:** not needed.
- **Reject if** you consider stage 1 part of "the matching model". The README says the file must be what the matching model runs inference over. The fallback is dropping the sparse record→S1 direction (25.7 per S1, −0.00004), which is approximate because it perturbs stage-1 context features.
- Either way, document stage 1 as a learned blocking filter.

**2. Submit v12 (band model retrained on test-like competition context).**
- **Evidence:** validated 0.98977 → 0.99052, and leave-one-country-out US→India also gains (+0.0008).
- It addresses the only measured validation/test mismatch.
- **Worth a submission.**
- **Reject if** the leaderboard score is ≥0.0003 below your current best.

**3. France-only calibration-direction probe.**
- **Why:** France carries 2.7× the uncertain mass, so the decision is most leveraged there, and the leaderboard is the only France label.
- **Change:** probe France-only odds calibration, the opposite direction to the earlier v6c probe (odds ×0.5). This is only worth doing if your v11fL/v11fC probe scores don't already answer it.
- **Worth a submission** only if the direction is still unknown.
- **Reject if** it doesn't beat the base run by more than the leaderboard noise.

**4. Dense k = 10 for address-less records.**
- **Expected effect:** recovers up to 3,566 retrieval misses, but they are mostly 1/k-ambiguous, so the gain is at most about +0.0002.
- **Cost:** hours of re-featurizing and cross-encoder scoring.
- **Not recommended today.**

**5. Expected-F empty_weight 1.5.**
- **Expected effect:** +0.00006, which is below validation noise.
- **Reject.**

I'll stop here and wait for your instruction before changing anything.

Two things would help me advise on #2 and #3:
- what your v11, v11fL and v11fC public-leaderboard scores were;
- which of them is your current rank-170 submission.

The diagnostic extracts take about 900 MB of the Mac's 1.5 GB free disk. I can delete them whenever you want.