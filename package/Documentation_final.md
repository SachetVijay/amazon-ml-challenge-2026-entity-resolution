# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team AMS
**Team Members:** Sachet Vijay, Lochan Patil, Achutha Venkata Satyanarayana, Neeradi Chandra Sagar
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We resolve Source-2/Source-3 business records to deduplicated Source-1 entities with a
**retrieve → classify → decide** pipeline designed around three properties of the data and metric:
(i) noise is *script- and country-agnostic* (typos, transliteration, reordering, abbreviation, junk tokens),
(ii) every S2/S3 record belongs to **at most one** S1 entity, and (iii) macro F0.5 rewards precision and
correct singletons. Key contributions: (1) a hybrid **sparse + fine-tuned multilingual bi-encoder** blocker
that raises full-density recall from 88 % to 99.2 %; (2) a **full-density training protocol**, so validation
matches the test distribution; (3) a **cross-encoder re-ranker** for the few uncertain pairs, combined with
the feature model in a **stage-2 stacker**; (4) **exclusive assignment** plus a **Bayes plug-in expected-F0.5
rule** per entity; (5) **entity-disjoint training buckets** for every learned component (no label leakage);
(6) a **band specialist** that re-decides the ~3 % of genuinely uncertain pairs from the full feature vector
plus the logits of **six cross-encoders** trained on disjoint entity buckets; (7) **test-like competition
context**: validation samples S1 entities, which silently hid 7/8 of the rival S1s competing for each record,
so we score the missing rivals and train the specialist on the competition it actually faces at test time
(+0.00094 on the public leaderboard), followed by a **second pass that measures each rival's strength with
the cross-encoders**, leakage-controlled per entity bucket; and (8) a **country-agnostic, open-set design**
validated by leave-one-country-out experiments as a proxy for the unseen France split. Out-of-fold macro F0.5
on 276k held-out entities under test-like competition: **0.9910** (US + India); public leaderboard score of the
final submission (v22): **0.990367**. The candidate set the matching models score is **5.67 pairs per S1** (9.8M pairs for 1.73M S1). All models are MIT-licensed; the largest has 560M parameters (limit: 8B, MIT/Apache).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the 2.2M-entity training set (7.6M labelled links):
- **Structure.** Each S2/S3 record matches at most one S1 entity (max multiplicity = 1), so the labels
  partition S2 ∪ S3. 26.6 % of S2/S3 records match nothing (distractors). 5.6 % of S1 entities are
  singletons. The mean is 3.46 matches per entity (1.67 S2 + 1.79 S3), identical for US and India.
- **Name noise.** Character typos (`Midsteam`, `Webtemr`), injected diacritics (`Préferred`), prefix/suffix
  insertion (`Sri`, `Services`, `[Center]`, `née`), legal-form swaps (`Pvt Ltd` ↔ `Private Limited`,
  `Inc` ↔ `Incorporated`), `Last, First` inversion, domain-style names (`kananisai.com`),
  **full replacement by an unrelated coined word** (`Brixarc`), and **native-script names**
  (9.5 % of S2, 5.3 % of S3 are Devanagari/Tamil/Kannada/Malayalam; S1 is 100 % Latin).
- **Address noise.** Component reordering, upper-casing, `##`/`#` prefixes, abbreviation ↔ expansion,
  state code ↔ full name ↔ native-script state name, `NULL`/`N/A` tokens, added `PO Box`/`PMB`, missing
  components, and **truncated or shifted house numbers** (`030` vs `2030`, `1915-1919` vs `1915`).
- **Ambiguity.** 30 % of S1 names are duplicated (chains/common names), so the address must disambiguate.
- **Scale.** Test: 1.73M S1 × 9.97M S2/S3, including an unseen country (France, 15 % of S1).

### 2.2 Solution Strategy
**Approach Type:** Hybrid (dense + sparse) blocking → gradient-boosted pair classifier → cross-encoder
re-ranking of uncertain pairs → stage-2 stacker → constrained, metric-aware decision.
**Core Innovation:** see §1.

```
 S1, S2, S3 records ──► normalise (transliterate, canonicalise legal forms / abbreviations, skeleton keys)
        │
        ├─► sparse TF-IDF top-5 (both directions) ─┐
        └─► fine-tuned bi-encoder top-5 (both dir.) ┴─► hybrid blocking set (≈53 / S1, 99.2 % recall)
                                                            │
                     69 pair features (string, number, dense, competition) ─► LightGBM  → p₁
                                                            │
                     learned filter p₁ > 0.001 ─► output/candidate_pairs.tsv (5.67 / S1, 99.15 % recall)
                                                            │
                        0.005 < p₁ < 0.995 (3.3 %) ─► 6 cross-encoder logits (disjoint entity buckets)
                                                            │
                  stage-2 LightGBM (p₁, cross-encoder, cluster context)  → p₂  (outside the band)
                  second-pass band specialist LightGBM (all features + 6 logits + test-like competition
                    context + rival strength q + slot filling) → p  (inside the band)
                                                            │
             exclusive assignment (record → best S1)  →  per-S1 expected-F0.5 selection
                                                            │
             countries absent from training only: 4-logit models (one cross-encoder adapted with
             pseudo-labels), then demotion of same-address category-word swaps  →  TSVs
```

**Leakage control.** S1 entities are split by hash(id) mod 8: bucket 0 trains/validates the pair classifier,
stage 2 and the band specialist (4-fold, entity-grouped, out-of-fold), bucket 1 trains the bi-encoder,
buckets 2–3 the e5-base and normalised-text cross-encoders, buckets 4–5 the continued e5-base and the
hard-pair cross-encoder, and buckets 6–7 mDeBERTa-v3 and e5-large. No learned component is ever evaluated
on entities it was trained on.
**Compute.** One 24-thread PC with 10 GB RAM and an 8 GB RTX 3050, plus free Kaggle T4 GPUs for training the
larger cross-encoders. Every stage streams from parquet or disk memmaps and processes one country (and S1
chunk) at a time. Full test inference takes about 3.5 h (stage 1) plus cross-encoder scoring of the band.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** normalised name character 3-grams, consonant-skeleton name tokens, address-word skeletons
  and house numbers (sparse TF-IDF), plus a fine-tuned multilingual bi-encoder embedding of name + address (dense).
  Retrieval runs within each country label (an open set: no country is hard-coded), then a learned filter (stage-1
  LightGBM, p₁ > 0.001) gives the final candidate set.
- **Normalisation (all stages).** `anyascii` transliteration of every script to Latin, NFKC, lower-case,
  junk and punctuation removal, legal-form canonicalisation (US/India/French forms), name/address
  abbreviation expansion, US/India state canonicalisation, and a **consonant-skeleton phonetic key**
  (`private`/`praivet` → `prvt`, `foods`/`phuds` → `fts`) that aligns transliterated and English tokens.
- **Sparse retriever.** One TF-IDF space mixing name character 3-grams, name skeleton tokens,
  address-word skeletons and house numbers; exact sparse top-k (`sparse_dot_topn`) in both directions
  (record → S1 and S1 → record), per country. Posting lists are capped by absolute document frequency,
  which keeps cost roughly linear in N.
- **Dense retriever.** `intfloat/multilingual-e5-small` (MIT, 118M), fine-tuned with
  MultipleNegativesRanking loss on (record, true S1, **mined hard negative S1**) triplets. The entities used
  are disjoint from the classifier's. Exact GPU inner-product top-k, with the corpus streamed from disk
  memmaps in both directions.
- **Candidate pairs generated (final, `output/candidate_pairs.tsv`):** the hybrid blocking set (sparse top-5 ∪ bi-encoder
  top-5, both directions; 52.8 pairs per S1, 91.5M on test) is pruned by the stage-1 LightGBM, a *learned blocking
  filter* that keeps pairs with p₁ > 0.001. Everything downstream (cross-encoders, second-pass specialists, decision)
  scores exactly this set: **5.67 candidates per S1** (median 5, p95 10, p99 15, max 212; 9.8M pairs), a reduction ratio
  of 0.9999985 against all same-country pairs. On validation this filter removes 0.01 points of recall (99.158 % →
  99.146 %) and changes no decision. Every submitted match is a candidate (the official validator passes).
- **How true matches were not lost:** union of two complementary retrievers in both directions; recall
  was measured at **full density** (see §5), because a 10 % slice overstated recall (98.7 % vs 87.8 %).

---

## 4. Matching Model
**Features (69 in the stage-1 model; deliberately no country feature):**
- *Name:* rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on clean, core (legal and
  generic words removed) and sorted names; space-free ratios (domain names); skeleton ratio and Jaccard;
  containment; legal-form agreement; S1 name frequency (chains).
- *Address:* ratio / token-set / partial on the normalised address; word and skeleton Jaccard and
  containment; house-number Jaccard, exact first-number match, **partial-number agreement**
  (prefix/suffix), and a conflicting-number count.
- *Retrieval / competition:* sparse and dense scores and ranks in both directions, the record's gap to its
  best S1, the S1's gap to its best record, candidate counts, and bi-encoder cosine.
- *Other:* source (S2/S3), non-Latin flag, empty-address flags, name-inside-address, an acronym feature
  (`Twin Union SARL` → `TU`), location-stripped name similarities (name words that also occur in either record's
  address removed, e.g. `Nantes Centre SARL` at a Nantes address), and name lengths.

**Model type:** stage 1 = LightGBM binary classifier (MIT, 255 leaves). Top gain features: dense rank,
dense score/gap to the record's best S1, cosine gap, house-number conflict/partial agreement, name
token-sort ratio, address token-set ratio.
**Cross-encoders (final: six, all MIT):** each reads a (record, S1) pair jointly through a 1-logit head, fine-tuned
with BCE on candidate pairs of its own disjoint entity buckets: `intfloat/multilingual-e5-base` on raw text
(buckets 2–3, then continued on 4–5), `microsoft/mdeberta-v3-base` (6–7), e5-base on normalised text (2–3), a
hard-pair e5-base trained only on the uncertain band (4–5), and `intfloat/multilingual-e5-large` (560M, 6–7). They
resolve token-level cues (one-letter name edits, adjacent house numbers, coined names) that aggregated similarity
features blur, and run only on the ~3.3 % of candidates with 0.005 < p₁ < 0.995. (The first version, v4, used a
single `multilingual-e5-small`; see §5.) For a country absent from training, the continued e5-base is adapted with
pseudo-labels (§5, v16).
**Band specialist and second pass:** LightGBMs that re-decide the uncertain band from all stage-1 features, the
cross-encoder logits and record-side competition context, including each rival S1's cross-encoder-measured strength
and per-source slot filling (§5, v12–v14).
**Stage 2:** LightGBM on p₁, cross-encoder logit, rank of the pair within its S1 and within its record, the
record's margin over its runner-up S1, S1 match mass, and sibling agreement. Trained on stage-1 out-of-fold
probabilities with the same entity folds.
**Training protocol:** the full training split is blocked exactly as at test time, and features/labels are
computed for a deterministic 12.5 % hash sample of S1 entities. Ranks, gaps and name frequencies therefore
have the test distribution. 4-fold CV grouped by S1 entity.
**Threshold selection method:** out-of-fold macro F0.5 with the official formula (singletons included):
(1) **exclusive assignment**: each record keeps only its most probable S1; (2) either a global threshold
or the **plug-in expected-F0.5 rule**: for each S1, pick the top-m candidates maximising
(1+β²)·Σᵢ≤ₘpᵢ / (β²·Σpᵢ + m), versus the empty set with expected score ∏(1−pᵢ).

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** **0.99102** out-of-fold on 276k held-out US + India S1 entities (official formula,
  singletons included, test-like competition); public leaderboard **0.990367** for the final submission (v22).

| version | blocking | train density | OOF F0.5 (all / US / India) | public LB |
|---|---|---|---|---|
| v1 | sparse k5 | 10 % slice | 0.985* / – / – | – |
| v2 | sparse k5 | full | 0.927 / 0.940 / 0.908 | – |
| v3 | sparse k5 ∪ bi-encoder k5 | full | 0.978 / 0.979 / 0.977 | – |
| v4 | v3 + cross-encoder (e5-small) on uncertain pairs + stage-2 | full | 0.985 | 0.9810 |
| v5 | v4 + e5-base cross-encoder + French-robust normalisation + acronym feature | full | 0.9865 | 0.9820 |
| v6 | v5 + location-stripped name features | full | 0.9868 | 0.98253 |
| v7 | v6 + French region↔department canonicalisation + band specialist over 3 cross-encoders | full | 0.9885 | 0.98588 |
| v8 | v7 + 4th cross-encoder reading normalised text (core name, canonical address) | full | 0.98864 | 0.98594 |
| v9 | v8 + native-script Indian state names (16 forms) + mojibake repair | full | 0.98861 | 0.98600 |
| v11 | v9 + hard-pair cross-encoder + e5-large cross-encoder (6 cross-encoders) | full | **0.98902** | 0.98604 |
| v11 + open-set rule | v11 for seen countries; unseen country (France) uses the 4-cross-encoder specialist | full | 0.98902 | 0.98649 |
| v12 | specialists retrained on **test-like competition** (rival S1s of every band record scored) | full, rivals | 0.99052 (v11 under the same context: 0.98977) | **0.98743** |
| v13 | v12 + second pass: rival strength measured with the cross-encoders | full, rivals | 0.99089 | – |
| v14 | v13 + per-source slot-filling features | full, rivals | 0.99102 | – |
| v14 | US/India: v13 + slot filling | full, rivals | 0.99102 | 0.98782 |
| v16 | v14 + French-adapted `ce_base2` for the unseen country (pseudo-labels; US/India unchanged) | full, rivals | 0.99102 (US/India) | 0.98900 |
| v17c | v16 + all raw-text CEs French-adapted, 5-CE France model | full, rivals | 0.99102 (US/India) | 0.98759 |
| v18 | v16 + demotion of same-address category-word swaps on the unseen country | full, rivals | 0.99102 (US/India) | **0.98992** |
| **v22 (final)** | v18 rule extended (swaps into native French words, in-place `cie`/`france`, rarer words, swaps with extra noise) | full, rivals | 0.99102 (US/India) | **0.990367** |

\*optimistic: the 10 % slice has 10× fewer competing S1 entities. –: no public-leaderboard score recorded.

- **Blocking recall at full density** (true pairs of held-out sampled S1 entities retrieved; both
  directions unioned):

  | retriever | US | India |
  |---|---|---|
  | sparse TF-IDF, k=1 | – | 77.9 % |
  | sparse TF-IDF, k=5 (v1/v2) | ≈ 88 % | 85.2 % |
  | sparse TF-IDF, k=40 | – | 91.2 % |
  | **fine-tuned bi-encoder, k=1** | **97.4 %** | **97.5 %** |
  | bi-encoder k=3 ∪ sparse k=5 | 98.8 % | 98.9 % |
  | **bi-encoder k=5 ∪ sparse k=5 (v3)** | **99.1 %** | **99.2 %** |

  At 5M records per country the sparse space must be pruned (≈ 6.6 features per record) and its ranking
  degrades. The contrastively fine-tuned dense space ranks the true S1 first for 97.5 % of pairs, and
  handles native-script names, typos and partial addresses jointly.
- **Where the remaining loss goes (v3 OOF, 276k held-out S1, F0.5 0.977).** Entities with only false
  negatives account for 48 % of the loss, entities with a false merge 24 %, singletons given a spurious
  match 18 %, and true entities predicted empty 10 %. Of 44k missed links, 36k were retrieved but scored
  low and only 8k were never retrieved, so after v3 the bottleneck moved from blocking to scoring.
- **Calibration.** Out-of-fold probabilities of exclusive winners are well calibrated (0.5–0.7 bin:
  58 % true; 0.9–0.95: 92.5 %; >0.99: 99.9 %), which is what the expected-F0.5 rule needs.
- **Common false positives (wrong merges).** (i) *Planted look-alike distractors*: the data contains
  near-copies of an entity that belong to no S1 record, e.g. the same name at an adjacent house number
  (`Mccullough Corporation Clinic, 6513 Meadow Rd` vs the true `6506`), a coined name at a singleton's
  exact address (`Dovawextavo Co`), or a one-typo copy at the same address (`Solell Pharmaceuticals`).
  (ii) Partial-address copies of Indian companies with a generic name (`Shivam Constructions Pvt Ltd,
  6th Floor, Mumbai`). Both are close to label-level ambiguity.
- **Common false negatives (missed links).** (i) *Address-less records with a lightly corrupted name*
  (`Tech Agro Private Ltd`, `Gharonda Stock`, `gais.com`). The model is cautious because 30 % of S1 names
  are shared by chains; this motivated the name-ambiguity features (how many S1 entities share the
  record's core name / skeleton). (ii) Coined-name records with a truncated address (`Sri Zetaavi,
  #40/27, Chennai`). (iii) Shifted house numbers on true matches (`44` ↔ `14`, `108` ↔ `106`), which the
  generator also applies to distractors.
- **Unseen-country simulation (leave-one-country-out, full density, OOF F0.5 on the held-out country):**

  | train → test | in-country | held-out | Δ |
  |---|---|---|---|
  | US → India | 0.908 | 0.833 | −7.5 |
  | India → US | 0.940 | 0.923 | −1.7 |

  Transfer is asymmetric: training on the noisier, multi-script India data generalises well to the US,
  while the reverse loses 7.5 points. We therefore train on both countries (maximum noise diversity) for
  the unseen France split. Under shift the expected-F0.5 rule beats the best fixed threshold
  (0.8325 vs 0.8277), so it is used as the final decision rule.
- **Stage-2 cluster-aware re-scoring.** LightGBM on stage-1 OOF probability + collective context (rank
  within S1 and within the record, margin over the record's runner-up S1, S1 match mass, sibling
  name/address agreement), trained with the same entity folds. *Without* a cross-encoder (on v2) it gives only
  0.9309 → 0.9318; gain importance is p1 69 %, record margin 30 %, sibling features ≈ 0. Pairwise evidence
  plus exclusivity already captures most cluster structure. Stage 2 pays off once it receives an
  independent signal (the cross-encoder, below).
- **Cross-encoder re-ranking (v4).** A second MIT model (multilingual-e5-small with a 1-logit head) reads
  both records jointly. It is trained on a third disjoint entity bucket (hash 2) using the bi-encoder's
  top-3 candidates (800k pairs, 24 % positive: true matches vs planted look-alikes, adjacent numbers and
  chain namesakes), and applied only to pairs with 0.005 < p₁ < 0.995 (3.7 % of pairs). Stage-2 LightGBM
  on (p₁, cross-encoder logit, cluster context) raises OOF F0.5 from **0.9781 to 0.9849**. Standalone AUCs
  on the uncertain band are nearly identical (0.965 vs 0.964); the gain comes from complementary errors
  between feature-based and token-level reasoning. *Leakage check:* 93 % of evaluated pairs involve
  records the cross-encoder never saw, and on the 7 % it did see its AUC is lower (0.956 vs 0.965),
  so the improvement is not memorisation.
- **France (no labels).** On test, France showed 2.2× more stage-2 mass in the uncertain band
  (0.40 vs 0.18 winners per S1 with 0.2 < p < 0.95), while the expected number of matches per S1 equalled
  the US (3.41), so the model was not biased, only less certain. Inspection of uncertain French pairs
  found (i) `N°` numbering transliterated by anyascii to a junk token `ndeg`, (ii) French equivalents of the
  generator's filler words (`& Fils`, `Groupe`, `Développement`, `Établissements`) missing from the generic
  list, and (iii) acronym renames (`Twin Union SARL` → `TU`, also seen in India). All three were fixed
  with knowledge-based rules and a language-independent acronym feature. No test data was used for
  training. The fixes are neutral on US/India validation, as intended.
- **Diagnosing the test gap without labels.** Public LB (0.9825) sat 0.004 below OOF (0.9868). (1) Scaling all
  test odds by 0.5 lowered the LB slightly (0.98242), so the model is not over-confident on test. (2) Test US and
  India reproduce OOF statistics almost exactly (predicted matches/S1 3.358/3.356 vs 3.363; confident winners/S1
  3.22/3.20 vs 3.22), so the gap is France (≈ 0.958 implied), which had fewer confident matches (3.09/S1).
  (3) **Noise mining on confidently matched French clusters** (token diffs between S1 and its records, compared
  with US) found the cause: in France the generator **swaps the region for a department**
  (`Hauts-de-France` ↔ `Nord`/`Pas-de-Calais`, `Nouvelle-Aquitaine` ↔ `Gironde`, `Pays de la Loire` ↔
  `Loire-Atlantique`) in ~25 % of pairs, the French analogue of US state abbreviation ↔ name, which we
  canonicalise for US/India. It also abbreviates French street types (`R.`, `All.`, `Ch.`, `Q.`, `Crs`) and uses
  `St` for *Saint* (the US generator likewise writes `Street` → `Saint`). v7 canonicalises all 13 metropolitan
  regions and their departments to one region token (static domain knowledge, like the US state table)
  and unifies street/saint/st.
- **Band specialist.** Oracle analysis: if every pair with 0.005 < p₁ < 0.995 (3.3 % of pairs) were decided
  correctly, OOF F0.5 would reach 0.9972 (vs 0.9868), while perfect decisions on *all* candidates reach only
  0.9977. The remaining error is concentrated in this band. A LightGBM trained **only on band pairs** with
  the full stage-1 feature vector + cross-encoder logit + cluster context lifts band AUC 0.9852 → 0.9867
  and OOF F0.5 0.98678 → 0.98766.
- **Name-ambiguity features** (how many S1 entities share a record's core name / skeleton): OOF
  0.9777 → 0.9781, within noise; kept, but the dense and competition features already carry this signal.
- **Self-training on the unseen country (negative result).** Source = US, target = India, 500 fixed
  rounds, pseudo-labels p ≥ 0.97 (exclusive winners) / p ≤ 0.03, no target labels used for training,
  stopping or thresholds. Pseudo-labels were 99.4 % accurate and covered 96–98 % of target pairs, yet
  target F0.5 moved only 0.8388 → 0.8400 → 0.8407 over two rounds. Confident pseudo-labels are exactly
  the pairs the model already gets right, while the transfer gap lives in uncertain pairs. We therefore
  invest in country-agnostic representations (normalisation, the multilingual bi-encoder) rather than
  self-training for France.

- **Cross-encoder ensemble (v7–v11).** The band specialist reads the logits of six MIT cross-encoders, each
  trained on its own entity buckets, so every logit is out-of-sample for the band specialist's bucket 0:
  (1) `multilingual-e5-base` on raw text (buckets 2–3, 1.2M pairs); (2) the same model continued on fresh
  entities (4–5); (3) `mdeberta-v3-base` (6–7); (4) e5-base on the *normalised* view (core name | canonical
  address; 2–3), which sees the canonicalised region/state tokens; (5) a **hard-pair** e5-base trained only on
  the uncertain band of buckets 4–5 (pairs that survive stage 1, i.e. the specialist's own distribution); and
  (6) `multilingual-e5-large` (560M, 6–7), trained on Kaggle T4s. Band-specialist OOF F0.5: 3 CEs 0.98850 →
  4 CEs 0.98864 → +hard 0.98899 → +large **0.98902**. Gain importance: hard-pair CE 43 %, continued e5-base 18 %,
  e5-large 16 %, record margin 7 %.
- **Open-set rule for unseen countries.** The country set is open, so nothing is keyed on country names; the
  pipeline only knows which countries appeared in training. For a country absent from training, the band
  specialist omits the two cross-encoders fitted most tightly to the training countries' hard examples and
  uses the 4-cross-encoder specialist. On the public leaderboard this rule is worth +0.00045 (0.98604 → 0.98649),
  all of it from France.
- **Robustness experiments on the band specialist (leave-one-country-out as the unseen-country proxy).**

  | variant | in-country OOF | US → India | India → US |
  |---|---|---|---|
  | LightGBM (used) | 0.98860 | 0.98928 | 0.98722 |
  | + monotone constraints on similarity features | 0.98781 | 0.98879 | 0.98649 |
  | 5-seed bagging | 0.98860 | 0.98930 | 0.98718 |

  Monotone constraints cost 0.0008 in-country and also transfer worse, so the interactions they forbid are
  real signal. Bagging and specialist ensembling are flat (+0.00007).
- **Test-like competition context (v12, the largest late gain).** Error analysis of the v11 out-of-fold
  decisions: of the 19k true links still missed, 16k were uncertain pairs (median p = 0.40) cut by the
  precision-weighted rule, and 74 % of the missed records have no address. Of 116k uncertain *false* pairs, **84 %
  are records that truly belong to another S1 entity**, typically a namesake (`Tri-State Project` in Oregon vs
  Virginia; a name-only record fits both), and in **87 % of those cases that rival S1 was outside the 12.5 %
  validation sample**. Validation hashes S1 entities into buckets, so a record's record-side context
  (`rank_in_b`, `b_max`, `b_second`, `b_margin`, the specialist's 4th most important feature) was computed from
  ~1/8 of the S1s that compete for it at test time, where every S1 is present:

  | band pairs where… | training (sampled S1s) | test US / India | test France |
  |---|---|---|---|
  | a better-scored rival S1 exists for the record | 12 % | 21 % | **44 %** |
  | runner-up rival has p > 0.05 | 4–5 % | 15–16 % | 30 % |
  | runner-up rival has p > 0.5 | 0.2 % | 1.2 % | 4.3 % |

  The specialist had learned that a visible rival almost always means a false match, which was true only
  because rivals were rarely visible. France, with twice the competition, suffered most. Fix
  (`build_competitors`): for every record in the bucket-0 band, score all its candidate S1s outside bucket 0
  (3.1M rival pairs from 1.16M rival entities) with the final stage-1 model, which never trained on them,
  exactly as at test time. Then recompute the context with the test-time p > 0.001 filter and retrain
  (`exp_band_fd`). The a-side context is unchanged (a sampled S1 keeps all its records):

  | out-of-fold macro F0.5 | all | US | India | US → India (unseen-country proxy) |
  |---|---|---|---|---|
  | v11 specialist, sampled context (old protocol) | 0.98902 | 0.98825 | 0.99017 | – |
  | v11 specialist facing test-like context (= what it does at test time) | 0.98977 | 0.98901 | 0.99090 | 0.99084 |
  | **retrained on test-like context (v12)** | **0.99052** | **0.98972** | **0.99173** | **0.99160** |
  | 4-cross-encoder specialist (unseen countries): before → after | 0.98932 → 0.99018 | | | 0.99027 → 0.99103 |

  The retrained model adds 3.8k validation links at 90.5 % precision and drops 2.7k that were only 51.7 %
  correct. On test, France's predicted matches per S1 rose from 3.31 to 3.36, matching US/India (3.39) and
  the training ground truth (3.36). Only 28 % of the French links it adds coincide with a blanket "liberal
  France" probe that had *lowered* the leaderboard, so this is not a calibration shift. Public leaderboard:
  0.98649 → **0.98743**. Stage 2 has the same bias but decides only confident pairs; at test time it
  overturns 12 / 40 / 337 of ~5M confident matches (US / India / France), so it was left unchanged.
  *Lesson:* sampling the entity side for validation keeps each sampled entity's own neighbourhood intact but
  thins the **other** side's competition. Any feature that aggregates over the unsampled side must be
  recomputed at test density.
- **Second pass: rival strength from the cross-encoders (v13).** v12 measures how strongly a rival S1 claims a
  record by the rival's stage-1 probability (band AUC 0.965). A light *rival scorer* q = LightGBM(p₁,
  record-side context, cross-encoder logits) reaches band AUC **0.991** without any S1-side context (which
  does not exist for rivals). It scores every band pair of a record: bucket-0 pairs out-of-fold, and 562k rival
  band pairs with **only the cross-encoders that never saw the rival's entity bucket** (every cross-encoder
  has seen 2–4 buckets, counting initialisation: bucket 1 → all six; 2–3 → mDeBERTa, e5-large; 4–5 →
  e5-base, mDeBERTa, normalised, e5-large; 6–7 → e5-base, base2, normalised, hard). The record-side context is
  recomputed from q (margin to the best rival, rival mass, rank; p₁ stands in outside the band) and added to
  the specialist:

  | out-of-fold macro F0.5 | all | US | India | band AUC |
  |---|---|---|---|---|
  | v12 features (control, reproduces v12) | 0.99052 | 0.98972 | 0.99173 | 0.9917 |
  | **+ rival-strength context (v13)** | **0.99089** | **0.99011** | **0.99207** | **0.9922** |
  | 4-cross-encoder variant (unseen countries) | 0.99018 → 0.99051 | | | |

  The margin to the strongest rival becomes the top feature (59 % of gain). At test time all cross-encoders
  score every pair, so rival strength is *sharper* than in training; the model can only under-react, not flip
  direction. On test, v13 changes 0.9 % of US/India entities and 4.4 % of French ones (France has twice the
  competition).
- **Slot filling (v14).** Almost every matched entity has at least one S2 and one S3 record (only 7–8 % have none
  from a given source). For an address-less record shared by namesakes, a candidate that already holds confident
  records from the record's own source is a less likely owner than one holding none. Three features (the S1's
  other confident records from the record's source / from the other source, and its probability mass from the
  record's source) give OOF 0.99087 → **0.99102**, in both countries (US +0.00013, India +0.00018; band AUC
  0.9921 → 0.9924).
- **Adapting to the unseen country without labels (v16).** Three findings located France's problem.
  - *Structure does not explain it.* Reweighting validation F by each country's namesake count, address co-location
    and same-street namesakes predicts France ≈ 0.990, like India, while the leaderboard implies ≈ 0.970-0.975.
  - *The raw-text cross-encoders are broken on French text.* Stage-1-certain links are ~100 % true in US/India, so they
    can serve as pseudo-positives. On France, the raw-text cross-encoders reject 7-20x more of them than on the US
    (e.g. `Centre Hospitalier de la Marcel` → `Centre de la Marcel Groupe`: generator noise, CE logit −10). The
    normalised-text cross-encoder is the exception (0.56 % vs 0.53 %). The two cross-encoders the leaderboard had
    already shown to hurt France were the two worst by this criterion, which validates the criterion.
  - *The France model depends on the broken input.* 62 % of the France model's rival scorer runs through `ce_base2`.

  `er.adapt_unseen_ce` fine-tunes that cross-encoder on French pseudo-labels plus labelled US/India pairs:
  - positives: stage-1-certain French links;
  - negatives: the planted look-alike pattern (name + qualifier word at a different house number) and clear rejects.

  On held-out pseudo-labels, French false rejections fall from 2.04 % to 0.17 % (US/India level) and French false
  accepts fall from 4.53 % to 1.01 %; US/India are unchanged. On test it changes 8.9 % of French entities. It accepts
  more same-address noisy copies (name + filler: 0.91 → 0.97, the US truth rate) and rejects more same-name records at
  other house numbers (the test's planted look-alikes). French expected links per S1 stay at 3.45. *Label-free
  diagnostics that ruled out other French explanations:*
  - The model's expected-F bias grows with namesake count (+0.001 for unique names to +0.0065 for 5+).
  - The test contains ~15x more multi-record planted look-alikes than training (S2+S3 copies of a fake neighbour with
    no S1). They emit no address-less records (ratio 1.03-1.10 vs 2.1 for real namesakes), and the model already
    rejects them.
  - Records that add a qualifier word at the *same* house number are 89 % true in the US; at a different number,
    3.9 %.
- **Leaderboard deltas as labels for the unseen country (v17c, v18).** Adapting *all* raw-text cross-encoders
  (v17c) lowered the score by 0.00141 even though every adapted model passed its pseudo-label check. Only France
  changed, so the diff v16 → v17c is labelled in aggregate. Its added links were dominated by one pattern: 8,291 French
  records that swap one native category word for another at the same address (`X Club` → `X Ecole`). That is the
  generator's planted-distractor recipe, and the pseudo-positives (stage-1-certain links) had contained some of them.
  The helpful step v14 → v16 had instead mostly added swaps *into* filler-like words (`X Club` → `X SARL Cie`, 2,724
  links). v18 keeps v16 and demotes its 6,684 French category-swap links (`er.demote_swaps`). Abbreviations and swaps
  into filler-like words are kept, and US/India are untouched. **Leaderboard: 0.988997 → 0.989917 (+0.00092, i.e.
  +0.006 on France, ≈0.24 F per removed link: nearly every demoted link was false).** v22 extends the rule, which the
  filler score justifies: words that are as frequent in S1 names as in records (`compagnie`, `centre`, `societe`: score
  0.8-0.9) are native name words, not noise, so swapping into them is still a category swap. `cie`/`france` are kept
  when appended after the legal form (`X SARL Cie`, the generator's noise position) and demoted when they replace the
  category word in place.
- **What did not help on the last day (all measured).** (i) *Cross-source evidence propagation* (using S2↔S3
  similarity to resolve address-less namesakes). The generator makes each record from its S1 independently: an
  S3 record equals its S2 sibling's raw name 3.0 % of the time vs its S1's 6.2 % (mean similarity 59.6 vs 81.2).
  Picking an address-less record's owner by its best match to each namesake's *other records* is right 30 % of
  the time, vs 37 % using the S1 name itself and 21 % by chance. So siblings carry less evidence than the S1
  record already used, and propagation cannot beat direct matching. (ii) *Label-free France audits.*
  - Same-address S1s missing from the candidates have different names in 99.4 % of cases, so namesakes are not
    crowding out true owners.
  - French house numbers are zero-padded less often than US/India ones.
  - Single-word swaps in French names are mostly the generator's filler noise (`X Club` → `X SARL Cie`).
  - Name-vs-address conflicts are calibrated in validation (US: 41.4 % true vs 41.4 % accepted).
  - The leaderboard's France ×2 probe matches the model's own predicted cost (−0.0021 vs −0.0016 if
    calibrated), so France is calibrated at the margin.

  France's remaining deficit (~0.015 below US/India) comes from confident errors that no label-free diagnostic
  located. US/India test link mixes and expected link counts match validation almost exactly, so their scores
  transfer.
- **Negative results after v7.** (i) *Stage-2 consensus features* (cosine of a record to the centroid of the
  S1's other confident records): flat. (ii) *3× more band training rows* from buckets 4–5: +0.00008.
  (iii) *Record-graph clustering* of S2/S3 records: distractor records are isolated look-alikes (a matched
  record's nearest record belongs to the same entity 86–91 % of the time vs ~20 % for distractors), a signal
  the sibling and competition features already carry. (iv) An *organisation-word rule* for French records
  (`Groupe`, `& Fils`) was refuted on US/India, where records adding such a word are true matches 91–95 % of
  the time. (v) *Liberal calibration for France* (odds × 2) lowered the leaderboard (−0.00032), so French
  errors at the margin are false merges, not misses.


### Compliance notes
- **Models:** LightGBM (MIT), `intfloat/multilingual-e5-small/base/large` (MIT), `microsoft/mdeberta-v3-base` (MIT); the
  largest has 560M parameters (limit 8B). The unseen-country cross-encoder is a fine-tuned copy of an MIT model.
- **No external data or lookups:** only the provided train/test files are used. The US/Indian state lists, the French
  region/department table and the abbreviation lists in `normalize.py` are static, hand-written domain knowledge
  (like a stop-word list), not an external database or API. The unseen-country adaptation uses pseudo-labels computed
  from the test *inputs* by our own pipeline, never labels.
- **Open-set countries:** no training or prediction path is keyed on a country name, and no model has a country feature.
  The code distinguishes only "seen in training" vs "not seen" (from the training records); the unseen branch
  (4-cross-encoder specialist, pseudo-label adaptation, category-swap demotion) applies to any such country. The training
  countries' names appear only in validation reports (per-country out-of-fold scores and the leave-one-country-out
  experiments in `exp_band_fd`, `exp_band_rival`, `exp_band_robust`), which do not affect the saved models.
- **Leaderboard use:** public-leaderboard feedback was used only to choose between whole-pipeline variants and to confirm
  one generator rule (same-address category-word swaps are planted distractors). The rule is applied identically to
  every unseen-country entity, and each extension moved the score by the same amount per changed link, which is
  consistent with removing genuine false links rather than fitting public-set noise.

---

## 6. Conclusion
Entity resolution at this scale is limited first by **candidate recall** and then by **fine-grained
disambiguation** of look-alikes. A fine-tuned multilingual bi-encoder removed the first limit (recall 88 % →
99.2 %; F0.5 0.927 → 0.978). A cross-encoder applied to the ~4 % of genuinely uncertain pairs, stacked with
the feature model, addressed the second (0.978 → 0.985). Three protocol choices mattered as much as the
models: validating at **full test density** (a 10 % slice overstated F0.5 by ~6 points), keeping
**every learned component on disjoint entities**, and making sure that features aggregating over the
*unsampled* side see the test's competition (v12: +0.00094 on the leaderboard from retraining alone). Negative results were informative too: self-training did
not close the unseen-country gap, and the gain from cluster features alone was negligible. Both point to
representation quality, not decision heuristics, as the lever that transfers to an unseen country such as
France.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/src/er/`:

| module | role |
|---|---|
| `normalize`, `prep` | normalisation, streaming TSV → parquet |
| `blocking` | sparse TF-IDF retriever |
| `biencoder` | bi-encoder training, memmap encoding, exact tiled GPU top-k search |
| `candidates` | hybrid candidate union + dense context features |
| `features`, `add_ambiguity` | pair features |
| `train_full` | full-density stage-1 training + OOF evaluation |
| `predict` | chunked per-country stage-1 inference |
| `cross_encoder`, `kaggle_train_ce`, `export_ce_pairs` | cross-encoder training (local GPU or Kaggle) / scoring |
| `exp_band`, `build_extra_band`, `export_band_pairs` | band specialist training (+ hard-pair data) |
| `redecide` | re-decision from cached probabilities (odds calibration probes) |
| `stage2`, `exp_stage2`, `predict_stage2` | cluster context, stage-2 training, final inference + TSV writing |
| `build_competitors`, `exp_band_fd` | test-like competition: score rival S1s outside the sample, retrain the specialists |
| `demote_swaps` | post-decision demotion of same-address category-word swaps (unseen country) |
| `adapt_unseen_ce` | pseudo-label adaptation of a cross-encoder to a country absent from training |
| `export_rival_band`, `exp_band_rival` | second pass: rival scorer on cross-encoder logits + rival-strength context + slot filling |
| `decide` | exclusivity, expected-F0.5 rule, official metric |

Entry points and exact commands are in `code/business_entity_resolution/README.md`
(prep → bi-encoder encode/search → `er.predict` (stage 1) → `er.predict_stage2` (first pass) → `er.adapt_unseen_ce` +
`kaggle_train_ce.py` (unseen countries) → `er.predict_stage2` (final pass, writes both TSVs) → `er.demote_swaps`
(unseen countries) → `output/matching_results.tsv`, `output/candidate_pairs.tsv`).

### B. Additional Results — ablation path (full-density OOF macro F0.5, 276k held-out S1)
| step | F0.5 | Δ |
|---|---|---|
| sparse blocking + LightGBM (v2) | 0.927 | – |
| + fine-tuned bi-encoder candidates & dense features (v3) | 0.978 | +5.1 |
| + name-ambiguity features | 0.978 | +0.0 |
| + cross-encoder (e5-small) on uncertain pairs + stage-2 stacker (v4) | 0.985 | +0.7 |
| cross-encoder e5-small → e5-base (278M), 1.2M training pairs | 0.9867 | +0.2 |
| + French-robust normalisation (`N°`, French filler words) + acronym feature (v5) | **0.9865** | ±0.0 on US/India (targets France) |

Decision rule (v3 OOF): fixed threshold 0.75 → 0.9777; expected-F0.5 → 0.9770–0.9776. The two are equal
in-distribution, but expected-F0.5 is more robust under country shift (§5), so it is the final rule.
