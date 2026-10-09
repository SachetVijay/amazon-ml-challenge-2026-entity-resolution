# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team AMS
**Team Members:** [List all team members]
**Submission Date:** [Date]

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
and (6) a **country-agnostic design validated by leave-one-country-out** experiments as a proxy for the
unseen France split. Out-of-fold macro F0.5 on 276k held-out entities at full density: **0.985**. All
models are MIT-licensed; the largest has 118M parameters.

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
        └─► fine-tuned bi-encoder top-5 (both dir.) ┴─► candidate pairs (≈53 / S1, 99.2 % recall)
                                                            │
                     ~70 pair features (string, number, dense, competition) ─► LightGBM  → p₁
                                                            │
                        0.005 < p₁ < 0.995 (3.7 %) ─► cross-encoder logit
                                                            │
                  stage-2 LightGBM (p₁, cross-encoder, cluster context)  → p₂
                                                            │
             exclusive assignment (record → best S1)  →  per-S1 expected-F0.5 selection  →  TSVs
```

**Leakage control.** S1 entities are split by hash(id) mod 8: bucket 0 trains/validates the pair classifier
and stage 2 (4-fold, entity-grouped), bucket 1 trains the bi-encoder, and buckets 2–3 train the
cross-encoder. No learned component is ever evaluated on entities it was trained on.
**Compute.** One 24-thread PC with 10 GB RAM and an 8 GB RTX 3050. Every stage streams from parquet or
disk memmaps and processes one country (and S1 chunk) at a time. The full test inference takes about 3.5 h.

---

## 3. Candidate Generation (Blocking)
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
- **Candidate pairs generated:** v3 ≈ 44 per S1 on training (12.1M pairs for 276k sampled S1); v1/v2: 29 per S1 (50.3M on test)
- **How true matches were not lost:** union of two complementary retrievers in both directions; recall
  was measured at **full density** (see §5), because a 10 % slice overstated recall (98.7 % vs 87.8 %).

---

## 4. Matching Model
**Features (~60, deliberately no country feature):**
- *Name:* rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on clean, core (legal and
  generic words removed) and sorted names; space-free ratios (domain names); skeleton ratio and Jaccard;
  containment; legal-form agreement; S1 name frequency (chains).
- *Address:* ratio / token-set / partial on the normalised address; word and skeleton Jaccard and
  containment; house-number Jaccard, exact first-number match, **partial-number agreement**
  (prefix/suffix), and a conflicting-number count.
- *Retrieval / competition:* sparse and dense scores and ranks in both directions, the record's gap to its
  best S1, the S1's gap to its best record, candidate counts, and bi-encoder cosine.
- *Other:* source (S2/S3), non-Latin flag, empty-address flags, name-inside-address.

**Model type:** stage 1 = LightGBM binary classifier (MIT, 255 leaves). Top gain features: dense rank,
dense score/gap to the record's best S1, cosine gap, house-number conflict/partial agreement, name
token-sort ratio, address token-set ratio.
**Cross-encoder:** `intfloat/multilingual-e5-small` (MIT, 118M) with a 1-logit head, fine-tuned with BCE on
800k (record, S1) pairs from the bi-encoder's top-3 (24 % positive). It reads both raw records jointly, so it
resolves token-level cues (one-letter name edits, adjacent house numbers, coined names) that aggregated
similarity features blur. It runs only on the 3.7 % of pairs with 0.005 < p₁ < 0.995.
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
| version | blocking | train density | OOF F0.5 (all / US / India) | public LB |
|---|---|---|---|---|
| v1 | sparse k5 | 10 % slice | 0.985* / – / – | [TBD] |
| v2 | sparse k5 | full | 0.927 / 0.940 / 0.908 | [TBD] |
| v3 | sparse k5 ∪ bi-encoder k5 | full | 0.978 / 0.979 / 0.977 | [TBD] |
| v4 | v3 + cross-encoder (e5-small) on uncertain pairs + stage-2 | full | 0.985 | 0.9810 |
| v5 | v4 + e5-base cross-encoder + French-robust normalisation + acronym feature | full | 0.9865 | 0.9820 |
| v6 | v5 + location-stripped name features | full | 0.9868 | 0.98253 |
| v7 | v6 + French region↔department canonicalisation + band specialist over 3 cross-encoders | full | **0.9885** | **0.98588** |

\*optimistic: the 10 % slice has 10× fewer competing S1 entities.

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

---

## 6. Conclusion
Entity resolution at this scale is limited first by **candidate recall** and then by **fine-grained
disambiguation** of look-alikes. A fine-tuned multilingual bi-encoder removed the first limit (recall 88 % →
99.2 %; F0.5 0.927 → 0.978). A cross-encoder applied to the ~4 % of genuinely uncertain pairs, stacked with
the feature model, addressed the second (0.978 → 0.985). Two protocol choices mattered as much as the
models: validating at **full test density** (a 10 % slice overstated F0.5 by ~6 points) and keeping
**every learned component on disjoint entities**. Negative results were informative too: self-training did
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
| `cross_encoder` | cross-encoder training / scoring |
| `stage2`, `exp_stage2`, `predict_stage2` | cluster context, stage-2 training, final inference + TSV writing |
| `decide` | exclusivity, expected-F0.5 rule, official metric |

Entry points and exact commands are in `code/business_entity_resolution/README.md`
(prep → bi-encoder encode/search → `er.predict` → `er.predict_stage2` → `output/*.tsv`).

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
