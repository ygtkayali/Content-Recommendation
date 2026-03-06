# Reranker Pipeline Audit — Findings, Issues & Improvement Recommendations

> Full review of: `anime_user_data.ipynb`, `reranker_training.ipynb`, `recommendation_reranker_v1.py`, and supporting artifacts.

---

## 1  Data Preparation (`anime_user_data.ipynb`)

### 1.1  What's done correctly

| Aspect | Status | Notes |
|--------|--------|-------|
| Chunked reading of 109M rows | ✅ | Proper `chunksize=5M`, avoids OOM |
| Drop rating==0 | ✅ | Removes ambiguous "watched, unrated" rows |
| Catalog filter | ✅ | Only keeps anime_ids present in the 9,185-item catalog |
| Stratified user sampling | ✅ | Uses `qcut` into 10 activity bins for representative selection |
| Item feature aggregation from FULL dataset | ✅ | Uses all ~109M rows (before user subsample) for robust item stats |
| Bayesian score merge | ✅ | Correctly maps through `anime_index` from metadata |

### 1.2  Issues found

#### ISSUE D-1: Item features computed from all users including future test users (Minor)

Item-level aggregates (`avg_user_rating`, `rating_count`, `positive_ratio`) are computed from **all** users in the raw dataset — including users who will later appear in the validation/test split. For a global reranker where these are population-level statistics, this is acceptable and common practice. But it means the model may slightly overfit to item quality signals computed from "future" data. Since these features are static aggregates and not per-user, the practical impact is negligible.

**Verdict:** Acceptable for v1. Flag for v2 if temporal splitting is added.

#### ISSUE D-2: Redundant rating==0 filter in reranker_training.ipynb (Cosmetic)

`interactions_clean.parquet` already has rating==0 rows removed (done in `anime_user_data.ipynb`). The training notebook filters again:
```python
interactions = interactions[interactions["rating"] != 0].copy()
```
This is harmless but adds confusion about where the canonical filter lives.

---

## 2  Training Pair Construction (`reranker_training.ipynb`, cells 1–7)

### 2.1  What's done correctly

| Aspect | Status |
|--------|--------|
| Binary labels (≥7 → 1, 1-6 → 0) | ✅ |
| Self-candidate exclusion (`aid != query_anime_id`) | ✅ |
| `used_negatives` set prevents duplicate candidates per query | ✅ |
| Three negative sources (hard, embedding, popularity) | ✅ |
| Embedding negatives exclude user-seen items | ✅ |
| Deduplication safety net on output | ✅ |
| Memory guards (MAX_TOTAL_PAIRS, MAX_POSITIVES_PER_USER) | ✅ |

### 2.2  Issues found

#### ISSUE T-1: User selection bias toward power users (High Impact)

```python
MAX_USERS_FOR_TRAINING = 20000
user_activity = interactions["user_id"].value_counts()
selected_users = set(user_activity.head(MAX_USERS_FOR_TRAINING).index.tolist())
```

This selects the **top 20k most active** users from the already-subsampled 50k–100k set. The result is a training set dominated by power users who have rated hundreds of items. This introduces two problems:

1. **Representation bias:** The model learns "what heavy raters like" rather than general population preferences. Heavy raters tend to have broader taste, watch more niche titles, and rate differently than casual users.
2. **Negative pool bias:** Power users have more hard negatives (many low-rated items), making their negative distribution different from casual users who have fewer interactions.

**Recommendation:** Use stratified sampling instead of top-N. Sample users from across activity tiers (e.g., 10 quantile bins) with equal or proportional representation. The existing `anime_user_data.ipynb` already implements this pattern — reuse it here.

#### ISSUE T-2: Positive candidates are random liked items — content-misaligned (Moderate)

For each query anime A (rated ≥7 by the user), the positive candidate is a **random other liked item** B from the same user:

```python
pos_pool = [aid for aid in user_pos if aid != query_anime_id]
positive_candidate = int(RNG.choice(np.array(pos_pool, dtype=np.int64), size=1)[0])
```

This means: "user liked Death Note and K-On! → K-On! is a positive candidate for Death Note." At inference time, the reranker is used after content-based retrieval where all candidates are already embedding-similar to the query. Training with content-dissimilar positives creates a distribution mismatch between training and inference.

**Recommendation:** Either:
- (a) Weight positive candidates by embedding similarity to the query (prefer content-similar positives), or
- (b) Add a "positive sampling temperature" that biases toward more similar items while still allowing diversity.

#### ISSUE T-3: Embedding negatives may be false negatives (Moderate)

```python
def embedding_negatives_for_positive(pos_anime_id, user_seen, n_take):
    # Returns items embedding-similar to pos but NOT in user history
```

These are the most embedding-similar items that the user simply **hasn't seen yet**. Many of these could be genuinely good recommendations — the user just hasn't discovered them. Labeling them as negative teaches the model to penalize unseen-but-similar items, which directly contradicts the purpose of a recommendation system.

**Recommendation:** Either:
- (a) Reduce the weight of embedding negatives (currently 2-3 per positive; reduce to 1), or
- (b) Use graded relevance labels instead of binary (e.g., embedding_negative=0.3 instead of 0), or
- (c) Draw embedding negatives from a broader similarity band (rank 50-200 instead of top-most-similar) to avoid penalizing the best candidates.

#### ISSUE T-4: Small LambdaRank query groups (~7-11 candidates) (Moderate)

Each (user, query) group contains:
- 1 positive
- 3-5 hard negatives
- 2-3 embedding negatives
- 1-2 popularity negatives
= **~7-11 total candidates**

LambdaRank learns from pairwise swaps within groups. With only 1 positive per group and ~7-11 total items, the ranking signal per group is limited. The model mostly learns a binary classification boundary rather than fine-grained ranking.

**Recommendation:** Increase negatives to ~15-25 per positive (e.g., hard 5-8, embedding 4-6, popularity 3-5). This would also better match the inference scenario where Stage 1 produces 100-200 candidates.

#### ISSUE T-5: Train/serve candidate distribution mismatch (High Impact)

At inference, **all** candidates come from Stage 1 and are already embedding-similar to the query (top-100 by content similarity). But during training:
- Hard negatives are random low-rated items (any genre/type — often very dissimilar to query)
- Popularity negatives are random popular items (also potentially very dissimilar)
- Only 2-3 out of ~8 negatives are embedding-similar

The model learns to distinguish "embedding-similar" from "not similar at all" rather than learning **fine-grained differentiation within** already-similar candidates. At inference, this coarse signal provides less value since all candidates are already similar.

**Recommendation:** Add "Stage 1 negatives": run the actual candidate generation pipeline for each query item during training data construction and include Stage 1 candidates that the user rated low, or randomly sample from the Stage 1 output for items the user hasn't seen. This aligns training distribution with inference distribution.

---

## 3  Feature Construction (`reranker_training.ipynb`, cells 8-10)

### 3.1  What's done correctly

| Aspect | Status |
|--------|--------|
| Embedding similarity as dot product | ✅ |
| Genre overlap via Jaccard | ✅ |
| Self-similarity masking in `mean/max_similarity_to_user_history` | ✅ |
| Chunked processing with part files | ✅ |
| Consistent lowercase/strip normalization | ✅ |

### 3.2  Issues found

#### ISSUE F-1: Train/serve skew on user features (Critical)

The model is trained with 4 user-dependent features:
- `user_avg_rating`
- `user_genre_overlap_with_candidate`
- `mean_similarity_to_user_history`
- `max_similarity_to_user_history`

At inference in `recommendation_reranker_v1.py`, these are **hardcoded to 0.0**:
```python
"user_avg_rating": 0.0,
"user_genre_overlap_with_candidate": 0.0,
"mean_similarity_to_user_history": 0.0,
"max_similarity_to_user_history": 0.0,
```

Examining the trained model's feature importance:
```
feature                              importance_gain   splits
embedding_similarity                 102,319           5
candidate_avg_rating                  50,403           5
max_similarity_to_user_history        19,997           4  ← ALWAYS 0 AT INFERENCE
rating_count_log                      15,927           8
candidate_positive_ratio              12,413           7
candidate_popularity_rank              4,611           1
[10 other features]                        0           0
```

**`max_similarity_to_user_history` is the 3rd most important feature by gain, with 4 splits — and it's ALWAYS ZERO at inference time.** This means ~10% of the model's learned signal is completely absent at serving time, causing unpredictable prediction shifts. The model's internal decision boundaries that depend on this feature will always take the "≤ threshold" branch when the value is 0, regardless of what the true value should be.

**Recommendation (immediate):** Retrain the model with user features **excluded** from the feature set. Since the model is a global (non-personalized) reranker, user features should not be part of it. This will force the model to rely entirely on query-item and candidate features that are available at inference.

**Alternative:** If user features demonstrably improve quality and you plan to add user auth later, train two model variants — one with and one without user features — and deploy the user-free variant.

#### ISSUE F-2: 10 out of 16 features have zero importance (High Impact)

The trained model only uses **6 out of 16 features**:

| Feature | Importance | Used? |
|---------|-----------|-------|
| embedding_similarity | 102,319 | ✅ |
| candidate_avg_rating | 50,403 | ✅ |
| max_similarity_to_user_history | 19,997 | ✅ (but broken at inference) |
| rating_count_log | 15,927 | ✅ |
| candidate_positive_ratio | 12,413 | ✅ |
| candidate_popularity_rank | 4,611 | ✅ |
| genre_overlap | 0 | ❌ |
| type_match | 0 | ❌ |
| studio_match | 0 | ❌ |
| source_match | 0 | ❌ |
| candidate_bayesian_norm | 0 | ❌ |
| candidate_rating_count | 0 | ❌ |
| score_diff | 0 | ❌ |
| user_avg_rating | 0 | ❌ |
| user_genre_overlap_with_candidate | 0 | ❌ |
| mean_similarity_to_user_history | 0 | ❌ |

Observations:
- **`genre_overlap`, `type_match`, `studio_match`, `source_match` all have zero importance.** These are the content-matching features that Stage 1 already uses. By the time candidates reach the reranker, the Stage 1 scoring has already heavily filtered on these features, so they have low variance across candidates. The model can't learn from features with no discriminative power in the training population.
- **`candidate_rating_count` has zero importance** but `rating_count_log` has high importance — the log transform helped.
- **`candidate_bayesian_norm` has zero importance** — likely correlated with `candidate_avg_rating`, which the model prefers.
- The model is essentially: `f(embedding_sim, avg_rating, history_sim, log_count, positive_ratio, pop_rank)` — a simple quality-adjusted similarity scorer.

**The very shallow tree (only 30 total splits across all features)** suggests the model stopped very early during training, likely because the validation metric plateaued quickly with limited useful signal.

#### ISSUE F-3: Studio/source matching is exact string comparison (Minor)

Multi-studio shows produce strings like `"madhouse, bones"`. Exact equality fails to capture partial overlaps. This contributes to why these features have zero importance — most comparisons evaluate to 0.

**Recommendation:** Parse studios/sources into sets and compute Jaccard similarity (like genres), or check if any studio in the candidate's set matches any in the query's set.

---

## 4  Train/Val/Test Split & Model Training

### 4.1  What's done correctly

| Aspect | Status |
|--------|--------|
| Split by user (no leakage across splits) | ✅ |
| 70/15/15 ratio | ✅ |
| LambdaRank objective | ✅ |
| Query groups defined by (user_id, query_anime_id) | ✅ |
| Sorted data matches group ordering | ✅ |
| Early stopping on validation NDCG | ✅ |
| Sanity checks for self-candidates | ✅ |

### 4.2  Issues found

#### ISSUE M-1: No baseline comparison available (Moderate)

The notebook checks for a `baseline_score` column in the test split but doesn't find one, so baseline comparison is skipped. Without this, there's no way to validate that the reranker actually improves over the existing Stage 1 scoring. The implementation plan requires ≥5% relative NDCG@10 improvement.

**Recommendation:** Add a baseline column to the training data by computing the Stage 1 multi-feature score for each (query, candidate) pair. This is straightforward: `baseline_score = FEATURE_WEIGHTS["synopsis"] * emb_sim + FEATURE_WEIGHTS["genre"] * genre_overlap + ...`. Then evaluate with the existing comparison code.

#### ISSUE M-2: The model is extremely shallow (Moderate)

Only ~30 total splits across all trees suggests early stopping kicked in very quickly (likely around boost round 5-15 out of 500). This indicates:
- The training signal is weak or the task is too easy (binary classification with obvious negatives)
- Most negatives are trivially distinguishable from positives (content-dissimilar hard negatives vs. content-similar positives = easy separation by `embedding_similarity` alone)
- The model degenerates into a simple quality-adjusted embedding similarity scorer

This reinforces ISSUE T-5: the training negatives are too easy. Making them harder (more Stage-1-like candidates) would force the model to learn more nuanced patterns.

---

## 5  Inference Pipeline (`recommendation_reranker_v1.py`)

### 5.1  What's done correctly

| Aspect | Status |
|--------|--------|
| Full 4-stage pipeline (retrieve → score → rerank → MMR) | ✅ |
| Graceful fallback (returns Stage 1 results if reranker missing) | ✅ |
| Feature columns loaded from JSON (no hardcoded list) | ✅ |
| Missing item features default to 0.0 | ✅ |
| MMR is optional via `--disable-mmr` | ✅ |
| Consistent genre parsing with training | ✅ |
| Proper path resolution for artifacts | ✅ |

### 5.2  Issues found

#### ISSUE I-1: User features hardcoded to 0.0 (Critical — same as F-1)

Already covered in F-1. The 4 user features are set to zero, but `max_similarity_to_user_history` is the model's 3rd most important feature. This degrades inference quality.

#### ISSUE I-2: Feature computation uses row-by-row iteration (Performance)

```python
for _, row in candidates_df.iterrows():
    ...
    rows.append(feature_row)
```

For 100–200 candidates this iterates row-by-row with Python dict construction. While functionally correct and fast enough for batch sizes <500, it could be vectorized for consistency with the training code (which uses vectorized pandas/numpy throughout).

**Recommendation:** Vectorize using pandas operations (similar to the training notebook's approach). Not urgent — the current approach adds ~5ms for 120 candidates, which is acceptable.

#### ISSUE I-3: `embedding_similarity` reuses Stage 1 `synopsis_score` (Correct but note)

```python
"embedding_similarity": float(row["synopsis_score"]),
```

This reuses the dot-product similarity already computed in Stage 1 rather than recomputing it. This is correct and efficient. However, the naming difference (`synopsis_score` in Stage 1 vs `embedding_similarity` in features) could cause confusion.

#### ISSUE I-4: `score_diff` uses catalog `Score` column, not `avg_user_rating` (Consistency)

```python
q_score = pd.to_numeric(pd.Series([work_df.iloc[q_idx][score_col]]), errors="coerce")...
"score_diff": float(abs(float(q_score) - float(cand_score))),
```

This computes score difference using the **catalog Score** (MAL community score). The training notebook also uses catalog Score:
```python
score_map = catalog["Score"].to_dict()
```

This is consistent. However, `score_diff` has zero importance in the trained model, so it's a no-op currently.

---

## 6  Summary of Issues by Severity

### Critical (fix before next deployment)

| ID | Issue | Component |
|----|-------|-----------|
| F-1 / I-1 | **Train/serve skew on user features** — `max_similarity_to_user_history` is 3rd most important feature but always 0 at inference | Training + Inference |

### High Impact (address in next training iteration)

| ID | Issue | Component |
|----|-------|-----------|
| T-1 | **User selection bias** — top-20k most active users, not representative | Pair construction |
| T-5 | **Training/inference distribution mismatch** — negatives too easy, don't match Stage 1 output | Pair construction |
| F-2 | **10/16 features unused** — model is an extremely shallow quality-adjusted similarity scorer | Training |

### Moderate (plan for v1.1)

| ID | Issue | Component |
|----|-------|-----------|
| T-2 | Positive candidates content-misaligned with inference scenario | Pair construction |
| T-3 | Embedding negatives may be false negatives | Pair construction |
| T-4 | Small query groups (~8 candidates) limit LambdaRank learning | Pair construction |
| M-1 | No baseline comparison to validate reranker adds value | Evaluation |
| M-2 | Model stops training very early — signal is too weak or task too easy | Training |

### Minor

| ID | Issue | Component |
|----|-------|-----------|
| D-1 | Item features include test user ratings (acceptable for global reranker) | Data prep |
| D-2 | Redundant rating==0 filter | Data prep |
| F-3 | Studio/source exact match misses multi-value fields | Features |
| I-2 | Row-by-row feature construction at inference | Inference |

---

## 7  Improvement Recommendations (Within Current Pipeline)

These recommendations focus on feature and data quality improvements that don't require architectural changes.

### 7.1  Immediate: Retrain without user features

**Action:** Remove the 4 user features from the feature set.

```python
# In feature_columns.json, remove:
# "user_avg_rating", "user_genre_overlap_with_candidate",
# "mean_similarity_to_user_history", "max_similarity_to_user_history"
```

Zero the columns during training data construction or exclude them from `feature_cols`. This eliminates train/serve skew and forces the model to rely on available signals. Expected impact: the model will redistribute the `max_similarity_to_user_history` signal across other features (likely `embedding_similarity` and quality features absorb most of it).

### 7.2  Better user sampling for training

**Action:** Replace `user_activity.head(MAX_USERS_FOR_TRAINING)` with stratified sampling.

```python
# Instead of:
selected_users = set(user_activity.head(MAX_USERS_FOR_TRAINING).index.tolist())

# Use:
n_bins = 10
activity_bins = pd.qcut(user_activity, q=n_bins, labels=False, duplicates="drop")
users_per_bin = MAX_USERS_FOR_TRAINING // n_bins
selected = []
for bin_id in range(n_bins):
    bin_users = user_activity.index[activity_bins == bin_id]
    take = min(len(bin_users), users_per_bin)
    selected.extend(RNG.choice(bin_users.to_numpy(), size=take, replace=False).tolist())
selected_users = set(selected)
```

This ensures casual users (who are the majority of real traffic) are represented in training.

### 7.3  Harder negatives aligned with inference distribution

**Action:** For each query, run a lightweight version of Stage 1 candidate generation and sample negatives from those candidates.

```python
# For each query_anime_id, find top-200 by embedding similarity
sims = embeddings @ embeddings[anime_to_index[query_anime_id]]
top200 = np.argsort(-sims)[1:201]  # exclude self
top200_ids = [index_to_anime[i] for i in top200 if i in index_to_anime]

# Sample negatives from top200 that the user rated low or didn't interact with
stage1_hard = [aid for aid in top200_ids if aid in user_neg_set]
stage1_unseen = [aid for aid in top200_ids if aid not in user_seen]
```

This creates negatives that the reranker will actually encounter at inference time, forcing it to learn fine-grained quality differences within the similarity neighborhood.

### 7.4  Increase negatives per positive

**Action:** Increase sampling ranges:
```python
HARD_NEG_RANGE = (5, 8)    # was (3, 5)
EMB_NEG_RANGE = (4, 6)     # was (2, 3)
POP_NEG_RANGE = (3, 5)     # was (1, 2)
# + add STAGE1_NEG_RANGE = (5, 10)
```

Target ~20-30 candidates per query group. This provides richer pairwise signal for LambdaRank.

### 7.5  New/improved features

These are computable from existing data with no pipeline changes:

| Feature | Description | Why |
|---------|-------------|-----|
| `genre_overlap_count` | `len(q_genres & c_genres)` (raw count, not Jaccard) | Jaccard penalizes items with many genres; count captures absolute overlap |
| `studio_jaccard` | Parse multi-studio strings into sets, compute Jaccard | Fixes the exact-match limitation |
| `source_jaccard` | Same for source | Same fix |
| `candidate_rating_cv` | `rating_std / avg_user_rating` (coefficient of variation) | Items with high CV are polarizing — useful signal |
| `popularity_log` | `log1p(popularity_rank)` | Similar to `rating_count_log` transform which the model already uses |
| `score_ratio` | `candidate_score / max(query_score, 1e-6)` | Relative quality instead of absolute difference |
| `embedding_sim_squared` | `embedding_similarity ** 2` | Non-linear transform — lets the model capture quadratic similarity effects without deep trees |
| `genre_count_diff` | `abs(n_genres_query - n_genres_candidate)` | Items with similar genre breadth may be better matches |

### 7.6  Graded relevance labels

**Action:** Instead of binary 0/1, use graded labels for richer ranking signal:

| Source | Label |
|--------|-------|
| Positive (user rated ≥ 7) | 2 |
| Hard negative (user rated 4-6) | 1 |
| Hard negative (user rated 1-3) | 0 |
| Embedding negative (unseen) | 0 |
| Popularity negative (unseen) | 0 |

Or even finer: use the actual normalized rating (1-10 → 0.0-1.0) as the relevance label. LightGBM LambdaRank supports continuous relevance labels.

### 7.7  Add baseline score column for proper evaluation

**Action:** During feature construction, compute and store the Stage 1 multi-feature score as `baseline_score`:

```python
baseline_score = (
    0.45 * embedding_similarity
    + 0.20 * genre_overlap
    + 0.10 * type_match
    + 0.05 * studio_match
    + 0.05 * source_match
    + 0.15 * candidate_bayesian_norm
)
```

This enables the existing baseline comparison code in the training notebook to measure actual reranker lift.

---

## 8  Recommended Implementation Order

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Remove user features, retrain | 30 min | Eliminates train/serve skew |
| 2 | Add baseline score column, measure lift | 1 hr | Validates reranker adds value |
| 3 | Stratified user sampling | 30 min | Better generalization |
| 4 | Stage-1-aligned negatives | 2-3 hrs | Much harder negatives, better model |
| 5 | Increase negatives per group | 15 min | Richer LambdaRank signal |
| 6 | New features (7.5) | 2 hrs | Better candidate differentiation |
| 7 | Graded labels | 30 min | Richer ranking signal |
| 8 | Studio/source Jaccard | 30 min | May help, low effort |
