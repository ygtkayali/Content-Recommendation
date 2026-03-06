# Reranker Implementation Plan

> Covers the end-to-end path from raw data to deployed reranker.
> Each phase produces concrete artifacts that feed the next phase.

---

## 0  Current System Snapshot

| Component | Detail |
|-----------|--------|
| Item catalog | `data/processed/anime.parquet` — 9,185 rows × 16 columns (`anime_id`, `Name`, `Score`, `Genres`, `Synopsis`, `Type`, `Studios`, `Source`, `Scored By`, `Image URL`, …) |
| User interactions | `data/raw/anime_user_filtered/user-filtered.csv` — ~109 M rows × 3 columns (`user_id`, `anime_id`, `rating` 0-10). Rating = 0 means "watched but not rated" on MAL. ~1.5 GB on disk. |
| Embeddings | `synopsis_embeddings.npy` — 9,185 × 768 float32 (all-mpnet-base-v2, L2-normalized) |
| Feature arrays | `feature_arrays.npz` — bayesian scores, ratings, votes |
| Config | `config.json` — feature weights, bayesian params, retrieval config |
| Backend | FastAPI, loads artifacts at startup, candidate generation via dot-product + argpartition + MMR |
| Docker image | ~395 MB, deployed on Render free tier (512 MB limit) |

### Key constraint: no user auth at serving time
The reranker will be a **global (non-personalized) reranker**: trained on population interaction data to learn "what items tend to be good recommendations" given item and query-item features — but at inference time it does **not** require the user dataset or per-user features.

---

## Phase 1 — Data Preparation

### 1.1  Clean the interaction dataset

The raw file is ~109 M rows and ~1.5 GB. Most of this is not needed for a global reranker.

**Steps:**

1. **Load in chunks** (the file won't fit in memory on most dev machines if loaded naively).
   ```python
   chunks = pd.read_csv("data/raw/anime_user_filtered/user-filtered.csv", chunksize=5_000_000)
   ```

2. **Drop unrated rows** — `rating == 0` means "watched, no score." These are ambiguous; drop them for the first version.
   - Expected reduction: ~42% of rows are rating=0 (based on 500k sample).

3. **Filter to items that exist** in our anime catalog (9,185 items in `data/processed/anime.parquet`).
   - This ensures every `anime_id` in the interaction set maps to an item we have embeddings for.

4. **Subsample users** — for a global reranker we don't need all ~300k+ users. A stratified sample of 20k–50k active users (sorted by interaction count) is sufficient and will keep training manageable.

5. **Save** as `data/processed/interactions_clean.parquet`.

**Output artifact:**

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int64 | User identifier |
| `anime_id` | int64 | Item identifier (matches anime catalog) |
| `rating` | int8 | 1-10 explicit score |

Expected size: ~5-15 M rows, ~50-150 MB parquet.

---

### 1.2  Build item feature table

Precompute item-level statistics from the *full* interaction dataset (before user subsampling) to get robust aggregates.

**Computed columns:**

| Feature | Source | Description |
|---------|--------|-------------|
| `avg_user_rating` | interactions | Mean rating across all users who rated this item |
| `rating_count` | interactions | Number of ratings (not counting 0s) |
| `rating_std` | interactions | Rating standard deviation |
| `positive_ratio` | interactions | Fraction of ratings ≥ 7 |
| `catalog_score` | anime.parquet | MAL score from catalog |
| `catalog_scored_by` | anime.parquet | MAL vote count |
| `bayesian_score_norm` | feature_arrays.npz | Already computed, reuse |
| `popularity_rank` | anime.parquet | Popularity column |
| `type_encoded` | anime.parquet | One-hot or label-encoded Type |
| `source_encoded` | anime.parquet | One-hot or label-encoded Source |
| `genre_vector` | anime.parquet | Multi-hot genre encoding |
| `n_genres` | anime.parquet | Number of genres |

**Save** as `data/processed/item_features.parquet` (~9,185 rows × ~30 columns).

This table is small (~1 MB) and will be deployed alongside the model.

---

### 1.3  Build user profile aggregates (training only)

These features are used during *training* to provide query context but are **not needed at inference time** (since there's no user auth).

| Feature | Description |
|---------|-------------|
| `user_avg_rating` | Mean rating given by this user |
| `user_rating_count` | Total items rated |
| `user_rating_std` | Rating variance |
| `user_genre_vector` | Average genre vector across rated items |
| `user_type_distribution` | Distribution of rated item types |
| `user_avg_item_popularity` | Mean popularity of items the user rated |

**Save** as `data/processed/user_features.parquet`.

Only used offline — **does not ship in Docker**.

---

## Phase 2 — Training Dataset Construction

### 2.1  Label construction

Convert ratings to binary labels:

| Rating | Label | Rationale |
|--------|-------|-----------|
| ≥ 7 | 1 (positive) | User explicitly liked the item |
| 1–6 | 0 (negative) | User rated but didn't like |
| 0 | dropped | Ambiguous (watched but not scored) |

### 2.2  Candidate generation for training pairs

For each user's **positive** interaction, generate a candidate set:

1. **True positive:** the item the user rated ≥ 7 (label = 1).
2. **Hard negatives (3-5 per positive):** items from the user's *low-rated* items (rating 1-6). These are items the user actually watched but disliked — strong negatives.
3. **Embedding negatives (2-3 per positive):** items that are embedding-similar to the positive item but the user did not interact with. These teach the model to distinguish "similar but not good" from "similar and good."
4. **Popularity negatives (1-2 per positive):** randomly sampled from the top-500 most popular items that the user didn't rate. Prevents pure popularity bias.

**Total negative ratio:** ~6-10 negatives per positive.

### 2.3  Feature vector construction

For each (user, candidate_item) pair, compute:

**Query-item features** (these are available at inference time):
- `embedding_similarity` — dot product between query item and candidate item embeddings
- `genre_overlap` — Jaccard similarity between query and candidate genres
- `type_match` — binary, same type
- `studio_match` — binary, same studio  
- `source_match` — binary, same source
- `candidate_avg_rating` — from item features table
- `candidate_rating_count` — from item features table
- `candidate_positive_ratio` — from item features table
- `candidate_bayesian_norm` — from item features table
- `candidate_popularity_rank` — from item features table
- `rating_count_log` — log(1 + rating_count)
- `score_diff` — abs(query_score - candidate_score)

**User-item features** (training only, dropped at inference):
- `user_avg_rating`
- `user_genre_overlap_with_candidate`
- `mean_similarity_to_user_history` — average embedding similarity between candidate and user's top-rated items
- `max_similarity_to_user_history`

### 2.4  Query groups

Group training samples by `user_id` for LambdaRank.

```
group_sizes = train_df.groupby("user_id").size().values
# e.g., [87, 92, 64, 110, ...]
```

### 2.5  Train/val/test split

Split **by user** (not by row) to prevent leakage:
- Train: 70% of users
- Validation: 15% of users
- Test: 15% of users

**Save:**
- `data/processed/reranker_train.parquet`
- `data/processed/reranker_val.parquet`
- `data/processed/reranker_test.parquet`

---

## Phase 3 — Candidate Generation Pipeline

### How the current system fits in

The existing content-based pipeline **becomes Stage 1** with zero changes:

```
Query item (anime_id)
│
├── 1. Embedding retrieval (dot product → argpartition → top 100)
│       Already implemented in recommend_by_index()
│
├── 2. Multi-feature scoring (synopsis + genre + type + studio + source + bayesian)
│       Already implemented with FEATURE_WEIGHTS
│
└── 3. MMR diversity pass
        Already implemented in apply_mmr()
```

**What changes:** Instead of returning the MMR output directly, we now pass the **pre-MMR candidate pool** (top 100-200 by multi-feature score) into the reranker as Stage 2.

### Updated pipeline flow

```
User clicks on item X
│
▼
Stage 1: Candidate Generation (existing code)
  embeddings @ query_vec → argpartition top-200
  → multi-feature scoring → top-100 candidates
│
▼
Stage 2: Feature Construction
  For each candidate: compute reranker features
  (embedding_sim, genre_overlap, item stats, etc.)
│
▼
Stage 3: LightGBM Reranker
  scores = model.predict(candidate_features)
  → sort by predicted score → top-20
│
▼
Stage 4: Post-processing
  Optional: MMR diversity pass on reranked top-20
  → Return final top-10
```

### What stays the same
- Embedding computation (dot product)
- Genre/type/studio/source matching
- Artifact loading at startup

### What's new
- Item feature table loaded at startup (~1 MB)
- LightGBM model loaded at startup (~5-20 MB)
- Feature vector construction for candidates (cheap: numpy ops on precomputed arrays)
- `model.predict()` call (~1-5 ms for 100 candidates)

---

## Phase 4 — Model Training & Evaluation

### 4.1  Model: LightGBM LambdaRank

```python
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train,           # feature matrix
    label=y_train,     # binary labels
    group=group_train,  # query group sizes
)

val_data = lgb.Dataset(
    X_val,
    label=y_val,
    group=group_val,
    reference=train_data,
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)
```

### 4.2  Evaluation metrics

| Metric | Target | Description |
|--------|--------|-------------|
| NDCG@10 | > 0.70 | Primary ranking quality metric |
| NDCG@20 | > 0.65 | Broader ranking quality |
| MAP@10 | > 0.50 | Precision-oriented |
| HitRate@10 | > 0.80 | At least one relevant item in top 10 |

### 4.3  Baseline comparison

Evaluate the **current system** (multi-feature + MMR) on the same test set using the same metrics. The reranker should beat this baseline by a meaningful margin (≥5% relative NDCG improvement) to justify the added complexity.

### 4.4  Feature importance analysis

After training, inspect `model.feature_importance()` to:
- Identify which features the model relies on
- Drop low-importance features to simplify the inference feature set
- Validate that the model isn't just learning popularity

### 4.5  Hyperparameter tuning

Tune with Optuna or manual grid search:
- `num_leaves`: [15, 31, 63]
- `learning_rate`: [0.01, 0.05, 0.1]
- `min_data_in_leaf`: [10, 20, 50]
- `num_boost_round`: early stopping handles this
- Negative sampling ratio: [4, 6, 10]

### 4.6  Save model

```python
model.save_model("ml/artifacts/reranker_v1/model.txt")
```

LightGBM text format is portable and typically 5-15 MB.

---

## Phase 5 — Deployment Update

### 5.1  New artifacts

```
ml/artifacts/reranker_v1/
├── model.txt              # LightGBM model (~5-15 MB)
├── item_features.csv      # Precomputed item stats (~1 MB)
└── config.json            # Feature names, version, thresholds
```

### 5.2  Dependency budget

Current Docker image: ~395 MB.

| Addition | Estimated size | Notes |
|----------|---------------|-------|
| `lightgbm` (pip, no sklearn) | ~5-8 MB | LightGBM can run standalone without sklearn |
| Model artifact | ~5-15 MB | Text format |
| Item features CSV | ~1 MB | Precomputed |
| **Total addition** | **~11-24 MB** | |
| **Projected image** | **~406-419 MB** | **Well under 512 MB** |

**Critical: do NOT add scikit-learn to the backend Docker image.**
- Train with sklearn offline (in notebooks/scripts).
- Deploy with `lightgbm` only — it has no sklearn/scipy dependency for `model.predict()`.
- This saves ~100+ MB (sklearn + scipy).

### 5.3  Backend changes

Update `backend/app/main.py`:

1. **Load reranker at startup** (in `_load_artifacts`):
   ```python
   import lightgbm as lgb
   reranker_model = lgb.Booster(model_file=str(reranker_dir / "model.txt"))
   item_features = pd.read_csv(reranker_dir / "item_features.csv")
   ```

2. **Add reranker to recommend() endpoint:**
   ```python
   # After candidate generation (existing code), before return:
   if store.reranker_model is not None:
       features = build_reranker_features(query_item, candidates, store.item_features)
       scores = store.reranker_model.predict(features)
       candidates["reranker_score"] = scores
       candidates = candidates.sort_values("reranker_score", ascending=False)
   ```

3. **Feature construction function** — operates on numpy arrays already in memory, no external data needed:
   ```python
   def build_reranker_features(query_idx, candidate_indices, item_features, embeddings):
       # embedding_similarity: already computed in candidate gen
       # genre_overlap: already computed
       # item stats: lookup from item_features array
       # All vectorized numpy operations, <1ms for 100 candidates
       ...
   ```

4. **Fallback** — if reranker model is not present, the endpoint works exactly as before (content-based + MMR only). This makes the reranker an opt-in upgrade.

### 5.4  Dockerfile changes

```dockerfile
# In requirements.txt, add:
lightgbm>=4.0.0

# In Dockerfile, add COPY for reranker artifacts:
COPY ml/artifacts/reranker_v1/ /app/ml/artifacts/reranker_v1/
```

### 5.5  Render config update

Add env var:
```yaml
- key: RERANKER_DIR
  value: ml/artifacts/reranker_v1
```

---

## Implementation Order & Notebook Plan

| Step | Notebook / Script | Output |
|------|-------------------|--------|
| 1 | `ml/experiments/reranker_01_data_prep.ipynb` | `interactions_clean.parquet`, `item_features.parquet`, `user_features.parquet` |
| 2 | `ml/experiments/reranker_02_training_dataset.ipynb` | `reranker_train.parquet`, `reranker_val.parquet`, `reranker_test.parquet` |
| 3 | `ml/experiments/reranker_03_model_training.ipynb` | Trained model, evaluation metrics, feature importance |
| 4 | `ml/experiments/reranker_04_evaluation.ipynb` | Baseline vs reranker comparison, ablation studies |
| 5 | Backend integration PR | Updated `main.py`, new artifacts in Docker |

Each notebook is self-contained and produces artifacts consumed by the next.

---

## Risk Checklist

| Risk | Mitigation |
|------|------------|
| 109M-row file doesn't fit in memory | Process in chunks, subsample users |
| Reranker doesn't beat baseline | Feature importance analysis → iterate features; if <5% gain, don't deploy |
| Docker image exceeds 512 MB | Use `lightgbm` only (no sklearn); keep item_features as CSV (~1 MB) |
| Cold-start latency increases | Reranker predict adds ~1-5 ms for 100 candidates — negligible |
| Model overfits popularity | Include embedding negatives; monitor popularity bias in feature importance |
| Rating=0 ambiguity | Drop rating=0 rows entirely in v1; revisit as implicit signal in v2 |
