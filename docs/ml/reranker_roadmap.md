# Reranker System Implementation Roadmap

## 1. Project Objective

The goal is to improve recommendation quality by introducing a **two-stage recommendation pipeline** consisting of:

1. **Candidate Generation**
2. **Learning-to-Rank Reranker**

The candidate generation stage retrieves a set of potentially relevant items using similarity-based methods.  
The reranker then orders these candidates using a **LightGBM ranking model trained on user interaction data**.

The system should remain **lightweight enough to deploy on limited infrastructure (e.g., Render or similar services)**.

---

# 2. System Architecture

The recommendation pipeline will follow a two-stage architecture:

User Request
│
▼
Candidate Generator
(Embedding similarity / heuristics)
│
Top-K candidates (~100–200)
│
▼
Feature Builder
(user features + item features + interaction features)
│
▼
LightGBM Ranker
(Learning-to-rank model)
│
▼
Top-N Recommendations


### Design Principles

- Keep inference latency low
- Avoid deep models in the reranking stage
- Use precomputed features whenever possible
- Maintain compatibility with low-cost deployment environments

---

# 3. Dataset Preparation

### Available Data

User interaction dataset:

| user_id | item_id | rating |
|--------|--------|--------|
| U1 | A10 | 8 |
| U2 | A42 | 7 |

This dataset represents **explicit user feedback**.

---

### Interaction Label Construction

Convert ratings into **binary preference labels**.

Example rule:

rating >= 7 → positive interaction (label = 1)
rating < 7 → ignore or treat as negative


---

### Training Sample Structure

Each training row represents a **(user, candidate item)** pair.

Example:

| user_id | item_id | label |
|--------|--------|------|
| U1 | A10 | 1 |
| U1 | A32 | 0 |
| U1 | A44 | 0 |

Positive items are items the user interacted with.  
Negative samples are items the user has **not interacted with**.

---

### Negative Sampling

For each positive interaction:

1 positive item
4-10 negative items.


Negative items can be sampled:

- randomly
- from popular items
- from embedding neighbors

---

# 4. Candidate Generation Stage

Candidate generation produces a **small subset of potentially relevant items**.

Typical number:
50-200 candidate items


Possible candidate sources:

---

### Embedding Similarity

Items similar to user history based on embeddings.

Example:cosine_similarity(user_embedding, item_embedding)


---

### Genre-Based Candidates

Items with overlapping genres.

Example heuristic: items that share >=1 genre with user history

    

---

### Popularity-Based Candidates

Top popular items.

Used for exploration and cold-start situations.

---

### Candidate Pool Construction

Merge candidates from multiple sources.

Example: 50 embedding candidates, 50 genre candidates, 50 popular items

Final candidate pool: 100-200 items



---

# 5. Feature Engineering

The reranker relies entirely on **handcrafted features**.

These features must be **cheap to compute at inference time**.

---

## 5.1 Candidate Score Features

Features generated during candidate retrieval.

Examples: embedding_similarity content_similarity


---

## 5.2 Item Features

Precompute these offline.

Examples:
average_rating
rating_count
popularity_score
release_year
genre_vector

---

## 5.3 User Features

Derived from user history.

Examples:


user_average_rating
user_rating_variance
favorite_genres
interaction_count


---

## 5.4 Interaction Features

These are often the **most important ranking signals**.

Examples:


genre_overlap(user, item)

mean_similarity_to_user_history

max_similarity_to_user_history

time_since_last_similar_item


---

# 6. Feature Table Structure

Example final training dataset:

| user | item | label | emb_sim | avg_rating | popularity | genre_match | hist_sim_mean |
|----|----|----|----|----|----|----|----|
| U1 | A12 | 1 | 0.81 | 8.7 | 0.72 | 0.66 | 0.74 |
| U1 | A43 | 0 | 0.32 | 7.1 | 0.40 | 0.12 | 0.18 |

---

# 7. Model Training

The reranker will use **LightGBM with a ranking objective**.

Recommended objective:


LambdaRank


Training procedure:

1. Group samples by user
2. Provide candidate items per user
3. Optimize ranking quality using NDCG

---

### Example Parameters


objective = lambdarank
metric = ndcg
ndcg_eval_at = 10
learning_rate = 0.05
num_leaves = 31
num_boost_round = 500


---

### Query Groups

Ranking models require group sizes.

Example:


User1 → 100 candidates
User2 → 100 candidates
User3 → 100 candidates


Group array:


[100, 100, 100, ...]


---

# 8. Evaluation Strategy

Model quality should be evaluated using **ranking metrics**.

Primary metric:


NDCG@10


Additional metrics:


MAP
Recall@K
HitRate@K


Evaluation procedure:

1. Split dataset into train/validation/test
2. Generate candidates for test users
3. Apply reranker
4. Compare ranked results with ground truth interactions

---

# 9. Deployment Pipeline

The deployed inference system will follow this workflow.

### Step 1 — Retrieve Candidates

From candidate generator:


top 100 candidate items


---

### Step 2 — Feature Construction

Compute ranking features for each candidate.

Example:


embedding similarity
genre overlap
popularity
user statistics


---

### Step 3 — Ranking

Use the trained LightGBM model.


scores = model.predict(candidate_features)


---

### Step 4 — Final Ranking

Sort candidates by predicted score.

Return:


Top 20 recommendations


---

# 10. Infrastructure Requirements

The reranker is lightweight.

Typical resource requirements:

| Component | Cost |
|---|---|
Model size | 5–20 MB |
Inference latency | <10 ms |
RAM usage | <200 MB |

This allows deployment on:

- small cloud instances
- low-tier hosting services
- containerized microservices

---

# 11. Implementation Timeline

### Phase 1 — Dataset Preparation

Tasks:

- clean interaction dataset
- convert ratings to binary labels
- implement negative sampling
- build ranking dataset

---

### Phase 2 — Feature Engineering

Tasks:

- compute item statistics
- compute user statistics
- implement interaction features

---

### Phase 3 — Model Training

Tasks:

- train LightGBM ranking model
- tune hyperparameters
- evaluate with ranking metrics

---

### Phase 4 — Integration

Tasks:

- integrate candidate generator
- build feature pipeline
- implement ranking inference

---

### Phase 5 — Deployment

Tasks:

- export LightGBM model
- integrate into backend API
- deploy with application

---

# 12. Expected Improvements

Compared to pure similarity-based recommendations, the reranker should:

- incorporate **user preference signals**
- improve **recommendation ordering**
- reduce irrelevant recommendations
- better model **interaction patterns**

---

## Final System Outcome
