# Core Machine Learning Paradigms: A Practical Learning Guide

Learn the 4 major learning types—Supervised, Unsupervised, Semi‑Supervised, and Reinforcement Learning—through approachable theory, clear problem formulation patterns, evaluation strategies, and hands‑on code templates.

---
## 1. Supervised Learning
You have labeled examples: each input X has a known target y. Goal: learn a mapping f: X → y that generalizes to unseen data.

### 1.1 Core Tasks
| Task | Output Type | Typical Algorithms | Common Metrics |
|------|-------------|--------------------|----------------|
| Classification | Discrete labels | Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Nets | Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Log Loss |
| Regression | Continuous values | Linear Regression, Random Forest Regressor, Gradient Boosting, SVR, Neural Nets | RMSE, MAE, R², MAPE, Pinball Loss (quantiles) |

### 1.2 When to Use
Use supervised learning when you have enough labeled data and a clear target variable that reflects success.

### 1.3 Problem Formulation Checklist
1. Define the business question (e.g., "Reduce churn?" → Predict churn probability; "Optimize pricing?" → Predict demand).
2. Identify target variable and confirm label quality (completeness, leakage risk, timing alignment).
3. Choose prediction granularity (per user, per session, per transaction?).
4. Establish baseline (heuristic or simple model).
5. Determine cost of errors (false positives vs false negatives) to choose metrics & threshold strategy.
6. Split data respecting time / groups to avoid leakage.

### 1.4 Key Evaluation Concepts
- Classification thresholds: Use ROC for ranking models; use Precision-Recall when classes are imbalanced.
- Calibrated probabilities: Reliability curves / Brier score when downstream decision logic uses probabilities.
- Regression error distribution: Plot residuals; check heteroscedasticity; consider prediction intervals.
- Robustness: Evaluate across user segments, time periods, drift windows.

### 1.5 Quick Code Templates
Classification (scikit-learn):
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:,1]
preds = (proba > 0.35).astype(int)  # threshold tuned to cost

print('ROC AUC:', roc_auc_score(y_test, proba))
print(classification_report(y_test, preds))
```

Regression uncertainty (quantile gradient boosting):
```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

quantiles = [0.1, 0.5, 0.9]
models = {q: GradientBoostingRegressor(loss='quantile', alpha=q, random_state=42) for q in quantiles}
for q, m in models.items():
    m.fit(X_train, y_train)

pred_interval = {q: m.predict(X_test) for q, m in models.items()}
median = pred_interval[0.5]
lower = pred_interval[0.1]
upper = pred_interval[0.9]
```

---
## 2. Unsupervised Learning
No labels; goal: discover structure/patterns. Often exploratory or for feature engineering.

### 2.1 Core Tasks
| Task | Purpose | Examples |
|------|---------|----------|
| Clustering | Group similar entities | K-Means, DBSCAN, Hierarchical, Gaussian Mixture |
| Dimensionality Reduction | Compress / visualize / denoise | PCA, t-SNE, UMAP, Autoencoders |
| Association Rules | Find co-occurrence patterns | Apriori, FP-Growth |
| Outlier / Anomaly Detection | Identify rare / abnormal cases | Isolation Forest, LOF, One-Class SVM, Autoencoder |

### 2.2 When to Use
Use when labels are missing or when you want to: segment customers, detect fraud, preprocess features, explore latent structure.

### 2.3 Problem Formulation Patterns
1. Segmentation: Define business dimension (value, behavior, lifecycle) → Feature matrix → Number of clusters (k) guided by elbow/silhouette + business interpretability.
2. Anomaly Detection: Define what counts as "rare" (frequency, density, deviation) → Choose algorithm aligned with assumption (e.g., Isolation Forest handles high-dimensional sparse).
3. Association: Transaction-like data → Set min support & confidence tied to actionability.

### 2.4 Evaluation Strategies
Intrinsic (model-based) vs extrinsic (business outcome). Without labels:
- Clustering internal indices: Silhouette, Davies-Bouldin, Calinski-Harabasz.
- Stability: Run with bootstrapped samples; measure adjusted Rand between solutions.
- Business lift: e.g., marketing response per cluster, churn variance across segments.
- Anomaly evaluation: Use small labeled validation subset or synthetic anomalies (inject perturbations) to estimate precision/recall.

### 2.5 Practical Code Snippets
Clustering + profiling:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

features = ['visits', 'avg_order_value', 'days_since_last', 'lifetime_value']
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)
best_k = max(scores, key=scores.get)

km = KMeans(n_clusters=best_k, n_init='auto', random_state=42)
clusters = km.fit_predict(X_scaled)
df['cluster'] = clusters
cluster_profile = df.groupby('cluster')[features].mean().assign(count=df.groupby('cluster').size())
```

Association rules (mlxtend library assumed installed):
```python
from mlxtend.frequent_patterns import apriori, association_rules

# basket_df: one-hot encoded transactions
freq = apriori(basket_df, min_support=0.02, use_colnames=True)
rules = association_rules(freq, metric='lift', min_threshold=1.1)
actionable = rules[(rules['confidence'] > 0.3) & (rules['lift'] > 1.2)]
```

Anomaly detection (Isolation Forest):
```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.02, random_state=42)
scores = iso.fit_predict(X_scaled)  # -1 anomaly, 1 normal
df['is_anomaly'] = (scores == -1)
```

---
## 3. Semi-Supervised Learning
Mix of a small labeled set L and a larger unlabeled set U. Goal: exploit structure in U to improve predictive performance.

### 3.1 Common Approaches
| Method | Idea | When Useful |
|--------|------|-------------|
| Self-Training | Train model on L → pseudo-label high-confidence U → retrain | Classifier confident on subset |
| Label Propagation / Spreading | Graph-based diffusion of labels via similarity | Data lies on manifold, smooth transitions |
| Consistency Regularization | Encourage stable predictions under perturbations | Images, text augmentations |
| Pseudo-Labeling (Deep) | Use model’s own high-prob outputs as labels | Large U, moderate L |
| MixMatch / FixMatch | Combine augmentations + entropy minimization | Modern deep SSL |

### 3.2 Evaluation Strategy
- Hold out a labeled validation set from L; never use for pseudo-labeling.
- Track: baseline supervised (only L) vs semi-supervised method.
- Monitor noise: pseudo-label precision (sample audit).
- Early stopping: stop when validation performance plateaus; avoid overfitting to erroneous pseudo-labels.

### 3.3 Simple Pseudo-Label Template (tabular)
```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

model = GradientBoostingClassifier(random_state=42)
model.fit(X_L, y_L)
proba_U = model.predict_proba(X_U)
confidence = proba_U.max(axis=1)
high_conf_idx = np.where(confidence > 0.9)[0]

X_pl = X_U[high_conf_idx]
y_pl = model.predict(X_pl)

X_aug = np.vstack([X_L, X_pl])
y_aug = np.concatenate([y_L, y_pl])

model.fit(X_aug, y_aug)
```

---
## 4. Reinforcement Learning (RL)
Agent learns to act in an environment by maximizing cumulative reward. No labeled pairs; learning driven by trial-and-error feedback.

### 4.1 Core Concepts
| Term | Meaning |
|------|---------|
| State (s) | Environment observation at time t |
| Action (a) | Decision taken by agent |
| Reward (r) | Scalar feedback for (s,a) |
| Policy (π) | Mapping from states to actions (deterministic or stochastic) |
| Value Function | Expected cumulative reward from a state or state-action |
| Episode | Sequence from start to terminal state |

### 4.2 When to Use
- Sequential decisions with delayed outcomes (recommendation re-ranking over session, inventory control, robotics, trading, adaptive tutoring).
- System dynamics depend on actions; need long-term optimization.

### 4.3 Algorithm Families
| Family | Examples | Notes |
|--------|----------|-------|
| Value-Based | Q-Learning, DQN | Learn action-value table / network |
| Policy Gradient | REINFORCE, PPO | Directly optimize policy; stable with clipping/baselines |
| Actor-Critic | A2C, PPO, SAC | Combine value + policy for variance reduction |
| Model-Based | Dyna-Q, MuZero | Learn environment model for planning |

### 4.4 Minimal Q-Learning (tabular) Example
```python
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1  # learning rate
gamma = 0.99
epsilon = 1.0
decay = 0.995

for episode in range(500):
    s, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s = s2
    epsilon = max(0.05, epsilon * decay)
```

### 4.5 Evaluation
- Average episodic return over seeds.
- Sample efficiency (reward vs timesteps curve).
- Policy stability (variance of return).
- Safety constraints (violations per episode) if applicable.

---
## 5. Choosing the Right Learning Type
| Scenario | Best Paradigm | Rationale |
|----------|---------------|-----------|
| Predict next week sales | Supervised (regression) | Continuous target with history |
| Group customers for targeted campaigns | Unsupervised (clustering) | No labels; need segments |
| Improve chatbot dialog strategy | Reinforcement Learning | Sequential, reward-driven |
| Only 500 labeled images + 50k unlabeled | Semi-Supervised | Leverage large unlabeled pool |
| Detect fraudulent transactions (few labels) | Start unsupervised / anomaly → supervised fine-tune | Scarce labels & evolving patterns |

Decision Flow: Do you have labels? (Yes → Supervised). No labels but need structure? (Unsupervised). Few labels + many unlabeled? (Semi-Supervised). Is it sequential with action feedback? (Reinforcement Learning).

---
## 6. Evaluation Strategy Design (Practice Guidance)
Goal: Align metrics with business value + model purpose.

### 6.1 General Framework
1. Define objective function (business, not just metric).
2. Map objective → proxy metrics (precision at k, lifetime value uplift, RMSE on revenue).
3. Define test protocol (random split, time-based split, group split, cross-validation).
4. Address class/segment imbalance (stratified sampling, weighted loss, oversampling, focal loss).
5. Add robustness & fairness slices (by geography, device, cohort).
6. Create model comparison scoreboard with statistical significance (paired t-test, McNemar, bootstrap CI).

### 6.2 Scenario Examples
- Imbalanced fraud detection: Use Precision-Recall AUC, set operating threshold to maximize expected monetary savings = TP*avg_loss - FP*investigation_cost.
- Demand forecasting: Evaluate WMAPE & coverage of 90% prediction intervals; track bias (mean error) separately.
- Customer churn: Calibrated probability (Brier), expected uplift in retention campaign simulation.
- Recommendation session policy (RL): Off-policy evaluation (importance sampling) before online A/B.

### 6.3 Hands-On Exercise Ideas
| Exercise | Task | Deliverable |
|----------|------|-------------|
| Threshold Tuning | Binary classifier | Plot F1 vs threshold; choose business-optimal |
| Drift Monitoring | Rolling window evaluation | Chart ROC AUC per month; explain shifts |
| Cluster Stability | Bootstrap clustering | Adjusted Rand index distribution |
| Pseudo-Label Audit | Semi-supervised | Precision of pseudo-labels (sample of 100) |
| RL Off-Policy Eval | Logged bandit data | IPS / DR estimate vs naive average |

---
## 7. Comparing Supervised vs Unsupervised (Practice Guidance)
| Dimension | Supervised | Unsupervised |
|----------|-----------|--------------|
| Data Requirement | Labeled | Unlabeled |
| Objective Clarity | Explicit (target) | Implicit (structure) |
| Evaluation | Direct metrics vs ground truth | Indirect or internal indices |
| Typical Risk | Overfitting to labels; leakage | Spurious clusters; over-interpretation |
| Business Use | Prediction & decision scoring | Exploration, segmentation, anomaly seeding |

Key Practice: Run unsupervised clustering first to engineer group-based features → feed into supervised model to improve performance. Evaluate uplift by ablation (with vs without cluster features).

---
## 8. Project: Customer Segmentation Analysis
Design an end-to-end unsupervised segmentation workflow + optional supervised extension.

### 8.1 Goal
Identify actionable customer segments to improve marketing personalization and retention.

### 8.2 Data Assumptions
Transactions + customer master table with features: recency, frequency, monetary (RFM), product category diversity, acquisition channel, tenure, support_tickets, NPS score.

### 8.3 Steps
1. Data Prep: Aggregate to customer-level; compute RFM, ratios (tickets per order), churn proxy (days_since_last > threshold).
2. Feature Selection: Remove leakage (future revenue). Log-transform skewed monetary; scale features.
3. Dimensionality Reduction (PCA for visualization, keep components with >70–80% variance cumulative).
4. Clustering: Try K-Means for baseline; evaluate silhouette & business interpretability. Optionally density method (HDBSCAN) for irregular shapes.
5. Profiling: Per-cluster mean/median + distribution of key KPIs; label clusters (e.g., "High Value Loyal", "At-Risk High Potential").
6. Validation: Stability via bootstrap; marketing response simulation using past campaign outcomes by cluster.
7. Action Mapping: For each cluster define targeted strategy + KPI to monitor.
8. (Optional) Supervised Layer: Train model to predict cluster assignment for new customers in real-time pipeline.

### 8.4 Example Code Skeleton
```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

cust = pd.read_csv('customers.csv')
cust['monetary'] = cust['revenue_12m']
cust['frequency'] = cust['orders_12m']
cust['recency'] = cust['days_since_last_order']
cust['avg_order_value'] = cust['monetary'] / np.maximum(cust['frequency'], 1)

features = ['recency','frequency','monetary','avg_order_value','nps','tickets','tenure_months']
X = cust[features]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_scaled)

best_k = 0; best_score = -1
from sklearn.metrics import silhouette_score
for k in range(3, 10):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    if score > best_score:
        best_score, best_k, best_model = score, k, km

cust['segment'] = best_model.predict(X_pca)
profile = cust.groupby('segment')[features].agg(['mean','median'])
counts = cust['segment'].value_counts()
```

### 8.5 Deliverables
- Notebook: data prep → clustering → profiling → action mapping.
- Segment dictionary (id, label, description, recommended tactic).
- Evaluation appendix: stability metrics + business lift estimation.

### 8.6 Extension Ideas
- Add churn prediction model using segments as features.
- Apply RFM quantile encoding & compare clustering quality.
- Replace K-Means with Gaussian Mixture; compare BIC.
- Build dashboard (Streamlit) to explore segments.

---
## 9. Practice Roadmap
Week 1: Supervised fundamentals (regression + classification) → implement baseline + cross-validation.
Week 2: Evaluation deep dive; threshold tuning; calibration.
Week 3: Clustering & dimensionality reduction; segment profiling exercises.
Week 4: Anomaly detection + association rules; build synthetic evaluation harness.
Week 5: Semi-supervised mini project (pseudo-labeling on small labeled set).
Week 6: RL basics with tabular + simple DQN; analyze reward curves.
Week 7: Customer segmentation project full write-up.

---
## 10. Quick Reference Metric Selection
| Goal | Primary Metrics | Secondary |
|------|-----------------|-----------|
| Imbalanced classification | PR AUC, F1 at chosen threshold | ROC AUC, calibration |
| Forecasting | WMAPE, RMSE | Coverage %, bias |
| Clustering | Silhouette, stability | Business KPI variance |
| Anomaly detection | Precision@k, Recall (if labels) | ROC AUC (if labels), alert volume |
| RL policy | Average return | Std of return, sample efficiency |
| Semi-supervised | Validation AUC vs baseline | Pseudo-label precision |

---
## 11. Common Pitfalls & Anti-Patterns
- Data leakage: using post-outcome features (e.g., future revenue in training).
- Over-indexing on single metric: ignore cost asymmetry or calibration.
- Interpreting every cluster as meaningful: apply domain sanity checks.
- Blindly trusting pseudo-labels: audit & set confidence thresholds.
- RL without offline safety checks: deploy only after off-policy evaluation.

---
## 12. Next Steps
1. Convert snippets into notebooks inside corresponding folders (`03_supervised_learning`, `04_unsupervised_learning`, etc.).
2. Add tests for data leakage detection utility (optional advanced).
3. Implement customer segmentation project in `12_projects` with README + Streamlit dashboard.

---
Feel free to adapt these templates. Start small, measure, iterate, and tie each model decision back to business impact.
