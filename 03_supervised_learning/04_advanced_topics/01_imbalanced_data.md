# Handling Imbalanced Datasets: When Life Isn't Fair ⚖️

## The Imbalanced Data Problem 🎯

Imagine you're trying to detect rare diseases, fraud, or manufacturing defects. In real life, these events are thankfully rare! But this creates a challenge: your dataset might be 99.9% "normal" cases and only 0.1% "interesting" cases.

**Most algorithms assume balanced classes.** When they don't get them, they often just learn to predict the majority class and call it a day!

## Why Accuracy Lies with Imbalanced Data 🚫

Let's see the problem in action:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Create severely imbalanced dataset (like real fraud detection)
X_imb, y_imb = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_redundant=10,
    n_clusters_per_class=1,
    weights=[0.995, 0.005],  # 99.5% vs 0.5% - extremely imbalanced!
    random_state=42
)

print(f"Class distribution:")
unique, counts = np.unique(y_imb, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:,} samples ({count/len(y_imb):.1%})")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)

# Train a standard classifier
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)

# Calculate metrics
accuracy = rf_baseline.score(X_test, y_test)
precision = precision_score(y_test, y_pred_baseline)
recall = recall_score(y_test, y_pred_baseline)
f1 = f1_score(y_test, y_pred_baseline)

print(f"\n📊 BASELINE MODEL RESULTS:")
print(f"Accuracy: {accuracy:.3f} - Looks great! 🎉")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f} - Uh oh... 😟")
print(f"F1-Score: {f1:.3f}")

# The shocking truth
cm = confusion_matrix(y_test, y_pred_baseline)
print(f"\n🎭 THE TRUTH (Confusion Matrix):")
print(f"True Negatives: {cm[0,0]:,} (correctly predicted normal)")
print(f"False Positives: {cm[0,1]:,} (false alarms)")
print(f"False Negatives: {cm[1,0]:,} (missed fraud cases!)")
print(f"True Positives: {cm[1,1]:,} (caught fraud cases)")

fraud_caught = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
print(f"\n💸 BUSINESS IMPACT:")
print(f"Fraud detection rate: {fraud_caught:.1%}")
print(f"That means {cm[1,0]} fraud cases went undetected!")
```

## Solution 1: Resampling Techniques 🔄

### Oversampling: Creating More Minority Samples

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# 1. Random Oversampling (simple duplication)
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

print(f"Original training set:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:,} samples")

print(f"\nAfter Random Oversampling:")
unique, counts = np.unique(y_ros, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:,} samples")

# 2. SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# 3. ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

# Visualize the difference (using 2D projection for visualization)
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)
X_smote_2d = pca.transform(X_smote)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original data
scatter = axes[0,0].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, alpha=0.6)
axes[0,0].set_title('Original Imbalanced Data')
plt.colorbar(scatter, ax=axes[0,0])

# SMOTE data
scatter = axes[0,1].scatter(X_smote_2d[:, 0], X_smote_2d[:, 1], c=y_smote, alpha=0.6)
axes[0,1].set_title('After SMOTE')
plt.colorbar(scatter, ax=axes[0,1])

# Compare performance
models_to_test = {
    'Baseline (Original)': (X_train, y_train),
    'Random Oversampling': (X_ros, y_ros),
    'SMOTE': (X_smote, y_smote),
    'ADASYN': (X_adasyn, y_adasyn)
}

results = {}
for name, (X_train_method, y_train_method) in models_to_test.items():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_method, y_train_method)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'Accuracy': model.score(X_test, y_test),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

# Plot comparison
results_df = pd.DataFrame(results).T
axes[1,0].bar(results_df.index, results_df['Recall'], alpha=0.7)
axes[1,0].set_title('Recall Comparison')
axes[1,0].set_ylabel('Recall')
axes[1,0].tick_params(axis='x', rotation=45)

axes[1,1].bar(results_df.index, results_df['F1'], alpha=0.7)
axes[1,1].set_title('F1-Score Comparison')
axes[1,1].set_ylabel('F1-Score')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\n🎯 RESAMPLING TECHNIQUES COMPARISON:")
print(results_df.round(3))
```

### Understanding SMOTE: Synthetic Data Generation

```python
def visualize_smote_process():
    """
    Show how SMOTE creates synthetic examples
    """
    # Create simple 2D dataset for visualization
    np.random.seed(42)
    
    # Majority class (many samples)
    majority_samples = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 200)
    majority_labels = np.zeros(200)
    
    # Minority class (few samples)  
    minority_samples = np.random.multivariate_normal([6, 6], [[0.5, 0.1], [0.1, 0.5]], 20)
    minority_labels = np.ones(20)
    
    # Combine
    X_simple = np.vstack([majority_samples, minority_samples])
    y_simple = np.hstack([majority_labels, minority_labels])
    
    # Apply SMOTE
    smote_simple = SMOTE(random_state=42, k_neighbors=3)
    X_smote_simple, y_smote_simple = smote_simple.fit_resample(X_simple, y_simple)
    
    # Plot before and after
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before SMOTE
    majority_mask = y_simple == 0
    minority_mask = y_simple == 1
    
    axes[0].scatter(X_simple[majority_mask, 0], X_simple[majority_mask, 1], 
                   c='blue', alpha=0.6, label=f'Majority (n={majority_mask.sum()})')
    axes[0].scatter(X_simple[minority_mask, 0], X_simple[minority_mask, 1], 
                   c='red', s=100, label=f'Minority (n={minority_mask.sum()})')
    axes[0].set_title('Before SMOTE: Severely Imbalanced')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After SMOTE
    majority_mask_smote = y_smote_simple == 0
    minority_mask_smote = y_smote_simple == 1
    
    # Original minority samples
    original_minority_idx = np.arange(len(X_simple))[minority_mask]
    
    axes[1].scatter(X_smote_simple[majority_mask_smote, 0], X_smote_simple[majority_mask_smote, 1],
                   c='blue', alpha=0.6, label=f'Majority (n={majority_mask_smote.sum()})')
    
    # Show original minority samples
    axes[1].scatter(X_simple[minority_mask, 0], X_simple[minority_mask, 1],
                   c='red', s=100, edgecolors='black', linewidth=2, 
                   label=f'Original Minority (n={minority_mask.sum()})')
    
    # Show synthetic minority samples
    synthetic_start_idx = len(X_simple)
    synthetic_minority = X_smote_simple[synthetic_start_idx:][y_smote_simple[synthetic_start_idx:] == 1]
    axes[1].scatter(synthetic_minority[:, 0], synthetic_minority[:, 1],
                   c='orange', s=60, alpha=0.8, marker='^',
                   label=f'Synthetic Minority (n={len(synthetic_minority)})')
    
    axes[1].set_title('After SMOTE: Balanced with Synthetic Samples')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"🧪 HOW SMOTE WORKS:")
    print(f"1. For each minority sample, find k nearest minority neighbors")
    print(f"2. Create synthetic sample along line connecting sample to neighbor")
    print(f"3. Repeat until classes are balanced")
    print(f"4. Result: {len(synthetic_minority)} new synthetic samples created")

visualize_smote_process()
```

### Undersampling: Reducing Majority Samples

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Tomek Links (removes borderline samples)
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)

# Edited Nearest Neighbours (removes noisy samples)
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X_train, y_train)

print(f"📉 UNDERSAMPLING TECHNIQUES COMPARISON:")
techniques = {
    'Original': (X_train, y_train),
    'Random Undersampling': (X_rus, y_rus),
    'Tomek Links': (X_tomek, y_tomek),
    'Edited Nearest Neighbours': (X_enn, y_enn)
}

for name, (X_data, y_data) in techniques.items():
    unique, counts = np.unique(y_data, return_counts=True)
    total = len(y_data)
    print(f"\n{name}:")
    print(f"  Total samples: {total:,}")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} ({count/total:.1%})")
```

### Combined Approaches: Best of Both Worlds

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)

# SMOTE + Edited Nearest Neighbours
smote_enn = SMOTEENN(random_state=42)
X_smote_enn, y_smote_enn = smote_enn.fit_resample(X_train, y_train)

print(f"🔄 COMBINED TECHNIQUES:")
combined_techniques = {
    'SMOTE + Tomek': (X_smote_tomek, y_smote_tomek),
    'SMOTE + ENN': (X_smote_enn, y_smote_enn)
}

for name, (X_data, y_data) in combined_techniques.items():
    unique, counts = np.unique(y_data, return_counts=True)
    total = len(y_data)
    print(f"\n{name}: {total:,} total samples")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} ({count/total:.1%})")
```

## Solution 2: Cost-Sensitive Learning 💰

Instead of changing the data, change how the algorithm learns:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights automatically
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train), 
    y=y_train
)

print(f"🏷️ CALCULATED CLASS WEIGHTS:")
for cls, weight in zip(np.unique(y_train), class_weights):
    print(f"Class {cls}: weight = {weight:.2f}")
    if weight > 1:
        print(f"  → {weight:.1f}x more important than baseline")

# Compare models with and without class weights
models_cost_sensitive = {
    'Baseline Random Forest': RandomForestClassifier(random_state=42),
    'Weighted Random Forest': RandomForestClassifier(
        class_weight='balanced', random_state=42
    ),
    'Custom Weighted RF': RandomForestClassifier(
        class_weight={0: 1, 1: 100},  # Make minority 100x more important
        random_state=42
    ),
    'Baseline Logistic': LogisticRegression(random_state=42, max_iter=1000),
    'Weighted Logistic': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    )
}

results_cost = {}
for name, model in models_cost_sensitive.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results_cost[name] = {
        'Accuracy': model.score(X_test, y_test),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }

# Display results
results_cost_df = pd.DataFrame(results_cost).T
print(f"\n💰 COST-SENSITIVE LEARNING RESULTS:")
print(results_cost_df.round(3))

# Visualize the trade-offs
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['Precision', 'Recall', 'F1', 'AUC']
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = results_cost_df[metric]
    bars = ax.bar(range(len(values)), values, alpha=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.grid(True, alpha=0.3)
    
    # Highlight best performer
    best_idx = values.idxmax()
    best_bar_idx = values.index.get_loc(best_idx)
    bars[best_bar_idx].set_color('gold')

plt.tight_layout()
plt.show()
```

## Solution 3: Ensemble Methods for Imbalanced Data 🎭

### Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

# Balanced Bagging with different base learners
bb_tree = BalancedBaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bb_tree.fit(X_train, y_train)
y_pred_bb = bb_tree.predict(X_test)

# Easy Ensemble (combines boosting with resampling)
from imblearn.ensemble import EasyEnsembleClassifier

easy_ensemble = EasyEnsembleClassifier(n_estimators=50, random_state=42)
easy_ensemble.fit(X_train, y_train)
y_pred_easy = easy_ensemble.predict(X_test)

# Compare ensemble approaches
ensemble_results = {
    'Balanced RF': {
        'Precision': precision_score(y_test, y_pred_brf),
        'Recall': recall_score(y_test, y_pred_brf),
        'F1': f1_score(y_test, y_pred_brf)
    },
    'Balanced Bagging': {
        'Precision': precision_score(y_test, y_pred_bb),
        'Recall': recall_score(y_test, y_pred_bb),
        'F1': f1_score(y_test, y_pred_bb)
    },
    'Easy Ensemble': {
        'Precision': precision_score(y_test, y_pred_easy),
        'Recall': recall_score(y_test, y_pred_easy),
        'F1': f1_score(y_test, y_pred_easy)
    }
}

ensemble_df = pd.DataFrame(ensemble_results).T
print(f"\n🎭 ENSEMBLE METHODS FOR IMBALANCED DATA:")
print(ensemble_df.round(3))
```

## Solution 4: Threshold Tuning 🎚️

Sometimes the best approach is to change the decision threshold:

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(model, X_test, y_test, optimize_for='f1'):
    """
    Find the best decision threshold for your business needs
    """
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    if optimize_for == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = f1_scores[optimal_idx]
        metric_name = "F1"
        
    elif optimize_for == 'precision':
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        # Find threshold where precision >= 0.9
        high_precision_mask = precision >= 0.9
        if high_precision_mask.any():
            optimal_idx = np.where(high_precision_mask)[0][-1]  # Last index with high precision
            optimal_threshold = thresholds[optimal_idx]
            optimal_score = precision[optimal_idx]
        else:
            optimal_threshold = 0.9
            optimal_score = 0
        metric_name = "Precision"
        
    elif optimize_for == 'recall':
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        # Find threshold where recall >= 0.9
        high_recall_mask = recall >= 0.9
        if high_recall_mask.any():
            optimal_idx = np.where(high_recall_mask)[0][0]  # First index with high recall
            optimal_threshold = thresholds[optimal_idx]
            optimal_score = recall[optimal_idx]
        else:
            optimal_threshold = 0.1
            optimal_score = 0
        metric_name = "Recall"
    
    return optimal_threshold, optimal_score, y_proba

# Train a model and find optimal thresholds
model_threshold = RandomForestClassifier(n_estimators=100, random_state=42)
model_threshold.fit(X_train, y_train)

# Find thresholds optimized for different goals
threshold_f1, score_f1, y_proba = find_optimal_threshold(
    model_threshold, X_test, y_test, 'f1'
)
threshold_precision, score_precision, _ = find_optimal_threshold(
    model_threshold, X_test, y_test, 'precision'
)
threshold_recall, score_recall, _ = find_optimal_threshold(
    model_threshold, X_test, y_test, 'recall'
)

print(f"🎚️ OPTIMAL THRESHOLDS:")
print(f"Best F1 threshold: {threshold_f1:.3f} (F1 = {score_f1:.3f})")
print(f"High precision threshold: {threshold_precision:.3f} (Precision ≥ 90%)")
print(f"High recall threshold: {threshold_recall:.3f} (Recall ≥ 90%)")

# Compare predictions with different thresholds
thresholds_to_compare = [0.5, threshold_f1, threshold_precision, threshold_recall]
threshold_names = ['Default (0.5)', 'Best F1', 'High Precision', 'High Recall']

plt.figure(figsize=(15, 10))

for i, (threshold, name) in enumerate(zip(thresholds_to_compare, threshold_names)):
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    
    # Plot confusion matrix
    plt.subplot(2, 2, i+1)
    cm = confusion_matrix(y_test, y_pred_threshold)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}\nP:{precision:.2f} R:{recall:.2f} F1:{f1:.2f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()
```

## Real-World Case Study: Credit Card Fraud 💳

Let's apply everything to a realistic fraud detection scenario:

```python
def create_realistic_fraud_dataset():
    """
    Create a dataset that mimics real credit card fraud characteristics
    """
    np.random.seed(42)
    n_transactions = 50000
    
    # Normal transactions
    n_normal = int(n_transactions * 0.998)  # 99.8% normal
    normal_features = {
        'amount': np.random.lognormal(3, 1, n_normal),  # Log-normal distribution
        'hour': np.random.choice(24, n_normal),
        'day_of_week': np.random.choice(7, n_normal),
        'merchant_category': np.random.choice(20, n_normal),
        'is_weekend': np.random.binomial(1, 0.3, n_normal),
        'previous_transactions_today': np.random.poisson(2, n_normal)
    }
    
    # Fraudulent transactions (different patterns)
    n_fraud = n_transactions - n_normal
    fraud_features = {
        'amount': np.random.lognormal(5, 0.5, n_fraud),  # Higher amounts
        'hour': np.random.choice([2, 3, 4, 23, 0, 1], n_fraud),  # Unusual hours
        'day_of_week': np.random.choice(7, n_fraud),
        'merchant_category': np.random.choice([15, 16, 17, 18, 19], n_fraud),  # Specific categories
        'is_weekend': np.random.binomial(1, 0.7, n_fraud),  # More weekend fraud
        'previous_transactions_today': np.random.poisson(0.5, n_fraud)  # Fewer previous transactions
    }
    
    # Combine features
    features = {}
    for key in normal_features:
        features[key] = np.concatenate([normal_features[key], fraud_features[key]])
    
    # Create labels
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Create DataFrame
    fraud_df = pd.DataFrame(features)
    fraud_df['is_fraud'] = labels
    
    return fraud_df

# Create and analyze the fraud dataset
fraud_df = create_realistic_fraud_dataset()

print(f"💳 FRAUD DETECTION DATASET:")
print(f"Total transactions: {len(fraud_df):,}")
print(f"Fraud rate: {fraud_df['is_fraud'].mean():.3%}")

# Analyze feature differences between fraud and normal
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

features_to_plot = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                   'is_weekend', 'previous_transactions_today']

for i, feature in enumerate(features_to_plot):
    if feature in ['hour', 'day_of_week', 'merchant_category']:
        # Categorical features - use bar plots
        fraud_data = fraud_df[fraud_df['is_fraud'] == 1][feature]
        normal_data = fraud_df[fraud_df['is_fraud'] == 0][feature]
        
        fraud_counts = pd.Series(fraud_data).value_counts().sort_index()
        normal_counts = pd.Series(normal_data).value_counts().sort_index()
        
        x_pos = np.arange(len(fraud_counts))
        width = 0.35
        
        axes[i].bar(x_pos - width/2, normal_counts.values / len(normal_data), 
                   width, alpha=0.7, label='Normal')
        axes[i].bar(x_pos + width/2, fraud_counts.values / len(fraud_data), 
                   width, alpha=0.7, label='Fraud')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(fraud_counts.index)
        
    else:
        # Continuous features - use histograms
        fraud_data = fraud_df[fraud_df['is_fraud'] == 1][feature]
        normal_data = fraud_df[fraud_df['is_fraud'] == 0][feature]
        
        axes[i].hist(normal_data, bins=30, alpha=0.7, label='Normal', density=True)
        axes[i].hist(fraud_data, bins=30, alpha=0.7, label='Fraud', density=True)
    
    axes[i].set_title(f'{feature.replace("_", " ").title()}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Comprehensive Fraud Detection Pipeline

```python
def build_fraud_detection_pipeline():
    """
    Complete pipeline for fraud detection
    """
    # Prepare features and target
    X_fraud = fraud_df.drop('is_fraud', axis=1)
    y_fraud = fraud_df['is_fraud']
    
    # Split data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )
    
    # Define different strategies
    strategies = {
        'Baseline': {
            'resampler': None,
            'model': RandomForestClassifier(random_state=42)
        },
        'SMOTE': {
            'resampler': SMOTE(random_state=42),
            'model': RandomForestClassifier(random_state=42)
        },
        'Cost-Sensitive': {
            'resampler': None,
            'model': RandomForestClassifier(class_weight='balanced', random_state=42)
        },
        'SMOTE + Cost-Sensitive': {
            'resampler': SMOTE(random_state=42),
            'model': RandomForestClassifier(class_weight='balanced', random_state=42)
        },
        'Threshold Tuned': {
            'resampler': None,
            'model': RandomForestClassifier(random_state=42),
            'custom_threshold': True
        }
    }
    
    # Test each strategy
    fraud_results = {}
    
    for strategy_name, config in strategies.items():
        print(f"Testing {strategy_name}...")
        
        # Apply resampling if specified
        if config['resampler'] is not None:
            X_train_strategy, y_train_strategy = config['resampler'].fit_resample(
                X_train_fraud, y_train_fraud
            )
        else:
            X_train_strategy, y_train_strategy = X_train_fraud, y_train_fraud
        
        # Train model
        model = config['model']
        model.fit(X_train_strategy, y_train_strategy)
        
        # Make predictions
        if config.get('custom_threshold', False):
            # Find optimal threshold
            y_proba = model.predict_proba(X_test_fraud)[:, 1]
            optimal_threshold, _, _ = find_optimal_threshold(
                model, X_test_fraud, y_test_fraud, 'f1'
            )
            y_pred = (y_proba >= optimal_threshold).astype(int)
        else:
            y_pred = model.predict(X_test_fraud)
            y_proba = model.predict_proba(X_test_fraud)[:, 1]
        
        # Calculate business metrics
        cm = confusion_matrix(y_test_fraud, y_pred)
        
        # Business impact calculation
        fraud_missed = cm[1, 0]  # False negatives
        false_alarms = cm[0, 1]  # False positives
        fraud_caught = cm[1, 1]  # True positives
        
        avg_fraud_amount = 500  # Average fraud amount
        investigation_cost = 50  # Cost to investigate each alert
        
        money_saved = fraud_caught * avg_fraud_amount
        money_lost = fraud_missed * avg_fraud_amount
        investigation_costs = false_alarms * investigation_cost
        
        net_benefit = money_saved - money_lost - investigation_costs
        
        fraud_results[strategy_name] = {
            'Precision': precision_score(y_test_fraud, y_pred),
            'Recall': recall_score(y_test_fraud, y_pred),
            'F1': f1_score(y_test_fraud, y_pred),
            'AUC': roc_auc_score(y_test_fraud, y_proba),
            'Fraud_Caught': fraud_caught,
            'Fraud_Missed': fraud_missed,
            'False_Alarms': false_alarms,
            'Net_Benefit': net_benefit
        }
    
    return fraud_results

# Run the comprehensive analysis
fraud_comparison = build_fraud_detection_pipeline()
fraud_results_df = pd.DataFrame(fraud_comparison).T

print(f"\n💰 BUSINESS IMPACT ANALYSIS:")
print(fraud_results_df.round(2))

# Visualize business impact
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.bar(fraud_results_df.index, fraud_results_df['Recall'], alpha=0.7)
plt.title('Fraud Detection Rate')
plt.ylabel('Recall')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
plt.bar(fraud_results_df.index, fraud_results_df['Precision'], alpha=0.7)
plt.title('Precision (Alert Accuracy)')
plt.ylabel('Precision')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
plt.bar(fraud_results_df.index, fraud_results_df['Net_Benefit'], alpha=0.7)
plt.title('Net Business Benefit ($)')
plt.ylabel('Net Benefit')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
plt.bar(fraud_results_df.index, fraud_results_df['Fraud_Missed'], alpha=0.7, color='red')
plt.title('Fraud Cases Missed')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
plt.bar(fraud_results_df.index, fraud_results_df['False_Alarms'], alpha=0.7, color='orange')
plt.title('False Alarms')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
plt.bar(fraud_results_df.index, fraud_results_df['AUC'], alpha=0.7, color='green')
plt.title('AUC Score')
plt.ylabel('AUC')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Identify the best strategy
best_strategy = fraud_results_df.loc[fraud_results_df['Net_Benefit'].idxmax()]
print(f"\n🏆 WINNING STRATEGY: {fraud_results_df['Net_Benefit'].idxmax()}")
print(f"Net benefit: ${best_strategy['Net_Benefit']:,.0f}")
print(f"Catches {best_strategy['Fraud_Caught']:.0f} fraud cases")
print(f"Misses {best_strategy['Fraud_Missed']:.0f} fraud cases")
print(f"Generates {best_strategy['False_Alarms']:.0f} false alarms")
```

## Advanced Metrics for Imbalanced Data 📊

### Matthews Correlation Coefficient (MCC)

MCC is considered one of the best metrics for imbalanced data:

```python
from sklearn.metrics import matthews_corrcoef

def calculate_advanced_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate advanced metrics particularly useful for imbalanced data
    """
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Balanced Accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Average Precision Score (area under PR curve)
    if y_proba is not None:
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_true, y_proba)
    else:
        avg_precision = None
    
    print(f"🎯 ADVANCED METRICS FOR IMBALANCED DATA:")
    print(f"Matthews Correlation Coefficient: {mcc:.3f}")
    print(f"  → Ranges from -1 to 1 (1 = perfect, 0 = random, -1 = perfectly wrong)")
    print(f"  → Considers all four confusion matrix categories")
    
    print(f"\nBalanced Accuracy: {balanced_acc:.3f}")
    print(f"  → Average of recall for each class")
    print(f"  → Less biased towards majority class than regular accuracy")
    
    print(f"\nCohen's Kappa: {kappa:.3f}")
    print(f"  → Agreement between predictions and truth, accounting for chance")
    print(f"  → 0.6-0.8 = substantial agreement, 0.8+ = almost perfect")
    
    if avg_precision is not None:
        print(f"\nAverage Precision: {avg_precision:.3f}")
        print(f"  → Area under precision-recall curve")
        print(f"  → Better than AUC for imbalanced data")
    
    return mcc, balanced_acc, kappa, avg_precision

# Test on our best fraud detection model
best_model_name = fraud_results_df['Net_Benefit'].idxmax()
print(f"Analyzing advanced metrics for: {best_model_name}")

# Get the predictions for the best model (you'd need to implement this based on your best strategy)
# For demonstration, using SMOTE + Cost-Sensitive
smote = SMOTE(random_state=42)
X_train_best, y_train_best = smote.fit_resample(X_train_fraud, y_train_fraud)

best_model = RandomForestClassifier(class_weight='balanced', random_state=42)
best_model.fit(X_train_best, y_train_best)
y_pred_best = best_model.predict(X_test_fraud)
y_proba_best = best_model.predict_proba(X_test_fraud)[:, 1]

mcc, balanced_acc, kappa, avg_precision = calculate_advanced_metrics(
    y_test_fraud, y_pred_best, y_proba_best
)
```

## When to Use Each Technique 🤔

### Decision Framework

```python
def imbalance_strategy_guide(imbalance_ratio, dataset_size, business_context):
    """
    Guide for choosing the right imbalance handling strategy
    """
    print(f"📋 IMBALANCE STRATEGY RECOMMENDATION:")
    print(f"Imbalance ratio: {imbalance_ratio:.1%}")
    print(f"Dataset size: {dataset_size:,} samples")
    print(f"Business context: {business_context}")
    
    recommendations = []
    
    # Based on imbalance severity
    if imbalance_ratio < 0.01:  # < 1% minority class
        recommendations.append("🔴 Severe imbalance - combine multiple techniques")
        if dataset_size > 10000:
            recommendations.append("✅ Try: SMOTE + Ensemble methods + Threshold tuning")
        else:
            recommendations.append("✅ Try: Cost-sensitive learning + Threshold tuning")
    
    elif imbalance_ratio < 0.1:  # 1-10% minority class
        recommendations.append("🟡 Moderate imbalance")
        recommendations.append("✅ Try: Class weights or SMOTE")
        
    else:  # > 10% minority class
        recommendations.append("🟢 Mild imbalance")
        recommendations.append("✅ Try: Class weights first")
    
    # Based on dataset size
    if dataset_size < 1000:
        recommendations.append("⚠️ Small dataset - be careful with oversampling")
        recommendations.append("✅ Prefer: Cost-sensitive learning or simpler models")
    elif dataset_size > 100000:
        recommendations.append("✅ Large dataset - undersampling is viable")
        
    # Based on business context
    if business_context == "medical":
        recommendations.append("🏥 Medical context - prioritize recall (don't miss cases)")
        recommendations.append("✅ Strategy: High recall threshold + interpretable model")
    elif business_context == "marketing":
        recommendations.append("📧 Marketing context - balance precision and recall")
        recommendations.append("✅ Strategy: F1-optimized threshold")
    elif business_context == "fraud":
        recommendations.append("💰 Fraud context - consider business costs")
        recommendations.append("✅ Strategy: Custom cost function + ensemble methods")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return recommendations

# Example usage
fraud_rate = fraud_df['is_fraud'].mean()
dataset_size = len(fraud_df)

recommendations = imbalance_strategy_guide(fraud_rate, dataset_size, "fraud")
```

## Production Considerations 🏭

### Monitoring Imbalanced Models

```python
def create_monitoring_dashboard(model, X_test, y_test, y_pred, y_proba):
    """
    Create a monitoring dashboard for imbalanced data models
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Prediction distribution
    axes[0,0].hist(y_proba, bins=50, alpha=0.7)
    axes[0,0].set_xlabel('Prediction Probability')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Prediction Probability Distribution')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Calibration plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10
    )
    axes[0,1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    axes[0,1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes[0,1].set_xlabel('Mean Predicted Probability')
    axes[0,1].set_ylabel('Fraction of Positives')
    axes[0,1].set_title('Calibration Plot')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    axes[0,2].plot(recall, precision, linewidth=2)
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        axes[1,0].barh(range(len(feature_imp)), feature_imp['Importance'])
        axes[1,0].set_yticks(range(len(feature_imp)))
        axes[1,0].set_yticklabels(feature_imp['Feature'])
        axes[1,0].set_xlabel('Importance')
        axes[1,0].set_title('Feature Importance')
    
    # 5. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Confusion Matrix')
    axes[1,1].set_ylabel('True Label')
    axes[1,1].set_xlabel('Predicted Label')
    
    # 6. Performance over time (if you have temporal data)
    # For demonstration, we'll show performance by prediction confidence
    confidence_bins = pd.cut(y_proba, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    performance_by_confidence = pd.DataFrame({
        'Confidence': confidence_bins,
        'True_Label': y_test,
        'Predicted': y_pred
    })
    
    precision_by_conf = performance_by_confidence.groupby('Confidence').apply(
        lambda x: precision_score(x['True_Label'], x['Predicted']) if len(x) > 0 else 0
    )
    
    axes[1,2].bar(range(len(precision_by_conf)), precision_by_conf.values, alpha=0.7)
    axes[1,2].set_xticks(range(len(precision_by_conf)))
    axes[1,2].set_xticklabels(precision_by_conf.index, rotation=45)
    axes[1,2].set_ylabel('Precision')
    axes[1,2].set_title('Precision by Confidence Level')
    
    plt.tight_layout()
    plt.show()

# Create monitoring dashboard for our best model
create_monitoring_dashboard(best_model, X_test_fraud, y_test_fraud, y_pred_best, y_proba_best)
```

## Key Takeaways 🎯

### Do's ✅
- **Always check class distribution** before training
- **Use appropriate metrics** (precision, recall, F1, AUC, MCC)
- **Consider business costs** when optimizing
- **Combine multiple techniques** for severe imbalance
- **Monitor model performance** in production
- **Use stratified sampling** to maintain proportions

### Don'ts ❌
- **Don't rely on accuracy alone** for imbalanced data
- **Don't oversample before splitting** (causes data leakage)
- **Don't ignore the business context** when choosing metrics
- **Don't assume balanced techniques work** for imbalanced problems
- **Don't forget to validate** with proper cross-validation

### Business Impact Checklist 💼
- [ ] Define costs of false positives vs false negatives
- [ ] Choose evaluation metrics that reflect business goals
- [ ] Set decision thresholds based on business constraints
- [ ] Plan for model monitoring and maintenance
- [ ] Prepare explanations for stakeholders

## Real-World Applications 🌍

### 1. Medical Diagnosis 🏥
- **Challenge**: Rare diseases (0.1-1% prevalence)
- **Strategy**: High recall + interpretable models
- **Key metric**: Sensitivity (recall) > 95%

### 2. Quality Control 🏭
- **Challenge**: Defect rates < 1%
- **Strategy**: Cost-sensitive learning + ensemble methods
- **Key metric**: Custom cost function based on defect costs

### 3. Cybersecurity 🛡️
- **Challenge**: Attack detection (< 0.01% of network traffic)
- **Strategy**: Anomaly detection + supervised learning hybrid
- **Key metric**: Precision-recall balance for alert fatigue

### 4. Customer Analytics 📊
- **Challenge**: High-value customer identification (5-10%)
- **Strategy**: Stratified sampling + feature engineering
- **Key metric**: Business value optimization

## Next Steps 🚀

1. **Practice**: Work with the fraud detection exercise
2. **Experiment**: Try different combinations of techniques
3. **Measure**: Always validate with proper business metrics
4. **Deploy**: Plan for production monitoring
5. **Learn more**: Explore advanced ensemble methods and deep learning approaches

## Quick Exercise 💪

You're tasked with building a model to detect equipment failures in a manufacturing plant:
- **Data**: 50,000 sensor readings, 0.3% failure rate
- **Business cost**: Missing a failure costs $10,000, false alarm costs $500
- **Constraint**: Model must explain its decisions to maintenance team

Design a complete solution including:
1. Imbalance handling strategy
2. Evaluation approach
3. Business-oriented metrics
4. Production monitoring plan

*Detailed solution in `exercises/equipment_failure_detection.py`!*

---

*Next: Dive into missing data handling with `02_missing_data.md` →*
