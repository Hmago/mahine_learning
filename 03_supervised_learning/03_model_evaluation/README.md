# Model Evaluation: How Good is Your Model Really? 🎯

## Why Model Evaluation Matters 🤔

Imagine you built a spam email detector and tested it on your own emails. It works perfectly! But when you deploy it to production, it fails miserably. What happened? 

**You evaluated your model wrong.**

Model evaluation is like taking a practice test before the real exam. If done correctly, it tells you how your model will perform in the real world. If done incorrectly, you'll be in for some nasty surprises!

## The Cardinal Sin: Testing on Training Data ❌

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                          n_redundant=15, random_state=42)

# WRONG way to evaluate
model_wrong = RandomForestClassifier(random_state=42)
model_wrong.fit(X, y)
wrong_accuracy = model_wrong.score(X, y)  # Testing on same data we trained on!

print(f"'Accuracy' when testing on training data: {wrong_accuracy:.3f}")
print("This is meaningless! The model has seen this data before.")

# RIGHT way to evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_right = RandomForestClassifier(random_state=42)
model_right.fit(X_train, y_train)
right_accuracy = model_right.score(X_test, y_test)

print(f"Real accuracy on unseen data: {right_accuracy:.3f}")
print(f"Reality check: {wrong_accuracy - right_accuracy:.3f} points lower!")
```

## Classification Metrics: Beyond Simple Accuracy 📊

### The Confusion Matrix: Your Model's Report Card

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Create imbalanced dataset (like real-world scenarios)
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                                  n_features=20, random_state=42)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)

# Train a model
from sklearn.linear_model import LogisticRegression
model_imb = LogisticRegression()
model_imb.fit(X_train_imb, y_train_imb)

y_pred_imb = model_imb.predict(X_test_imb)

# Create confusion matrix
cm = confusion_matrix(y_test_imb, y_pred_imb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add explanations to the plot
plt.text(0.5, -0.1, 'True Negatives\n(Correctly predicted negative)', 
         transform=plt.gca().transAxes, ha='center', va='top')
plt.text(1.5, -0.1, 'False Positives\n(Incorrectly predicted positive)', 
         transform=plt.gca().transAxes, ha='center', va='top')

plt.show()

print("Confusion Matrix Breakdown:")
print(f"True Negatives (correct negatives): {cm[0,0]}")
print(f"False Positives (wrong positives): {cm[0,1]}")  
print(f"False Negatives (missed positives): {cm[1,0]}")
print(f"True Positives (correct positives): {cm[1,1]}")
```

### Understanding Precision and Recall 🎯

These are the most important metrics for real-world applications:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def explain_precision_recall(y_true, y_pred, positive_class_name="Positive"):
    """
    Explain precision and recall with intuitive examples
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"🎯 PRECISION: {precision:.3f}")
    print(f"   → Of all {positive_class_name} predictions, {precision:.1%} were actually correct")
    print(f"   → Think: 'When I predict {positive_class_name}, how often am I right?'")
    print(f"   → Formula: True Positives / (True Positives + False Positives)")
    print(f"   → {tp} / ({tp} + {fp}) = {precision:.3f}")
    
    print(f"\n🔍 RECALL: {recall:.3f}")
    print(f"   → Of all actual {positive_class_name} cases, {recall:.1%} were correctly identified")
    print(f"   → Think: 'How many real {positive_class_name} cases did I catch?'")
    print(f"   → Formula: True Positives / (True Positives + False Negatives)")
    print(f"   → {tp} / ({tp} + {fn}) = {recall:.3f}")
    
    print(f"\n⚖️ F1-SCORE: {f1:.3f}")
    print(f"   → Harmonic mean of precision and recall")
    print(f"   → Balances both metrics into a single number")
    
    return precision, recall, f1

# Test on our imbalanced data
precision, recall, f1 = explain_precision_recall(y_test_imb, y_pred_imb, "Rare Class")
```

### Real-World Example: Medical Diagnosis 🏥

Let's see why precision and recall matter with a cancer detection example:

```python
# Simulate cancer detection scenario
# 98% of people don't have cancer, 2% do
np.random.seed(42)
n_patients = 1000

# Create realistic medical data
medical_data = {
    'Age': np.random.normal(50, 15, n_patients),
    'Family_History': np.random.binomial(1, 0.15, n_patients),
    'Smoking': np.random.binomial(1, 0.3, n_patients),
    'Test_Result_1': np.random.normal(50, 10, n_patients),
    'Test_Result_2': np.random.normal(100, 20, n_patients)
}

# Create cancer labels (2% prevalence)
cancer_probability = (
    0.01 +  # Base rate
    medical_data['Age'] * 0.0003 +  # Age factor
    medical_data['Family_History'] * 0.05 +  # Family history
    medical_data['Smoking'] * 0.03 +  # Smoking
    medical_data['Test_Result_1'] * 0.0002 +
    medical_data['Test_Result_2'] * 0.0001
)

has_cancer = np.random.binomial(1, np.clip(cancer_probability, 0, 0.2), n_patients)

medical_df = pd.DataFrame(medical_data)
medical_df['Has_Cancer'] = has_cancer

print(f"Cancer prevalence in dataset: {has_cancer.mean():.1%}")

# Train model
X_medical = medical_df.drop('Has_Cancer', axis=1)
y_medical = medical_df['Has_Cancer']

X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_medical, y_medical, test_size=0.2, random_state=42, stratify=y_medical
)

# Two different models with different thresholds
from sklearn.metrics import precision_recall_curve

lr_medical = LogisticRegression()
lr_medical.fit(X_train_med, y_train_med)

# Get prediction probabilities
y_proba = lr_medical.predict_proba(X_test_med)[:, 1]

# Plot precision-recall curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_med, y_proba)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recall_curve, precision_curve, linewidth=2)
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve\nCancer Detection')
plt.grid(True, alpha=0.3)

# Compare different thresholds
thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for threshold in thresholds_to_test:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_test_med, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test_med, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test_med, y_pred_threshold, zero_division=0)
    
    results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

results_df = pd.DataFrame(results)

plt.subplot(1, 2, 2)
plt.plot(results_df['Threshold'], results_df['Precision'], 'o-', label='Precision')
plt.plot(results_df['Threshold'], results_df['Recall'], 'o-', label='Recall')
plt.plot(results_df['Threshold'], results_df['F1'], 'o-', label='F1-Score')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Threshold Analysis for Cancer Detection:")
print(results_df.round(3))

print("\n💡 Medical Context:")
print("• High Recall (0.9+): Don't miss any cancer cases (few false negatives)")
print("• High Precision (0.8+): Don't scare healthy people (few false positives)")  
print("• Trade-off: Usually can't maximize both simultaneously")
```

## ROC Curve: The Complete Picture 📈

ROC (Receiver Operating Characteristic) curves show the trade-off between True Positive Rate and False Positive Rate:

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test_med, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Cancer Detection')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare with different models
models_to_compare = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

plt.subplot(2, 2, 2)

for name, model in models_to_compare.items():
    model.fit(X_train_med, y_train_med)
    y_proba_model = model.predict_proba(X_test_med)[:, 1]
    
    fpr_model, tpr_model, _ = roc_curve(y_test_med, y_proba_model)
    auc_model = auc(fpr_model, tpr_model)
    
    plt.plot(fpr_model, tpr_model, linewidth=2, label=f'{name} (AUC = {auc_model:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison: Different Models')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision-Recall comparison
plt.subplot(2, 2, 3)

for name, model in models_to_compare.items():
    y_proba_model = model.predict_proba(X_test_med)[:, 1]
    precision_model, recall_model, _ = precision_recall_curve(y_test_med, y_proba_model)
    
    plt.plot(recall_model, precision_model, linewidth=2, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance comparison
plt.subplot(2, 2, 4)
rf_model = models_to_compare['Random Forest']
feature_imp = pd.DataFrame({
    'Feature': X_medical.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(range(len(feature_imp)), feature_imp['Importance'])
plt.yticks(range(len(feature_imp)), feature_imp['Feature'])
plt.xlabel('Feature Importance')
plt.title('What Matters for Cancer Detection?')

plt.tight_layout()
plt.show()
```

## Cross-Validation: The Gold Standard 🏆

Simple train/test splits can be misleading. Cross-validation gives you a more robust estimate:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Demonstrate why cross-validation matters
def compare_evaluation_methods(X, y, model, n_trials=10):
    """
    Compare single split vs cross-validation
    """
    single_split_scores = []
    cv_scores = []
    
    for trial in range(n_trials):
        # Single random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=trial
        )
        model.fit(X_train, y_train)
        single_score = model.score(X_test, y_test)
        single_split_scores.append(single_score)
        
        # Cross-validation
        cv_score = cross_val_score(model, X, y, cv=5, random_state=trial)
        cv_scores.append(cv_score.mean())
    
    return np.array(single_split_scores), np.array(cv_scores)

# Test with our data
model_test = RandomForestClassifier(n_estimators=50, random_state=42)
single_scores, cv_scores = compare_evaluation_methods(X, y, model_test)

print(f"Single Split Evaluation:")
print(f"  Mean: {single_scores.mean():.3f}")
print(f"  Std:  {single_scores.std():.3f}")
print(f"  Range: {single_scores.min():.3f} - {single_scores.max():.3f}")

print(f"\nCross-Validation:")
print(f"  Mean: {cv_scores.mean():.3f}")
print(f"  Std:  {cv_scores.std():.3f}") 
print(f"  Range: {cv_scores.min():.3f} - {cv_scores.max():.3f}")

# Visualize the difference
plt.figure(figsize=(10, 6))
plt.boxplot([single_scores, cv_scores], labels=['Single Split', 'Cross-Validation'])
plt.ylabel('Accuracy Score')
plt.title('Evaluation Method Comparison: Stability of Estimates')
plt.grid(True, alpha=0.3)
plt.show()
```

### Different Types of Cross-Validation 🔄

```python
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit

# 1. K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model_test, X, y, cv=kfold)

# 2. Stratified K-Fold (maintains class proportions)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model_test, X, y, cv=stratified_kfold)

# 3. Leave-One-Out (for small datasets)
loo = LeaveOneOut()
# loo_scores = cross_val_score(model_test, X[:50], y[:50], cv=loo)  # Small subset

print(f"K-Fold CV (5 folds): {kfold_scores.mean():.3f} (+/- {kfold_scores.std()*2:.3f})")
print(f"Stratified K-Fold: {stratified_scores.mean():.3f} (+/- {stratified_scores.std()*2:.3f})")

# For time series data
print("\n📅 Time Series Cross-Validation:")
print("Use TimeSeriesSplit for temporal data where future shouldn't predict past!")

# Visualize different CV strategies
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# K-Fold
cv_viz_data = np.arange(20)
kfold_viz = KFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(kfold_viz.split(cv_viz_data)):
    axes[0].scatter(train_idx, [fold] * len(train_idx), c='blue', s=20, alpha=0.6)
    axes[0].scatter(test_idx, [fold] * len(test_idx), c='red', s=20)
axes[0].set_title('K-Fold Cross-Validation')
axes[0].set_ylabel('Fold')

# Time Series Split
ts_split = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(ts_split.split(cv_viz_data)):
    axes[1].scatter(train_idx, [fold] * len(train_idx), c='blue', s=20, alpha=0.6)
    axes[1].scatter(test_idx, [fold] * len(test_idx), c='red', s=20)
axes[1].set_title('Time Series Cross-Validation')
axes[1].set_ylabel('Fold')

# Leave-One-Out visualization (first few folds only)
loo_viz = LeaveOneOut()
for fold, (train_idx, test_idx) in enumerate(list(loo_viz.split(cv_viz_data[:10]))[:5]):
    axes[2].scatter(train_idx, [fold] * len(train_idx), c='blue', s=20, alpha=0.6)
    axes[2].scatter(test_idx, [fold] * len(test_idx), c='red', s=20)
axes[2].set_title('Leave-One-Out Cross-Validation (first 5 folds shown)')
axes[2].set_ylabel('Fold')
axes[2].set_xlabel('Sample Index')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Training'),
                  Patch(facecolor='red', label='Testing')]
axes[0].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
```

## Regression Metrics: Measuring Prediction Quality 📏

### Mean Squared Error (MSE) vs Mean Absolute Error (MAE)

```python
def compare_regression_metrics(y_true, y_pred):
    """
    Compare different regression metrics with intuitive explanations
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate which metric is affected more by outliers
    errors = y_true - y_pred
    largest_error_idx = np.argmax(np.abs(errors))
    largest_error = errors[largest_error_idx]
    
    print(f"📊 REGRESSION METRICS COMPARISON:")
    print(f"   Mean Squared Error (MSE): {mse:.2f}")
    print(f"   → Squares all errors (penalizes large errors heavily)")
    print(f"   → Largest error: {largest_error:.2f}, contributes {largest_error**2:.2f} to MSE")
    
    print(f"\n   Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   → Same units as your target variable")
    print(f"   → Think: 'On average, predictions are off by {rmse:.1f} units'")
    
    print(f"\n   Mean Absolute Error (MAE): {mae:.2f}")
    print(f"   → Average absolute difference")
    print(f"   → Less sensitive to outliers than MSE")
    print(f"   → Largest error contributes only {abs(largest_error):.2f} to MAE")
    
    return mse, rmse, mae

# Create example with outlier
y_true_example = np.array([100, 105, 95, 102, 98, 101, 97, 150])  # Last value is outlier
y_pred_example = np.array([98, 103, 97, 100, 99, 99, 95, 105])   # Prediction for outlier is off

mse, rmse, mae = compare_regression_metrics(y_true_example, y_pred_example)

# Visualize the impact
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
errors = y_true_example - y_pred_example
plt.bar(range(len(errors)), errors, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Error (True - Predicted)')
plt.title('Prediction Errors')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
squared_errors = errors ** 2
absolute_errors = np.abs(errors)

plt.bar(np.arange(len(errors)) - 0.2, squared_errors, width=0.4, 
        label='Squared Errors', alpha=0.7)
plt.bar(np.arange(len(errors)) + 0.2, absolute_errors, width=0.4, 
        label='Absolute Errors', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Error Magnitude')
plt.title('Squared vs Absolute Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### R-Squared: The Variance Explained 📊

```python
def explain_r_squared(y_true, y_pred):
    """
    Explain R-squared with visual intuition
    """
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # Baseline prediction (just the mean)
    y_mean_pred = np.full_like(y_true, np.mean(y_true))
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.7, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Perfect Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Model Predictions\n(R² = {r_squared:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals from model
    plt.subplot(1, 3, 2)
    model_residuals = y_true - y_pred
    plt.scatter(range(len(model_residuals)), model_residuals, alpha=0.7, c='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'Model Residuals\nSS_res = {ss_res:.2f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals from mean baseline
    plt.subplot(1, 3, 3)
    mean_residuals = y_true - y_mean_pred
    plt.scatter(range(len(mean_residuals)), mean_residuals, alpha=0.7, c='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals from Mean')
    plt.title(f'Baseline Residuals\nSS_tot = {ss_tot:.2f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 R-SQUARED EXPLANATION:")
    print(f"   R² = 1 - (SS_res / SS_tot) = 1 - ({ss_res:.2f} / {ss_tot:.2f}) = {r_squared:.3f}")
    print(f"   → Your model explains {r_squared:.1%} of the variance in the data")
    print(f"   → Baseline (just predicting mean) explains 0%")
    print(f"   → Perfect model would explain 100%")
    
    return r_squared

# Test with house price predictions
r2_explained = explain_r_squared(y_test, y_pred)
```

## Hyperparameter Tuning: Finding the Sweet Spot 🎯

### Grid Search vs Random Search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import time

# Create parameter grids
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search (exhaustive)
start_time = time.time()
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

# Random search (samples randomly)
start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"⏱️ TIMING COMPARISON:")
print(f"Grid Search: {grid_time:.1f} seconds")
print(f"Random Search: {random_time:.1f} seconds")
print(f"Speedup: {grid_time/random_time:.1f}x faster")

print(f"\n🎯 PERFORMANCE COMPARISON:")
print(f"Grid Search Best Score: {grid_search.best_score_:.3f}")
print(f"Random Search Best Score: {random_search.best_score_:.3f}")

print(f"\n⚙️ BEST PARAMETERS:")
print("Grid Search:", grid_search.best_params_)
print("Random Search:", random_search.best_params_)
```

### Validation Curves: Understanding Parameter Effects

```python
from sklearn.model_selection import validation_curve

# Study the effect of n_estimators
estimator_range = [10, 25, 50, 75, 100, 150, 200]

train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=42), 
    X_train, y_train,
    param_name='n_estimators', 
    param_range=estimator_range,
    cv=5, scoring='r2'
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(estimator_range, train_scores.mean(axis=1), 'o-', label='Training Score')
plt.plot(estimator_range, val_scores.mean(axis=1), 'o-', label='Validation Score')
plt.fill_between(estimator_range, 
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(estimator_range,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.title('Validation Curve: Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)

# Study the effect of max_depth
depth_range = [1, 3, 5, 7, 10, 15, None]

train_scores_depth, val_scores_depth = validation_curve(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, y_train,
    param_name='max_depth',
    param_range=depth_range[:-1],  # Exclude None for this visualization
    cv=5, scoring='r2'
)

plt.subplot(1, 2, 2)
plt.plot(depth_range[:-1], train_scores_depth.mean(axis=1), 'o-', label='Training Score')
plt.plot(depth_range[:-1], val_scores_depth.mean(axis=1), 'o-', label='Validation Score')
plt.xlabel('Max Depth')
plt.ylabel('R² Score')
plt.title('Validation Curve: Tree Depth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Curves: Do You Need More Data? 📈

Learning curves show how performance changes with training set size:

```python
from sklearn.model_selection import learning_curve

# Generate learning curves
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_lc.mean(axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, val_scores_lc.mean(axis=1), 'o-', label='Validation Score')

# Add confidence intervals
plt.fill_between(train_sizes,
                 train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
                 train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
                 val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
                 val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1), alpha=0.1)

plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves: Do We Need More Data?')
plt.legend()
plt.grid(True, alpha=0.3)

# Interpretation guide
gap_at_end = train_scores_lc.mean(axis=1)[-1] - val_scores_lc.mean(axis=1)[-1]
print(f"\n📊 LEARNING CURVE INTERPRETATION:")
print(f"Final gap between training and validation: {gap_at_end:.3f}")

if gap_at_end > 0.1:
    print("🔴 Large gap suggests OVERFITTING")
    print("   → Try: regularization, simpler model, more data")
elif val_scores_lc.mean(axis=1)[-1] < 0.7:
    print("🟡 Low validation score suggests UNDERFITTING") 
    print("   → Try: more complex model, feature engineering, less regularization")
else:
    print("🟢 Good balance between bias and variance!")

plt.show()
```

## Model Selection: Choosing the Best Algorithm 🏆

### Comprehensive Comparison Framework

```python
def comprehensive_model_comparison(X, y, problem_type='classification'):
    """
    Compare multiple models using proper evaluation
    """
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    if problem_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=42))
            ]),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(random_state=42))
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier())
            ])
        }
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        
    else:  # regression
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        models = {
            'Linear Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR())
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsRegressor())
            ])
        }
        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
        
        results[name] = {}
        for metric in scoring:
            scores = cv_results[f'test_{metric}']
            results[name][metric] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
    
    return results

# Run comprehensive comparison on house price data
print("🏠 COMPREHENSIVE MODEL COMPARISON: House Price Prediction")
regression_results = comprehensive_model_comparison(X_houses, y_houses, 'regression')

# Create results DataFrame for easy viewing
comparison_df = pd.DataFrame({
    model: {metric: f"{data['mean']:.3f} ± {data['std']:.3f}" 
           for metric, data in metrics.items()}
    for model, metrics in regression_results.items()
}).T

print(comparison_df)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
metric_names = ['R²', 'Negative MSE', 'Negative MAE']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    model_names = list(regression_results.keys())
    means = [regression_results[model][metric]['mean'] for model in model_names]
    stds = [regression_results[model][metric]['std'] for model in model_names]
    
    axes[i].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
    axes[i].set_ylabel(name)
    axes[i].set_title(f'Model Comparison: {name}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Statistical Significance Testing 📊

Don't just compare mean scores - test if differences are statistically significant:

```python
from scipy.stats import ttest_rel

def compare_models_statistically(model1, model2, X, y, cv_folds=10):
    """
    Test if one model is significantly better than another
    """
    # Get cross-validation scores for both models
    scores1 = cross_val_score(model1, X, y, cv=cv_folds)
    scores2 = cross_val_score(model2, X, y, cv=cv_folds)
    
    # Paired t-test
    t_stat, p_value = ttest_rel(scores1, scores2)
    
    print(f"Model 1 scores: {scores1.mean():.3f} ± {scores1.std():.3f}")
    print(f"Model 2 scores: {scores2.mean():.3f} ± {scores2.std():.3f}")
    print(f"Difference: {scores1.mean() - scores2.mean():.3f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    if p_value < 0.05:
        better_model = "Model 1" if scores1.mean() > scores2.mean() else "Model 2"
        print(f"✅ {better_model} is significantly better (p < 0.05)")
    else:
        print("❌ No significant difference between models")
    
    return scores1, scores2, p_value

# Compare Random Forest vs Linear Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

rf_scores, lr_scores, p_val = compare_models_statistically(
    rf_model, lr_model, X_houses, y_houses
)
```

## Avoiding Data Leakage: The Silent Killer 🚨

Data leakage is when information from the future "leaks" into your training data:

```python
# Example of data leakage (WRONG)
def wrong_way_preprocessing():
    """
    This is wrong! Don't do this!
    """
    # Scale entire dataset first
    scaler = StandardScaler()
    X_scaled_wrong = scaler.fit_transform(X)  # Leaked future information!
    
    # Then split
    X_train_wrong, X_test_wrong, y_train, y_test = train_test_split(
        X_scaled_wrong, y, test_size=0.2, random_state=42
    )
    
    return X_train_wrong, X_test_wrong

# Correct way to preprocess
def correct_way_preprocessing():
    """
    This is correct! Do this!
    """
    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale based only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
    X_test_scaled = scaler.transform(X_test)        # Transform test data using training stats
    
    return X_train_scaled, X_test_scaled

print("🚨 DATA LEAKAGE PREVENTION:")
print("Always split your data BEFORE any preprocessing!")
print("Test set should simulate truly unseen data")

# Demonstrate the impact
# Train models both ways and compare (spoiler: wrong way will look better)
```

## Advanced Evaluation Techniques 🚀

### 1. Nested Cross-Validation

For unbiased hyperparameter tuning evaluation:

```python
from sklearn.model_selection import cross_val_score

def nested_cross_validation(model, param_grid, X, y, outer_cv=5, inner_cv=3):
    """
    Proper way to evaluate hyperparameter tuning
    """
    outer_scores = []
    
    outer_kfold = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_kfold.split(X, y):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Evaluate best model on outer test set
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(score)
    
    return np.array(outer_scores)

# Example with smaller parameter grid
small_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}

nested_scores = nested_cross_validation(
    RandomForestClassifier(random_state=42),
    small_param_grid, X, y
)

print(f"Nested CV Score: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
print("This is the unbiased estimate of your tuned model's performance!")
```

### 2. Custom Scoring Functions

```python
from sklearn.metrics import make_scorer

def business_impact_score(y_true, y_pred):
    """
    Custom scoring function based on business impact
    Example: Cancer detection where false negatives are 10x worse than false positives
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Business costs
    cost_false_negative = 10000  # Missing cancer costs $10k
    cost_false_positive = 500    # False alarm costs $500
    
    total_cost = fn * cost_false_negative + fp * cost_false_positive
    max_possible_cost = len(y_true) * cost_false_negative
    
    # Convert to a score (higher is better)
    business_score = 1 - (total_cost / max_possible_cost)
    return business_score

# Create scorer
business_scorer = make_scorer(business_impact_score)

# Use in cross-validation
business_scores = cross_val_score(model_test, X, y, cv=5, scoring=business_scorer)
print(f"Business Impact Score: {business_scores.mean():.3f}")
```

## Model Evaluation Checklist ✅

Before trusting your model evaluation:

- [ ] **Split data properly**: Train/validation/test or cross-validation
- [ ] **No data leakage**: Preprocess after splitting
- [ ] **Stratify if needed**: Maintain class distributions in splits
- [ ] **Multiple metrics**: Don't rely on just accuracy/R²
- [ ] **Statistical significance**: Test if differences are meaningful
- [ ] **Business context**: Use metrics that matter to stakeholders
- [ ] **Temporal validation**: For time series, use proper time-based splits
- [ ] **Nested CV**: For hyperparameter tuning evaluation

## Common Evaluation Mistakes 🚫

### 1. Data Leakage Through Preprocessing
```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)  # Uses test data statistics!
X_train, X_test = train_test_split(X_scaled, ...)

# RIGHT  
X_train, X_test = train_test_split(X_all, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Using Accuracy for Imbalanced Data
```python
# With 99% negative class, always predicting negative gives 99% accuracy!
# Use precision, recall, F1, or AUC instead
```

### 3. Not Using Proper Cross-Validation for Time Series
```python
# WRONG for time series
cv_wrong = KFold(n_splits=5, shuffle=True)  # Shuffle breaks time order!

# RIGHT for time series
from sklearn.model_selection import TimeSeriesSplit
cv_right = TimeSeriesSplit(n_splits=5)  # Respects temporal order
```

## Key Takeaways 🎯

1. **Never test on training data** - it gives false confidence
2. **Cross-validation** provides more robust performance estimates
3. **Multiple metrics** give a complete picture of performance
4. **Precision vs Recall** trade-off depends on your problem's costs
5. **Statistical significance** testing prevents overinterpreting small differences
6. **Data leakage** can silently destroy your evaluation validity
7. **Business metrics** often matter more than statistical metrics

## Next Steps 🚀

1. **Practice**: Work through `../../notebooks/08_evaluation_lab.ipynb`
2. **Learn advanced topics**: Explore `../04_advanced_topics/`
3. **Apply to projects**: Use proper evaluation in your real work
4. **Read about specific metrics**: Deep dive into ROC, PR curves, and custom metrics

## Quick Exercise 💪

You're building a model to detect fraudulent credit card transactions:
- **Dataset**: 99.8% legitimate transactions, 0.2% fraudulent
- **Business cost**: Missing fraud costs $500, false alarm costs $5

Design an evaluation strategy that:
1. Properly handles the extreme class imbalance
2. Optimizes for business impact, not just accuracy
3. Provides confidence intervals for your estimates
4. Tests statistical significance vs a baseline model

*Detailed solution in the exercises folder!*
