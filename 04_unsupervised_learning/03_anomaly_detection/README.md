# 03 - Anomaly Detection: Finding the Unusual in the Usual

## 🕵️ What is Anomaly Detection?

Imagine you're a security guard watching a shopping mall. Most people follow predictable patterns - they walk at normal speed, visit 2-3 stores, stay for 1-2 hours. But occasionally, someone behaves differently - they run through the mall, visit 20 stores in 10 minutes, or loiter in one spot for hours.

**Anomaly detection is like being that security guard for data** - it automatically identifies patterns that don't fit the normal behavior.

## 🧠 Why Anomaly Detection Matters

### Real-World Impact Stories

**Credit Card Fraud (Bank of America)**:
- Normal: Small purchases near home
- Anomaly: Large purchase in foreign country at 3 AM
- Impact: Prevented $2.8 billion in fraud losses annually

**Network Security (Cloudflare)**:
- Normal: Regular web traffic patterns
- Anomaly: Sudden spike in requests from one IP
- Impact: Stopped DDoS attacks affecting millions of websites

**Manufacturing Quality (Toyota)**:
- Normal: Products within specification ranges
- Anomaly: Parts with unusual measurements
- Impact: Reduced defect rates by 85%, saved millions in recalls

**Healthcare Monitoring**:
- Normal: Stable vital signs patterns
- Anomaly: Sudden changes indicating medical emergency
- Impact: Early detection saves lives

## 🎯 Types of Anomalies

### 1. **Point Anomalies** 🎯
**Definition**: Individual data points that are unusual
**Think**: A 7-foot-tall basketball player in a kindergarten class
**Examples**:
- Fraudulent transaction: $10,000 purchase vs usual $50
- Medical outlier: Blood pressure 200/120 vs normal 120/80
- Website anomaly: 1000 page views vs usual 10

### 2. **Contextual Anomalies** 📅
**Definition**: Normal values that are unusual in specific contexts
**Think**: Wearing a winter coat (normal) in summer (unusual context)
**Examples**:
- High ice cream sales in December (unusual timing)
- Low website traffic on a Monday (unusual for that day)
- High electricity usage at 3 AM (unusual time)

### 3. **Collective Anomalies** 👥
**Definition**: Individual points are normal, but the pattern is unusual
**Think**: Each note in a song is normal, but together they're out of tune
**Examples**:
- Sequence of small transactions that together indicate fraud
- Network packets that individually look fine but together suggest an attack
- Stock trades that form an unusual pattern over time

## 📚 Anomaly Detection Approaches

### 1. **Statistical Methods** 📊
**Philosophy**: "If it's more than X standard deviations from normal, it's anomalous"

**Methods**:
- **Z-Score**: Distance from mean in standard deviations
- **Modified Z-Score**: More robust to outliers
- **Percentile-Based**: Flag top/bottom X% as anomalies

### 2. **Distance-Based Methods** 📏
**Philosophy**: "If it's far from most other points, it's anomalous"

**Methods**:
- **K-Nearest Neighbors**: Points with distant neighbors
- **Local Outlier Factor**: Density-based anomaly detection
- **Isolation Forest**: Random partitioning approach

### 3. **Model-Based Methods** 🤖
**Philosophy**: "If our model can't predict/reconstruct it well, it's anomalous"

**Methods**:
- **One-Class SVM**: Learn boundary around normal data
- **Autoencoders**: Neural networks for reconstruction
- **Gaussian Mixture Models**: Probabilistic approach

## 🚀 Learning Path

### Week 1: Statistical Methods (Days 1-2)
- **Day 1**: Z-score and basic statistical detection
- **Day 2**: Advanced statistical techniques

### Week 2: Machine Learning Methods (Days 3-4)
- **Day 3**: Isolation Forest and LOF
- **Day 4**: One-Class SVM and ensemble methods

### Week 3: Advanced Techniques (Days 5-6)
- **Day 5**: Autoencoders for anomaly detection
- **Day 6**: Time series and contextual anomalies

## 📁 Folder Structure

```
03_anomaly_detection/
├── statistical_methods/         # Z-score, percentiles, statistical tests
├── advanced_techniques/         # ML-based, autoencoders, ensembles
├── exercises/                   # Hands-on practice problems
├── projects/                    # Real-world applications
└── README.md                    # This file
```

## 🎯 Success Metrics

By the end of this module, you should be able to:

### Beginner Level:
- [ ] Explain different types of anomalies with examples
- [ ] Apply basic statistical methods for outlier detection
- [ ] Evaluate anomaly detection performance
- [ ] Handle class imbalance in anomaly detection

### Intermediate Level:
- [ ] Choose appropriate detection methods for different scenarios
- [ ] Implement ensemble anomaly detection systems
- [ ] Handle time series anomalies
- [ ] Build real-time anomaly detection pipelines

### Advanced Level:
- [ ] Design custom anomaly detection solutions
- [ ] Handle streaming data anomaly detection
- [ ] Optimize detection systems for business metrics
- [ ] Build explainable anomaly detection systems

## 🔍 Evaluation Challenges

### The Class Imbalance Problem

**Typical Scenario**: 99.9% normal data, 0.1% anomalies
- **Challenge**: Algorithm can get 99.9% accuracy by never detecting anomalies!
- **Solution**: Use specialized metrics and techniques

### Better Evaluation Metrics

#### 1. **Precision and Recall**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision: Of all points flagged as anomalies, how many were actually anomalies?
precision = precision_score(y_true, y_pred)

# Recall: Of all actual anomalies, how many did we catch?
recall = recall_score(y_true, y_pred)

# F1-Score: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f} (avoid false alarms)")
print(f"Recall: {recall:.3f} (catch real anomalies)")
print(f"F1-Score: {f1:.3f} (balanced measure)")
```

#### 2. **ROC and Precision-Recall Curves**
```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve (better for imbalanced data)
precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🛠 Practical Implementation Guide

### Step 1: Understand Your Data
```python
# Analyze normal vs anomaly characteristics
def data_profiling(data, labels):
    """Profile normal vs anomalous data"""
    normal_data = data[labels == 0]
    anomaly_data = data[labels == 1]
    
    print(f"Dataset composition:")
    print(f"  Normal points: {len(normal_data)} ({len(normal_data)/len(data)*100:.1f}%)")
    print(f"  Anomalous points: {len(anomaly_data)} ({len(anomaly_data)/len(data)*100:.1f}%)")
    
    # Statistical comparison
    for i, feature in enumerate(data.columns):
        normal_mean = normal_data[feature].mean()
        anomaly_mean = anomaly_data[feature].mean()
        
        print(f"\n{feature}:")
        print(f"  Normal mean: {normal_mean:.2f}")
        print(f"  Anomaly mean: {anomaly_mean:.2f}")
        print(f"  Difference: {abs(normal_mean - anomaly_mean):.2f}")
```

### Step 2: Choose Detection Strategy
```python
def choose_anomaly_method(data_characteristics):
    """Guide for choosing anomaly detection method"""
    
    if data_characteristics['size'] < 1000:
        return "Statistical methods (Z-score, percentiles)"
    
    elif data_characteristics['shape'] == 'irregular':
        return "Local Outlier Factor or DBSCAN"
    
    elif data_characteristics['dimensions'] > 10:
        return "Isolation Forest or Autoencoders"
    
    elif data_characteristics['has_time_component']:
        return "LSTM Autoencoders or time series methods"
    
    else:
        return "Start with Isolation Forest (general purpose)"
```

### Step 3: Implement Multiple Methods
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np

def ensemble_anomaly_detection(data):
    """Combine multiple detection methods"""
    
    # Method 1: Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_predictions = iso_forest.fit_predict(data)
    iso_scores = iso_forest.score_samples(data)
    
    # Method 2: Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof_predictions = lof.fit_predict(data)
    lof_scores = lof.negative_outlier_factor_
    
    # Method 3: One-Class SVM
    oc_svm = OneClassSVM(nu=0.1)
    svm_predictions = oc_svm.fit_predict(data)
    svm_scores = oc_svm.score_samples(data)
    
    # Combine results (majority vote)
    predictions = np.column_stack([iso_predictions, lof_predictions, svm_predictions])
    ensemble_predictions = np.array([
        -1 if sum(row == -1) >= 2 else 1 for row in predictions
    ])
    
    # Combine scores (average)
    # Normalize scores to [0, 1] range first
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_scores_norm = (-lof_scores - (-lof_scores).min()) / ((-lof_scores).max() - (-lof_scores).min())
    svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
    
    ensemble_scores = (iso_scores_norm + lof_scores_norm + svm_scores_norm) / 3
    
    return ensemble_predictions, ensemble_scores
```

## 🚀 Real-World Example: Credit Card Fraud Detection

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Simulate credit card transaction data
np.random.seed(42)
n_transactions = 10000

# Normal transactions
normal_transactions = {
    'amount': np.random.lognormal(3, 1, int(n_transactions * 0.995)),  # Log-normal distribution
    'time_since_last': np.random.exponential(24, int(n_transactions * 0.995)),  # Hours
    'merchant_category': np.random.choice([1, 2, 3, 4, 5], int(n_transactions * 0.995)),
    'location_risk_score': np.random.beta(2, 8, int(n_transactions * 0.995)),  # Low risk
}

# Fraudulent transactions (anomalies)
fraud_transactions = {
    'amount': np.random.lognormal(6, 1.5, int(n_transactions * 0.005)),  # Larger amounts
    'time_since_last': np.random.exponential(1, int(n_transactions * 0.005)),  # Rapid succession  
    'merchant_category': np.random.choice([6, 7, 8], int(n_transactions * 0.005)),  # Risky categories
    'location_risk_score': np.random.beta(8, 2, int(n_transactions * 0.005)),  # High risk
}

# Combine data
transactions = pd.DataFrame({
    'amount': np.concatenate([normal_transactions['amount'], fraud_transactions['amount']]),
    'time_since_last': np.concatenate([normal_transactions['time_since_last'], 
                                     fraud_transactions['time_since_last']]),
    'merchant_category': np.concatenate([normal_transactions['merchant_category'], 
                                       fraud_transactions['merchant_category']]),
    'location_risk_score': np.concatenate([normal_transactions['location_risk_score'], 
                                         fraud_transactions['location_risk_score']])
})

# Create true labels (1 = fraud, 0 = normal)
true_labels = np.concatenate([
    np.zeros(len(normal_transactions['amount'])),
    np.ones(len(fraud_transactions['amount']))
])

print(f"Dataset: {len(transactions)} transactions")
print(f"Fraud rate: {true_labels.mean():.3%}")

# Visualize data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Amount distribution
axes[0, 0].hist(transactions[true_labels == 0]['amount'], bins=50, alpha=0.7, label='Normal', density=True)
axes[0, 0].hist(transactions[true_labels == 1]['amount'], bins=50, alpha=0.7, label='Fraud', density=True)
axes[0, 0].set_xlabel('Transaction Amount')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Transaction Amount Distribution')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')

# Time patterns
axes[0, 1].hist(transactions[true_labels == 0]['time_since_last'], bins=50, alpha=0.7, label='Normal', density=True)
axes[0, 1].hist(transactions[true_labels == 1]['time_since_last'], bins=50, alpha=0.7, label='Fraud', density=True)
axes[0, 1].set_xlabel('Hours Since Last Transaction')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Transaction Timing Distribution')
axes[0, 1].legend()

# Location risk
axes[1, 0].hist(transactions[true_labels == 0]['location_risk_score'], bins=30, alpha=0.7, label='Normal', density=True)
axes[1, 0].hist(transactions[true_labels == 1]['location_risk_score'], bins=30, alpha=0.7, label='Fraud', density=True)
axes[1, 0].set_xlabel('Location Risk Score')
axes[1, 0].set_ylabel('Density') 
axes[1, 0].set_title('Location Risk Distribution')
axes[1, 0].legend()

# Merchant categories
normal_merchants = transactions[true_labels == 0]['merchant_category'].value_counts()
fraud_merchants = transactions[true_labels == 1]['merchant_category'].value_counts()

x = np.arange(len(normal_merchants))
width = 0.35

axes[1, 1].bar(x - width/2, normal_merchants.values, width, label='Normal', alpha=0.7)
axes[1, 1].bar(x + width/2, fraud_merchants.values, width, label='Fraud', alpha=0.7)
axes[1, 1].set_xlabel('Merchant Category')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Merchant Category Distribution')
axes[1, 1].legend()
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(normal_merchants.index)

plt.tight_layout()
plt.show()
```

### Apply Isolation Forest

```python
# Prepare data
scaler = StandardScaler()
transactions_scaled = scaler.fit_transform(transactions)

# Apply Isolation Forest
iso_forest = IsolationForest(
    contamination=0.005,  # Expected fraud rate
    random_state=42,
    n_estimators=100
)

# Fit and predict
anomaly_predictions = iso_forest.fit_predict(transactions_scaled)
anomaly_scores = iso_forest.score_samples(transactions_scaled)

# Convert predictions (-1 = anomaly, 1 = normal) to (1 = anomaly, 0 = normal)
predicted_labels = (anomaly_predictions == -1).astype(int)

# Evaluate performance
from sklearn.metrics import classification_report, confusion_matrix

print("Fraud Detection Performance:")
print(classification_report(true_labels, predicted_labels, 
                          target_names=['Normal', 'Fraud']))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Fraud Detection')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Fraud'])
plt.yticks(tick_marks, ['Normal', 'Fraud'])

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Business impact analysis
total_fraud_value = transactions[true_labels == 1]['amount'].sum()
detected_fraud_indices = np.where((true_labels == 1) & (predicted_labels == 1))[0]
detected_fraud_value = transactions.iloc[detected_fraud_indices]['amount'].sum()

print(f"\nBusiness Impact:")
print(f"Total fraud value: ${total_fraud_value:,.2f}")
print(f"Detected fraud value: ${detected_fraud_value:,.2f}")
print(f"Fraud detection rate: {detected_fraud_value/total_fraud_value:.1%}")

false_alarms = np.sum((true_labels == 0) & (predicted_labels == 1))
print(f"False alarms: {false_alarms} out of {np.sum(true_labels == 0)} normal transactions")
print(f"False alarm rate: {false_alarms/np.sum(true_labels == 0):.3%}")
```

## 🎨 Anomaly Detection Methods Deep Dive

### 1. Statistical Methods (Start Here!)

#### Z-Score Method
```python
def zscore_anomaly_detection(data, threshold=3):
    """Simple Z-score based anomaly detection"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    
    anomalies = z_scores > threshold
    return anomalies, z_scores

# Example: Detect unusual transaction amounts
amounts = transactions['amount']
anomalies, scores = zscore_anomaly_detection(amounts, threshold=2.5)

print(f"Detected {np.sum(anomalies)} anomalies out of {len(amounts)} transactions")
print(f"Anomaly rate: {np.sum(anomalies)/len(amounts):.3%}")

# Visualize
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(amounts, bins=50, alpha=0.7, label='All transactions')
plt.hist(amounts[anomalies], bins=20, alpha=0.8, label='Anomalies', color='red')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Transaction Amounts with Z-Score Anomalies')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.scatter(range(len(scores)), scores, alpha=0.6, c=anomalies, cmap='coolwarm')
plt.axhline(y=2.5, color='red', linestyle='--', label='Threshold')
plt.xlabel('Transaction Index')
plt.ylabel('Z-Score')
plt.title('Z-Scores for All Transactions')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Modified Z-Score (More Robust)
```python
def modified_zscore_anomaly_detection(data, threshold=3.5):
    """Modified Z-score using median and MAD (more robust to outliers)"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    
    # Modified Z-score
    modified_z_scores = 0.6745 * (data - median) / mad
    anomalies = np.abs(modified_z_scores) > threshold
    
    return anomalies, modified_z_scores

# Compare regular vs modified Z-score
anomalies_z, scores_z = zscore_anomaly_detection(amounts)
anomalies_mod_z, scores_mod_z = modified_zscore_anomaly_detection(amounts)

print(f"Regular Z-score detected: {np.sum(anomalies_z)} anomalies")
print(f"Modified Z-score detected: {np.sum(anomalies_mod_z)} anomalies")
```

### 2. Machine Learning Methods

#### Isolation Forest Explained
```python
def isolation_forest_explanation():
    """
    Isolation Forest Logic:
    1. Randomly select a feature
    2. Randomly select a split value for that feature
    3. Split data into two groups
    4. Repeat recursively until each point is isolated
    
    Key Insight: Anomalies get isolated with fewer splits!
    Normal points are buried deep in the tree.
    """
    
    # Visualize the concept
    from sklearn.ensemble import IsolationForest
    
    # Simple 2D example
    normal_points = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)
    anomaly_points = np.array([[4, 4], [-4, -4], [4, -4]])
    
    data = np.vstack([normal_points, anomaly_points])
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(data)
    scores = iso_forest.score_samples(data)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    normal_mask = predictions == 1
    anomaly_mask = predictions == -1
    
    plt.scatter(data[normal_mask, 0], data[normal_mask, 1], c='blue', label='Normal', alpha=0.6)
    plt.scatter(data[anomaly_mask, 0], data[anomaly_mask, 1], c='red', label='Anomaly', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Isolation Forest Results')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=scores, cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Anomaly Scores (Red = More Anomalous)')
    
    plt.tight_layout()
    plt.show()
    
    return predictions, scores

iso_predictions, iso_scores = isolation_forest_explanation()
```

#### Local Outlier Factor (LOF)
```python
def lof_explanation():
    """
    Local Outlier Factor Logic:
    1. For each point, find its k nearest neighbors
    2. Calculate local density around the point
    3. Compare with density of neighbors
    4. Points in sparse regions relative to neighbors = outliers
    
    Key Insight: Considers local neighborhood, not global patterns
    """
    
    from sklearn.neighbors import LocalOutlierFactor
    
    # Create data with varying densities
    cluster1 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 100)
    cluster2 = np.random.multivariate_normal([4, 4], [[0.2, 0], [0, 0.2]], 50)
    outliers = np.array([[2, 2], [6, 0], [0, 6]])
    
    data = np.vstack([cluster1, cluster2, outliers])
    
    # Apply LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions = lof.fit_predict(data)
    scores = lof.negative_outlier_factor_
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    normal_mask = predictions == 1
    anomaly_mask = predictions == -1
    
    plt.scatter(data[normal_mask, 0], data[normal_mask, 1], c='blue', label='Normal', alpha=0.6)
    plt.scatter(data[anomaly_mask, 0], data[anomaly_mask, 1], c='red', label='Anomaly', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Local Outlier Factor Results')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=-scores, cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='LOF Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('LOF Scores (Red = More Anomalous)')
    
    plt.tight_layout()
    plt.show()
    
    return predictions, scores

lof_predictions, lof_scores = lof_explanation()
```

## 🎯 Choosing the Right Method

### Decision Guide

```python
def anomaly_detection_decision_tree(data_info):
    """Help choose the right anomaly detection method"""
    
    print("Anomaly Detection Method Selection:")
    print("=====================================")
    
    if data_info['size'] < 1000:
        print("→ Small dataset: Use statistical methods (Z-score, percentiles)")
        
    elif data_info['interpretability'] == 'high':
        print("→ Need interpretability: Use statistical methods or One-Class SVM")
        
    elif data_info['real_time'] == True:
        print("→ Real-time detection: Use simple statistical methods or pre-trained models")
        
    elif data_info['dimensions'] > 50:
        print("→ High dimensions: Use Isolation Forest or Autoencoders")
        
    elif data_info['has_labels'] == True:
        print("→ Some labeled anomalies: Use supervised methods or semi-supervised")
        
    elif data_info['cluster_structure'] == 'complex':
        print("→ Complex patterns: Use Local Outlier Factor or ensemble methods")
        
    else:
        print("→ General purpose: Start with Isolation Forest")
    
    print("\nRecommended ensemble: Combine 2-3 methods for robustness")

# Example usage
data_characteristics = {
    'size': 10000,
    'dimensions': 20,
    'interpretability': 'medium',
    'real_time': False,
    'has_labels': False,
    'cluster_structure': 'moderate'
}

anomaly_detection_decision_tree(data_characteristics)
```

### Performance Comparison
```python
# Compare multiple methods on the same data
methods = {
    'Isolation Forest': IsolationForest(contamination=0.005, random_state=42),
    'Local Outlier Factor': LocalOutlierFactor(contamination=0.005),
    'One-Class SVM': OneClassSVM(nu=0.005)
}

results = {}

for name, method in methods.items():
    predictions = method.fit_predict(transactions_scaled)
    predicted_labels = (predictions == -1).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    results[name] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Display comparison
comparison_df = pd.DataFrame(results).T
print("Method Comparison:")
print(comparison_df.round(3))

# Visualize comparison
comparison_df.plot(kind='bar', figsize=(12, 6))
plt.title('Anomaly Detection Method Comparison')
plt.ylabel('Score')
plt.legend(['Precision', 'Recall', 'F1-Score'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 🧪 Hands-On Exercises

### Exercise 1: Website Anomaly Detection
```python
# Simulate website traffic data
website_data = {
    'page_views': np.random.poisson(100, 1000),
    'session_duration': np.random.exponential(5, 1000),  # minutes
    'bounce_rate': np.random.beta(3, 7, 1000),  # Skewed toward low values
    'pages_per_session': np.random.gamma(2, 1, 1000),
    'conversion_rate': np.random.beta(1, 20, 1000)  # Low conversion typical
}

# Your tasks:
# 1. Apply different anomaly detection methods
# 2. Identify what makes certain sessions anomalous
# 3. Determine if anomalies are good (viral content) or bad (bots)
# 4. Set up alerts for different types of anomalies
# 5. Create business rules based on findings
```

### Exercise 2: IoT Sensor Anomaly Detection
```python
# Simulate IoT sensor data from a manufacturing plant
sensor_data = {
    'temperature': np.random.normal(75, 5, 2000),  # Fahrenheit
    'humidity': np.random.normal(60, 10, 2000),    # Percentage
    'pressure': np.random.normal(14.7, 0.5, 2000), # PSI
    'vibration': np.random.exponential(2, 2000),   # Amplitude
    'power_consumption': np.random.gamma(3, 10, 2000)  # Watts
}

# Add some equipment failures (anomalies)
failure_indices = np.random.choice(2000, 20, replace=False)
sensor_data['temperature'][failure_indices] += np.random.normal(20, 5, 20)  # Overheating
sensor_data['vibration'][failure_indices] *= np.random.uniform(3, 8, 20)    # High vibration

# Your tasks:
# 1. Detect equipment failure patterns
# 2. Identify which sensors are most predictive of failures
# 3. Set up early warning thresholds
# 4. Calculate cost savings from early detection
# 5. Design a real-time monitoring dashboard
```

## 💰 Business Value of Anomaly Detection

### Cost-Benefit Analysis

```python
def calculate_business_impact(detection_results, transaction_values):
    """Calculate financial impact of anomaly detection"""
    
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))
    
    # Financial calculations
    avg_fraud_value = transaction_values[true_labels == 1].mean()
    avg_normal_value = transaction_values[true_labels == 0].mean()
    
    # Benefits
    fraud_prevented = true_positives * avg_fraud_value
    
    # Costs
    investigation_cost_per_alert = 50  # Cost to investigate each alert
    customer_friction_cost = 25       # Cost of blocking legitimate transaction
    
    investigation_costs = (true_positives + false_positives) * investigation_cost_per_alert
    friction_costs = false_positives * customer_friction_cost
    fraud_losses = false_negatives * avg_fraud_value
    
    total_benefits = fraud_prevented
    total_costs = investigation_costs + friction_costs + fraud_losses
    net_benefit = total_benefits - total_costs
    
    print("Business Impact Analysis:")
    print("=" * 40)
    print(f"Fraud prevented: ${fraud_prevented:,.2f}")
    print(f"Investigation costs: ${investigation_costs:,.2f}")
    print(f"Customer friction costs: ${friction_costs:,.2f}")
    print(f"Missed fraud losses: ${fraud_losses:,.2f}")
    print(f"Net benefit: ${net_benefit:,.2f}")
    print(f"ROI: {net_benefit/total_costs*100:.1f}%")
    
    return {
        'fraud_prevented': fraud_prevented,
        'total_costs': total_costs,
        'net_benefit': net_benefit,
        'roi': net_benefit/total_costs
    }

# Calculate for our fraud detection example
business_impact = calculate_business_impact(predicted_labels, transactions['amount'])
```

## 🏆 Best Practices

### 1. **Start Simple, Then Add Complexity**
```python
# Step 1: Try simple statistical methods first
simple_anomalies = zscore_anomaly_detection(data)

# Step 2: Add machine learning if needed
ml_anomalies = isolation_forest_detection(data)

# Step 3: Combine methods for robustness
ensemble_anomalies = combine_methods([simple_anomalies, ml_anomalies])
```

### 2. **Understand Your Domain**
```python
# Different domains have different anomaly characteristics
domain_considerations = {
    'finance': 'Time patterns matter, false positives are costly',
    'cybersecurity': 'Speed is critical, some false positives acceptable',
    'manufacturing': 'Seasonal patterns, equipment lifecycle important',
    'healthcare': 'Patient safety critical, explainability required'
}
```

### 3. **Monitor and Update**
```python
# Anomaly patterns change over time
def monitor_anomaly_detection(model, new_data, time_window='weekly'):
    """Monitor detection performance over time"""
    
    # Track key metrics
    metrics = {
        'anomaly_rate': [],
        'average_score': [],
        'score_variance': []
    }
    
    # Calculate for current time window
    predictions = model.predict(new_data)
    scores = model.score_samples(new_data)
    
    current_anomaly_rate = (predictions == -1).mean()
    current_avg_score = scores.mean()
    current_score_variance = scores.var()
    
    # Alert if patterns change significantly
    if current_anomaly_rate > historical_rate * 2:
        print("⚠️ ALERT: Anomaly rate significantly increased!")
        print("Consider retraining model or investigating data changes")
    
    return current_anomaly_rate, current_avg_score
```

## 💭 Reflection Questions

1. How would you explain the difference between outliers and anomalies to a business stakeholder?

2. Why is precision vs recall trade-off particularly important in fraud detection?

3. What are the ethical considerations when building anomaly detection systems for human behavior?

4. How would you convince a skeptical manager that anomaly detection is worth the investment?

## 🚀 Next Steps

Excellent work mastering anomaly detection! You now understand:
- Different types of anomalies and when they occur
- Multiple detection approaches and their trade-offs
- How to evaluate and optimize detection systems
- Real-world business applications and impact

**Coming Next**: Association Rules and Recommendation Systems - discover relationships and patterns in transactional data!

Remember: Anomaly detection is both an art and a science. The key is understanding your domain, choosing appropriate methods, and continuously monitoring and improving your systems.
