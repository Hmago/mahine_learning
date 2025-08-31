# K-Nearest Neighbors: Learning from Your Neighbors 🏘️

## What is K-Nearest Neighbors (KNN)? 🤔

Imagine you just moved to a new neighborhood and want to know if you'll like living there. What would you do? You'd probably talk to your **nearest neighbors** and see what they think!

KNN works exactly the same way:
1. **Find the K closest examples** to your new data point
2. **Ask them what class they belong to**
3. **Go with the majority vote**

It's that simple! No complex math, no training phase - just pure common sense.

## The Intuitive Example: Restaurant Recommendations 🍕

Let's say you want to predict if someone will like a restaurant based on:
- **Price level** (1-5 scale)
- **Noise level** (1-5 scale)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Sample restaurant data
restaurants = np.array([
    [1, 2],  # Cheap, quiet
    [1, 3],  # Cheap, moderate noise  
    [2, 2],  # Moderate price, quiet
    [4, 4],  # Expensive, loud
    [5, 5],  # Very expensive, very loud
    [4, 3],  # Expensive, moderate noise
    [2, 1],  # Moderate price, very quiet
    [3, 2]   # Moderate-high price, quiet
])

# Preferences: 0 = dislike, 1 = like
preferences = np.array([1, 1, 1, 0, 0, 0, 1, 1])

# New restaurant to evaluate
new_restaurant = np.array([[2, 3]])  # Moderate price, moderate noise

# Visualize the data
plt.figure(figsize=(10, 8))
colors = ['red' if pref == 0 else 'green' for pref in preferences]
labels = ['Dislike' if pref == 0 else 'Like' for pref in preferences]

plt.scatter(restaurants[:, 0], restaurants[:, 1], c=colors, s=100, alpha=0.7)
plt.scatter(new_restaurant[:, 0], new_restaurant[:, 1], c='blue', s=200, marker='*', label='New Restaurant')

# Add labels
for i, (x, y) in enumerate(restaurants):
    plt.annotate(f'R{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Price Level (1-5)')
plt.ylabel('Noise Level (1-5)')
plt.title('Restaurant Recommendation using KNN')
plt.legend()
plt.grid(True, alpha=0.3)

# Add custom legend for colors
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Dislike')
green_patch = mpatches.Patch(color='green', label='Like')
blue_patch = mpatches.Patch(color='blue', label='New Restaurant')
plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.show()

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(restaurants, preferences)

prediction = knn.predict(new_restaurant)
probabilities = knn.predict_proba(new_restaurant)

print(f"Prediction: {'Like' if prediction[0] == 1 else 'Dislike'}")
print(f"Confidence: {probabilities[0].max():.2f}")
```

## Understanding Distance: How "Close" is Close? 📏

The magic of KNN is in measuring **distance**. There are different ways to measure how similar two things are:

### 1. Euclidean Distance (Most Common)
Think of it as "as the crow flies" distance:

```python
def euclidean_distance(point1, point2):
    """
    Calculate straight-line distance between two points
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Example
person_a = [25, 50000]  # Age 25, Income $50k
person_b = [30, 55000]  # Age 30, Income $55k
person_c = [60, 40000]  # Age 60, Income $40k

distance_ab = euclidean_distance(np.array(person_a), np.array(person_b))
distance_ac = euclidean_distance(np.array(person_a), np.array(person_c))

print(f"Distance from A to B: {distance_ab:.2f}")
print(f"Distance from A to C: {distance_ac:.2f}")
print(f"Person A is closer to Person B" if distance_ab < distance_ac else "Person A is closer to Person C")
```

### 2. Manhattan Distance (City Block)
Think of it as walking distance in a city with grid streets:

```python
def manhattan_distance(point1, point2):
    """
    Calculate city-block distance (sum of absolute differences)
    """
    return np.sum(np.abs(point1 - point2))

# Compare with Euclidean
print(f"Euclidean distance A to B: {euclidean_distance(np.array(person_a), np.array(person_b)):.2f}")
print(f"Manhattan distance A to B: {manhattan_distance(np.array(person_a), np.array(person_b)):.2f}")
```

### 3. Custom Distance Metrics
```python
def weighted_distance(point1, point2, weights):
    """
    Give different importance to different features
    """
    diff = point1 - point2
    return np.sqrt(np.sum(weights * (diff ** 2)))

# Example: Age matters 3x more than income
weights = np.array([3, 0.00001])  # Age weight=3, Income weight=very small
weighted_dist = weighted_distance(np.array(person_a), np.array(person_b), weights)
print(f"Weighted distance: {weighted_dist:.2f}")
```

## Choosing the Right K Value 🎯

The **K** in KNN is crucial - it's the number of neighbors to consider:

```python
from sklearn.model_selection import validation_curve

# Test different K values
k_range = range(1, 31)
train_scores, val_scores = validation_curve(
    KNeighborsClassifier(), X, y,
    param_name='n_neighbors', param_range=k_range,
    cv=5, scoring='accuracy'
)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(k_range, train_scores.mean(axis=1), 'o-', label='Training Accuracy')
plt.plot(k_range, val_scores.mean(axis=1), 'o-', label='Validation Accuracy')

# Add error bars
plt.fill_between(k_range, 
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1),
                 alpha=0.1)
plt.fill_between(k_range,
                 val_scores.mean(axis=1) - val_scores.std(axis=1), 
                 val_scores.mean(axis=1) + val_scores.std(axis=1),
                 alpha=0.1)

plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Finding the Optimal K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find best K
best_k = k_range[np.argmax(val_scores.mean(axis=1))]
print(f"Best K value: {best_k}")
```

### K Value Guidelines

- **K = 1**: Very sensitive to noise, might overfit
- **K = 3-7**: Good starting range for small datasets
- **K = sqrt(n_samples)**: Common rule of thumb
- **K = n_samples**: Essentially predicts the majority class always

```python
# Rule of thumb implementation
def suggest_k_value(n_samples):
    """Suggest K based on dataset size"""
    if n_samples < 50:
        return 3
    elif n_samples < 200:
        return 5
    elif n_samples < 1000:
        return int(np.sqrt(n_samples))
    else:
        return int(np.sqrt(n_samples) / 2)

print(f"For {len(X)} samples, suggested K: {suggest_k_value(len(X))}")
```

## Real-World Example: Movie Recommendation System 🎬

```python
import pandas as pd

# Sample movie preference data
movie_data = {
    'User_Age': [25, 35, 45, 22, 50, 28, 40, 33, 55, 26],
    'Favorite_Genre_Code': [1, 2, 3, 1, 3, 2, 3, 1, 2, 1],  # 1=Action, 2=Comedy, 3=Drama
    'Avg_Rating_Given': [3.5, 4.2, 4.8, 3.8, 4.5, 3.9, 4.1, 3.7, 4.6, 3.6],
    'Movies_Per_Month': [8, 12, 4, 15, 3, 10, 5, 9, 2, 11],
    'Likes_SciFi': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  # Target: will they like a sci-fi movie?
}

movie_df = pd.DataFrame(movie_data)
X_movies = movie_df.drop('Likes_SciFi', axis=1)
y_movies = movie_df['Likes_SciFi']

# Train KNN
knn_movies = KNeighborsClassifier(n_neighbors=3)
knn_movies.fit(X_movies, y_movies)

# New user recommendation
new_user = [[30, 1, 4.0, 7]]  # Age 30, likes Action, rates 4.0 avg, watches 7/month
recommendation = knn_movies.predict(new_user)
confidence = knn_movies.predict_proba(new_user)

print(f"Will this user like sci-fi? {'Yes' if recommendation[0] == 1 else 'No'}")
print(f"Confidence: {confidence[0].max():.2f}")

# Let's see which neighbors influenced this decision
distances, indices = knn_movies.kneighbors(new_user)
print(f"\nClosest neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor_data = X_movies.iloc[idx]
    neighbor_label = y_movies.iloc[idx]
    print(f"Neighbor {i+1}: Distance={dist:.2f}, Age={neighbor_data['User_Age']}, "
          f"Genre={neighbor_data['Favorite_Genre_Code']}, Likes_SciFi={'Yes' if neighbor_label == 1 else 'No'}")
```

## The Curse of Dimensionality 😰

As the number of features increases, KNN becomes less effective. Here's why:

```python
# Demonstrate curse of dimensionality
from sklearn.datasets import make_classification

def test_knn_dimensions(n_features_list, n_samples=500):
    """Test KNN performance as dimensions increase"""
    accuracies = []
    
    for n_features in n_features_list:
        # Create dataset with varying number of features
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_informative=min(5, n_features),
                                 n_redundant=0, random_state=42)
        
        # Test KNN performance
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X, y, cv=5)
        accuracies.append(scores.mean())
        
        print(f"Features: {n_features:3d}, Accuracy: {scores.mean():.3f}")
    
    return accuracies

# Test with different numbers of features
feature_counts = [2, 5, 10, 20, 50, 100]
accuracies = test_knn_dimensions(feature_counts)

plt.figure(figsize=(10, 6))
plt.plot(feature_counts, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('KNN Accuracy')
plt.title('The Curse of Dimensionality in KNN')
plt.grid(True, alpha=0.3)
plt.show()
```

**Why this happens:**
- In high dimensions, all points become roughly equidistant
- The concept of "nearest" becomes meaningless
- Noise starts to dominate signal

## Feature Scaling: Critical for KNN! ⚖️

KNN is **extremely sensitive** to feature scales. Here's why:

```python
# Example showing why scaling matters
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create data with different scales
sample_data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45],           # Scale: 20-80
    'Income': [30000, 50000, 70000, 90000, 120000],  # Scale: 20k-200k
    'Credit_Score': [650, 700, 750, 800, 850]        # Scale: 300-850
})

print("Original data:")
print(sample_data)

# Calculate distances without scaling
person1 = sample_data.iloc[0].values  # [25, 30000, 650]
person2 = sample_data.iloc[1].values  # [30, 50000, 700]

euclidean_dist = np.sqrt(np.sum((person1 - person2) ** 2))
print(f"\nDistance without scaling: {euclidean_dist:.2f}")
print("Income dominates the distance calculation!")

# Now with scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sample_data)

scaled_person1 = scaled_data[0]
scaled_person2 = scaled_data[1]

scaled_dist = np.sqrt(np.sum((scaled_person1 - scaled_person2) ** 2))
print(f"Distance with scaling: {scaled_dist:.2f}")
print("Now all features contribute equally!")

# Visualize the difference
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original data (only first 2 features for visualization)
axes[0].scatter(sample_data['Age'], sample_data['Income'], s=100)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Income')
axes[0].set_title('Original Data (Income Dominates Distance)')

# Scaled data
axes[1].scatter(scaled_data[:, 0], scaled_data[:, 1], s=100)
axes[1].set_xlabel('Age (scaled)')
axes[1].set_ylabel('Income (scaled)')
axes[1].set_title('Scaled Data (Features Equally Important)')

plt.tight_layout()
plt.show()
```

## KNN for Classification: Complete Example 🎯

Let's build a customer segmentation system:

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create customer data
np.random.seed(42)
X_customers, y_customers = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)

# Add feature names for clarity
feature_names = ['Annual_Spending', 'Visit_Frequency']
customer_df = pd.DataFrame(X_customers, columns=feature_names)
customer_df['Segment'] = y_customers  # 0=Low, 1=Medium, 2=High value

print("Customer segments:")
print(customer_df['Segment'].value_counts().sort_index())

# Prepare data
X = customer_df[feature_names]
y = customer_df['Segment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (CRUCIAL for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN with different K values
k_values = [1, 3, 5, 7, 9]
results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_acc = knn.score(X_train_scaled, y_train)
    test_acc = knn.score(X_test_scaled, y_test)
    
    results[k] = {'train': train_acc, 'test': test_acc}
    print(f"K={k}: Train={train_acc:.3f}, Test={test_acc:.3f}")

# Visualize results
k_vals = list(results.keys())
train_accs = [results[k]['train'] for k in k_vals]
test_accs = [results[k]['test'] for k in k_vals]

plt.figure(figsize=(10, 6))
plt.plot(k_vals, train_accs, 'o-', label='Training Accuracy')
plt.plot(k_vals, test_accs, 'o-', label='Test Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Choose best K and evaluate
best_k = k_vals[np.argmax(test_accs)]
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)

y_pred = final_knn.predict(X_test_scaled)

print(f"\nFinal model with K={best_k}:")
print(f"Test Accuracy: {final_knn.score(X_test_scaled, y_test):.3f}")
print("\nDetailed Results:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
```

## Different Distance Metrics in Action 📐

```python
from sklearn.neighbors import KNeighborsClassifier

# Create dataset where different distance metrics matter
X_mixed = np.array([
    [1, 10],    # Feature 1 varies little, Feature 2 varies a lot
    [2, 12],
    [1.5, 11],
    [8, 15],
    [9, 17],
    [8.5, 16]
])
y_mixed = np.array([0, 0, 0, 1, 1, 1])

# Test different distance metrics
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for metric in distance_metrics:
    if metric == 'minkowski':
        knn = KNeighborsClassifier(n_neighbors=3, metric=metric, p=3)
    else:
        knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    
    knn.fit(X_mixed, y_mixed)
    accuracy = knn.score(X_mixed, y_mixed)
    print(f"{metric.capitalize()} distance: {accuracy:.3f} accuracy")

# Visualize different distance concepts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

point_a = np.array([2, 3])
point_b = np.array([5, 7])

# Euclidean distance
axes[0].plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'r-', linewidth=2)
axes[0].scatter(*point_a, s=100, c='blue', label='Point A')
axes[0].scatter(*point_b, s=100, c='red', label='Point B')
axes[0].set_title('Euclidean (Straight Line)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Manhattan distance
axes[1].plot([point_a[0], point_b[0]], [point_a[1], point_a[1]], 'r-', linewidth=2)
axes[1].plot([point_b[0], point_b[0]], [point_a[1], point_b[1]], 'r-', linewidth=2)
axes[1].scatter(*point_a, s=100, c='blue', label='Point A')
axes[1].scatter(*point_b, s=100, c='red', label='Point B')
axes[1].set_title('Manhattan (City Block)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Chebyshev distance (maximum of absolute differences)
max_diff_x = abs(point_b[0] - point_a[0])
max_diff_y = abs(point_b[1] - point_a[1])
axes[2].add_patch(plt.Rectangle(point_a, max_diff_x, max_diff_y, fill=False, edgecolor='red', linewidth=2))
axes[2].scatter(*point_a, s=100, c='blue', label='Point A')
axes[2].scatter(*point_b, s=100, c='red', label='Point B')
axes[2].set_title('Chebyshev (Maximum Difference)')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()
```

## KNN for Regression 📊

KNN isn't just for classification! It can predict continuous values too:

```python
from sklearn.neighbors import KNeighborsRegressor

# House price prediction example
house_data = {
    'Size_SqFt': [1200, 1500, 1800, 2000, 2200, 1400, 1600, 1900],
    'Bedrooms': [2, 3, 3, 4, 4, 2, 3, 3],
    'Age_Years': [5, 10, 15, 8, 3, 12, 7, 6],
    'Price': [200000, 250000, 300000, 350000, 400000, 220000, 270000, 320000]
}

house_df = pd.DataFrame(house_data)
X_houses = house_df[['Size_SqFt', 'Bedrooms', 'Age_Years']]
y_houses = house_df['Price']

# Scale features
scaler = StandardScaler()
X_houses_scaled = scaler.fit_transform(X_houses)

# Train KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_houses_scaled, y_houses)

# Predict price for new house
new_house = [[1700, 3, 8]]  # 1700 sq ft, 3 bedrooms, 8 years old
new_house_scaled = scaler.transform(new_house)

predicted_price = knn_reg.predict(new_house_scaled)
print(f"Predicted house price: ${predicted_price[0]:,.0f}")

# Show which houses influenced the prediction
distances, indices = knn_reg.kneighbors(new_house_scaled)
print(f"\nNearest neighbor houses:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor = house_df.iloc[idx]
    print(f"House {i+1}: Size={neighbor['Size_SqFt']}, Beds={neighbor['Bedrooms']}, "
          f"Age={neighbor['Age_Years']}, Price=${neighbor['Price']:,}")
```

## Advantages & Disadvantages 📊

### ✅ Advantages

**Simple to Understand**: No complex math, intuitive concept
**No Training Period**: Just stores the data (lazy learning)
**Works with Any Number of Classes**: Natural multi-class classifier
**Non-parametric**: Makes no assumptions about data distribution
**Local Patterns**: Can capture local patterns in data
**Versatile**: Works for both classification and regression

### ❌ Disadvantages

**Computationally Expensive**: Must calculate distances to all training points
**Sensitive to Irrelevant Features**: All features affect distance equally
**Curse of Dimensionality**: Performance degrades with many features
**Sensitive to Scale**: Must normalize features
**Memory Intensive**: Stores entire training dataset
**Poor with Sparse Data**: Many features with mostly zeros

## Optimizing KNN Performance 🚀

### 1. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Reduce dimensions before applying KNN
pca = PCA(n_components=10)  # Keep top 10 components
X_reduced = pca.fit_transform(X_scaled)

knn_reduced = KNeighborsClassifier(n_neighbors=5)
scores_reduced = cross_val_score(knn_reduced, X_reduced, y, cv=5)
print(f"KNN with PCA: {scores_reduced.mean():.3f}")
```

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top K features
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

knn_selected = KNeighborsClassifier(n_neighbors=5)
scores_selected = cross_val_score(knn_selected, X_selected, y, cv=5)
print(f"KNN with feature selection: {scores_selected.mean():.3f}")
```

### 3. Distance Metric Selection
```python
# Test different distance metrics for your specific data
metrics_to_test = ['euclidean', 'manhattan', 'chebyshev']

for metric in metrics_to_test:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"{metric.capitalize()}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

## Advanced KNN Techniques 🧠

### 1. Weighted KNN
Instead of simple voting, weight neighbors by their distance:

```python
# Distance-weighted voting
knn_weighted = KNeighborsClassifier(
    n_neighbors=5, 
    weights='distance'  # Closer neighbors have more influence
)

knn_weighted.fit(X_train_scaled, y_train)
weighted_accuracy = knn_weighted.score(X_test_scaled, y_test)

# Compare with uniform weights
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X_train_scaled, y_train)
uniform_accuracy = knn_uniform.score(X_test_scaled, y_test)

print(f"Uniform weights: {uniform_accuracy:.3f}")
print(f"Distance weights: {weighted_accuracy:.3f}")
```

### 2. Radius-Based Neighbors
Sometimes it's better to consider all neighbors within a certain distance:

```python
from sklearn.neighbors import RadiusNeighborsClassifier

# Instead of K nearest, use all within radius
radius_knn = RadiusNeighborsClassifier(radius=1.0)
# Note: Requires careful tuning of radius parameter
```

### 3. Approximate Nearest Neighbors (for Speed)
```python
# For very large datasets, use approximate methods
from sklearn.neighbors import NearestNeighbors

# LSH (Locality Sensitive Hashing) for approximate neighbors
# Available in libraries like faiss, annoy, or nmslib
```

## Implementation from Scratch 🔨

Here's a simple KNN implementation to understand the core algorithm:

```python
class SimpleKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        """Store training data (lazy learning!)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError("Unsupported distance metric")
    
    def predict_single(self, x):
        """Predict class for a single sample"""
        # Calculate distances to all training points
        distances = []
        for train_point in self.X_train:
            dist = self.calculate_distance(x, train_point)
            distances.append(dist)
        
        # Find K nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        
        # Return majority vote
        from collections import Counter
        votes = Counter(nearest_labels)
        return votes.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict classes for multiple samples"""
        predictions = []
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Test our implementation
simple_knn = SimpleKNN(k=3)
simple_knn.fit(X_train_scaled, y_train)

custom_accuracy = simple_knn.score(X_test_scaled, y_test)
sklearn_accuracy = KNeighborsClassifier(n_neighbors=3).fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

print(f"Our implementation: {custom_accuracy:.3f}")
print(f"Sklearn implementation: {sklearn_accuracy:.3f}")
print("Pretty close! 🎉")
```

## Real-World Applications 🌍

### 1. Recommendation Systems
```python
# Content-based movie recommendations
# Find movies similar to ones you've liked before

movie_features = ['Action_Score', 'Comedy_Score', 'Drama_Score', 'Runtime', 'Year']
# Use KNN to find similar movies based on features
```

### 2. Image Classification (with proper features)
```python
# Simple image classification using extracted features
# (Note: For raw pixels, deep learning is better)

from sklearn.datasets import load_digits

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# KNN works surprisingly well on digit images
knn_digits = KNeighborsClassifier(n_neighbors=3)
digit_scores = cross_val_score(knn_digits, X_digits, y_digits, cv=5)
print(f"Digit classification accuracy: {digit_scores.mean():.3f}")
```

### 3. Anomaly Detection
```python
# Use KNN for outlier detection
from sklearn.neighbors import LocalOutlierFactor

# Detect anomalies based on local density
lof = LocalOutlierFactor(n_neighbors=5)
outlier_scores = lof.fit_predict(X_scaled)

print(f"Number of outliers detected: {np.sum(outlier_scores == -1)}")
```

## Performance Comparison 🏁

Let's compare KNN with other algorithms on the same dataset:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Compare performance
results = {}
for name, model in models.items():
    if name == 'KNN':
        # KNN needs scaled data
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        # Other models can use original data
        scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = scores
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Visualize comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[name].mean() for name in model_names]
errors = [results[name].std() for name in model_names]

plt.bar(model_names, accuracies, yerr=errors, capsize=5, alpha=0.7)
plt.ylabel('Cross-validation Accuracy')
plt.title('Algorithm Comparison on Same Dataset')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## When to Use KNN 🎯

### Perfect for:
- **Small to medium datasets**: Where computation time isn't critical
- **Non-linear patterns**: That are difficult to model parametrically
- **Recommendation systems**: Finding similar users or items
- **Prototype-based learning**: When examples are the best explanation
- **Baseline model**: Quick and simple benchmark
- **Local pattern recognition**: Where global models fail

### Avoid when:
- **Large datasets**: Training and prediction become too slow
- **High-dimensional data**: Curse of dimensionality hurts performance
- **Real-time applications**: Distance calculations are expensive
- **Sparse data**: Many zero values make distances meaningless
- **Memory constraints**: Must store entire training set

## Common Pitfalls & Solutions ⚠️

### 1. Forgetting to Scale Features
```python
# Problem: Features with different scales
# Solution: Always use StandardScaler or MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

# Choose based on your data distribution
standard_scaler = StandardScaler()  # For normal distributions
minmax_scaler = MinMaxScaler()     # For bounded features
```

### 2. Using Wrong K Value
```python
# Problem: K too small (noisy) or too large (oversmoothed)
# Solution: Use cross-validation to find optimal K

def find_best_k(X, y, k_range=range(1, 21)):
    best_k = 1
    best_score = 0
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        avg_score = scores.mean()
        
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    
    return best_k, best_score
```

### 3. Not Handling Categorical Variables Properly
```python
# Problem: Mixing categorical and numerical features
# Solution: Use appropriate distance metrics or encoding

# For mixed data types, consider Gower distance
# or separate KNN for different feature types
```

## Advanced Tips 💡

### 1. Class Imbalance Handling
```python
# Use different K values for different classes
from collections import Counter

def adaptive_knn_predict(knn_model, X_test, y_train):
    """Adapt K based on class frequencies"""
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    
    predictions = []
    for x in X_test:
        neighbors = knn_model.kneighbors([x], return_distance=False)[0]
        neighbor_classes = y_train[neighbors]
        
        # Weight votes by inverse class frequency
        weighted_votes = Counter()
        for cls in neighbor_classes:
            weight = 1 / class_counts[cls]
            weighted_votes[cls] += weight
        
        predictions.append(weighted_votes.most_common(1)[0][0])
    
    return np.array(predictions)
```

### 2. Dynamic K Selection
```python
def dynamic_k_selection(X_train, y_train, x_query, k_range=range(1, 11)):
    """Select K based on local data density"""
    knn_temp = KNeighborsClassifier(n_neighbors=max(k_range))
    knn_temp.fit(X_train, y_train)
    
    distances, indices = knn_temp.kneighbors([x_query], n_neighbors=max(k_range))
    
    # Choose K based on distance distribution
    distance_gaps = np.diff(distances[0])
    if len(distance_gaps) > 0:
        largest_gap_idx = np.argmax(distance_gaps)
        optimal_k = largest_gap_idx + 1
    else:
        optimal_k = 3  # Default
    
    return min(optimal_k, max(k_range))
```

## Key Takeaways 🎯

1. **KNN is beautifully simple**: Find nearest neighbors and vote
2. **Feature scaling is absolutely critical**: Different scales will bias results
3. **K selection matters**: Use cross-validation to find the best K
4. **Distance metric choice**: Can significantly impact performance
5. **Curse of dimensionality**: Performance degrades with too many features
6. **Lazy learning**: No training phase, all computation at prediction time
7. **Local patterns**: Excellent for datasets with local structure

## Next Steps 🚀

1. **Practice**: Try the interactive notebook `../../notebooks/06_knn_lab.ipynb`
2. **Experiment**: Test KNN on different types of datasets
3. **Learn Naive Bayes**: Another intuitive algorithm `02_naive_bayes.md`
4. **Compare**: How does KNN perform vs tree-based methods on your data?

## Quick Challenge 💪

Build a KNN-based system that can:
- **Recommend** books based on user reading history
- **Predict** if a customer will return based on purchase patterns  
- **Classify** emails as important vs. not important based on content features

Which distance metric would work best for each use case? Why?

*Solutions and detailed analysis in the exercises folder!*
