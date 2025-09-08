# K-Nearest Neighbors: Learning from Your Neighbors 🏘️

## 🌟 What is K-Nearest Neighbors (KNN)?

Imagine you just moved to a new neighborhood and want to know if you'll like living there. What would you do? You'd probably talk to your **nearest neighbors** and see what they think! KNN works exactly the same way.

**The Simple Philosophy**: "Tell me who your neighbors are, and I'll tell you who you are."

KNN operates on three simple steps:
1. **Find the K closest examples** to your new data point
2. **Ask them what class they belong to**
3. **Go with the majority vote**

It's that simple! No complex math, no training phase - just pure common sense.

## 🎯 Why KNN Matters in the Real World

KNN powers numerous applications across industries:

- **Recommendation Systems**: Netflix, Amazon, Spotify ("Users like you also enjoyed...")
- **Medical Diagnosis**: Finding similar patient cases and their outcomes
- **Image Recognition**: Facial recognition, object detection
- **Finance**: Credit scoring based on similar borrower profiles
- **Marketing**: Customer segmentation and targeted advertising
- **Real Estate**: Property valuation based on similar properties
- **Quality Control**: Defect detection using similar product patterns

**Real Impact**: Netflix's recommendation system uses KNN variants to suggest movies to 200+ million users, driving 80% of content consumption!

## 🧠 Intuitive Understanding: The Neighborhood Principle

### The Core Philosophy

Think of KNN as asking advice from your most similar friends:

- **Buying a car?** Ask friends with similar income and family size
- **Choosing a restaurant?** Ask neighbors with similar taste preferences  
- **Medical symptoms?** Look at patients with similar characteristics
- **Investment decisions?** Follow investors with similar risk profiles

### Distance: What Makes Neighbors "Close"?

In the physical world, neighbors are close in space. In machine learning, neighbors are close in **feature space**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

def visualize_neighborhood_concept():
    """
    Demonstrate the neighborhood concept with simple 2D data
    """
    # Create sample data: restaurant preferences based on price and noise
    np.random.seed(42)
    
    # Generate data: [Price_Level, Noise_Level]
    centers = [[2, 2], [7, 7]]  # Two types of restaurant preferences
    X, y = make_blobs(n_samples=40, centers=centers, cluster_std=1.5, random_state=42)
    
    # New customer to classify
    new_customer = np.array([[4, 5]])
    
    # Train KNN with different K values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    k_values = [1, 3, 7]
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        
        # Find neighbors
        distances, neighbor_indices = knn.kneighbors(new_customer)
        
        # Plot all data points
        colors = ['red', 'blue']
        labels = ['Prefers Quiet/Cheap', 'Prefers Loud/Expensive']
        
        for class_idx in range(2):
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx], 
                      s=60, alpha=0.6, label=labels[class_idx])
        
        # Highlight neighbors
        neighbors = X[neighbor_indices[0]]
        ax.scatter(neighbors[:, 0], neighbors[:, 1], 
                  s=200, facecolors='none', edgecolors='orange', 
                  linewidth=3, label=f'{k} Nearest Neighbors')
        
        # Plot new customer
        ax.scatter(new_customer[:, 0], new_customer[:, 1], 
                  c='green', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label='New Customer')
        
        # Draw distance circles for visualization
        for i, distance in enumerate(distances[0]):
            circle = plt.Circle(new_customer[0], distance, 
                              fill=False, color='orange', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
        
        # Prediction
        prediction = knn.predict(new_customer)[0]
        prob = knn.predict_proba(new_customer)[0]
        
        ax.set_title(f'K={k}: Prediction = {labels[prediction]}\n'
                    f'Confidence = {max(prob):.2%}', fontsize=12)
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Noise Level')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_neighborhood_concept()
```

## 📏 Distance Metrics: How to Measure Similarity

The definition of "closeness" is crucial in KNN. Different distance metrics work better for different types of data:

### 1. Euclidean Distance (Most Common)

**Formula**: √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]

**When to use**: Continuous features with similar scales

```python
def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    """
    return np.sqrt(np.sum((point1 - point2)**2))

# Example
person_a = np.array([25, 50000])  # Age=25, Salary=50k
person_b = np.array([30, 55000])  # Age=30, Salary=55k
distance = euclidean_distance(person_a, person_b)
print(f"Euclidean distance: {distance:.2f}")
```

### 2. Manhattan Distance (City Block)

**Formula**: |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|

**When to use**: Features represent different units, high-dimensional data

```python
def manhattan_distance(point1, point2):
    """
    Calculate Manhattan (L1) distance
    """
    return np.sum(np.abs(point1 - point2))

# Comparison of distance metrics
def compare_distance_metrics():
    """
    Show how different metrics give different neighbors
    """
    # Sample data points
    points = np.array([
        [1, 1], [2, 3], [3, 1], [4, 4], [5, 2]
    ])
    query_point = np.array([3, 2])
    
    distances_euclidean = [euclidean_distance(query_point, p) for p in points]
    distances_manhattan = [manhattan_distance(query_point, p) for p in points]
    
    print("Distance Comparison:")
    print(f"{'Point':<12} {'Euclidean':<12} {'Manhattan':<12}")
    print("-" * 40)
    
    for i, point in enumerate(points):
        print(f"{str(point):<12} {distances_euclidean[i]:<12.2f} {distances_manhattan[i]:<12.2f}")
    
    # Show ranking differences
    euclidean_ranking = np.argsort(distances_euclidean)
    manhattan_ranking = np.argsort(distances_manhattan)
    
    print(f"\nNearest neighbors ranking:")
    print(f"Euclidean: {euclidean_ranking}")
    print(f"Manhattan: {manhattan_ranking}")

# Run comparison
compare_distance_metrics()
```

### 3. Cosine Distance

**Formula**: 1 - (A·B)/(||A|| × ||B||)

**When to use**: Text data, high-dimensional sparse data

```python
def cosine_distance(vec1, vec2):
    """
    Calculate cosine distance (1 - cosine similarity)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1  # Maximum distance for zero vectors
    
    cosine_sim = dot_product / (norm1 * norm2)
    return 1 - cosine_sim

# Example with text-like data (word frequencies)
doc1 = np.array([3, 1, 0, 2])  # [word1_count, word2_count, ...]
doc2 = np.array([1, 2, 1, 0])
distance = cosine_distance(doc1, doc2)
print(f"Cosine distance: {distance:.3f}")
```

## 🔧 Mathematical Foundation (Simplified)

### The KNN Algorithm Step-by-Step

```python
def simple_knn_implementation():
    """
    Simple implementation of KNN to understand the algorithm
    """
    class SimpleKNN:
        def __init__(self, k=3):
            self.k = k
        
        def fit(self, X_train, y_train):
            """Store training data (lazy learning)"""
            self.X_train = X_train
            self.y_train = y_train
        
        def predict(self, X_test):
            """Predict labels for test data"""
            predictions = []
            
            for test_point in X_test:
                # Calculate distances to all training points
                distances = []
                for train_point in self.X_train:
                    dist = euclidean_distance(test_point, train_point)
                    distances.append(dist)
                
                # Find k nearest neighbors
                neighbor_indices = np.argsort(distances)[:self.k]
                neighbor_labels = self.y_train[neighbor_indices]
                
                # Majority vote
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                prediction = unique_labels[np.argmax(counts)]
                predictions.append(prediction)
            
            return np.array(predictions)
        
        def predict_proba(self, X_test):
            """Predict class probabilities"""
            probabilities = []
            
            for test_point in X_test:
                distances = []
                for train_point in self.X_train:
                    dist = euclidean_distance(test_point, train_point)
                    distances.append(dist)
                
                neighbor_indices = np.argsort(distances)[:self.k]
                neighbor_labels = self.y_train[neighbor_indices]
                
                # Calculate probabilities based on neighbor votes
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                probs = counts / self.k
                
                # Create probability array for all classes
                all_classes = np.unique(self.y_train)
                prob_array = np.zeros(len(all_classes))
                for i, label in enumerate(unique_labels):
                    class_idx = np.where(all_classes == label)[0][0]
                    prob_array[class_idx] = probs[i]
                
                probabilities.append(prob_array)
            
            return np.array(probabilities)
    
    return SimpleKNN

# Test the implementation
SimpleKNN = simple_knn_implementation()
```

### Weighted KNN: Distance-Based Voting

Instead of simple majority voting, we can weight votes by distance:

```python
def weighted_knn_example():
    """
    Demonstrate weighted KNN where closer neighbors have more influence
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    # Create data where weighting matters
    X = np.array([
        [1, 1], [1.1, 1.1], [1.2, 1.2],  # Very close cluster
        [5, 5], [8, 8], [8.1, 8.1]       # Another cluster with close points
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    test_point = np.array([[1.05, 1.05]])  # Very close to first cluster
    
    # Compare uniform vs distance weighting
    knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
    
    knn_uniform.fit(X, y)
    knn_weighted.fit(X, y)
    
    prob_uniform = knn_uniform.predict_proba(test_point)[0]
    prob_weighted = knn_weighted.predict_proba(test_point)[0]
    
    print("Weighting Comparison:")
    print(f"Uniform weighting:   Class 0: {prob_uniform[0]:.3f}, Class 1: {prob_uniform[1]:.3f}")
    print(f"Distance weighting:  Class 0: {prob_weighted[0]:.3f}, Class 1: {prob_weighted[1]:.3f}")
    
    # Distance weighting gives more influence to very close neighbors

# Run weighted example
weighted_knn_example()
```

## 🎨 Comprehensive Real-World Examples

### Example 1: Movie Recommendation System

```python
def movie_recommendation_example():
    """
    Build a simple movie recommendation system using KNN
    """
    # Movie ratings data: [Action, Comedy, Drama, Horror, Romance]
    user_ratings = np.array([
        [5, 2, 4, 1, 3],  # User 1: Loves action, drama
        [1, 5, 2, 1, 4],  # User 2: Loves comedy, romance
        [4, 2, 5, 2, 2],  # User 3: Loves action, drama
        [2, 4, 1, 1, 5],  # User 4: Loves comedy, romance
        [5, 1, 4, 2, 1],  # User 5: Loves action, drama
        [1, 5, 2, 0, 4],  # User 6: Loves comedy, romance
        [3, 3, 3, 3, 3],  # User 7: Average on everything
        [4, 2, 5, 1, 2]   # User 8: Loves action, drama
    ])
    
    # User preferences: 0 = Action/Drama lover, 1 = Comedy/Romance lover
    user_types = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    
    # New user with ratings
    new_user = np.array([[4, 3, 4, 1, 2]])  # Seems to like action/drama
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(user_ratings, user_types)
    
    # Find similar users
    distances, neighbor_indices = knn.kneighbors(new_user)
    
    prediction = knn.predict(new_user)[0]
    probability = knn.predict_proba(new_user)[0]
    
    print("Movie Recommendation System:")
    print(f"New user ratings: {new_user[0]}")
    print(f"Most similar users: {neighbor_indices[0]}")
    print(f"Predicted type: {'Action/Drama Lover' if prediction == 0 else 'Comedy/Romance Lover'}")
    print(f"Confidence: {max(probability):.2%}")
    
    # Visualize in 2D (using first two features)
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue']
    labels = ['Action/Drama Lover', 'Comedy/Romance Lover']
    
    for user_type in range(2):
        mask = user_types == user_type
        plt.scatter(user_ratings[mask, 0], user_ratings[mask, 1], 
                   c=colors[user_type], s=100, alpha=0.7, label=labels[user_type])
    
    # Plot new user
    plt.scatter(new_user[0, 0], new_user[0, 1], 
               c='green', s=300, marker='*', 
               edgecolors='black', linewidth=2, label='New User')
    
    # Highlight neighbors
    neighbors = user_ratings[neighbor_indices[0]]
    plt.scatter(neighbors[:, 0], neighbors[:, 1], 
               s=200, facecolors='none', edgecolors='orange', linewidth=3)
    
    plt.xlabel('Action Movie Rating')
    plt.ylabel('Comedy Movie Rating')
    plt.title('Movie Recommendation using KNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run movie recommendation example
movie_recommendation_example()
```

### Example 2: Medical Diagnosis System

```python
def medical_diagnosis_example():
    """
    Medical diagnosis system using patient symptoms
    """
    # Patient data: [Age, Temperature, Blood_Pressure, Heart_Rate]
    patients = np.array([
        [65, 38.5, 140, 90],   # Elderly, fever, high BP → Infection
        [25, 36.8, 120, 70],   # Young, normal → Healthy
        [45, 37.2, 130, 80],   # Middle-aged, slight fever → Mild condition
        [70, 39.0, 150, 95],   # Elderly, high fever → Serious infection
        [30, 36.5, 110, 65],   # Young, normal → Healthy
        [55, 38.0, 135, 85],   # Middle-aged, fever → Infection
        [28, 36.9, 115, 68],   # Young, normal → Healthy
        [60, 38.8, 145, 92]    # Elderly, fever → Infection
    ])
    
    # Diagnoses: 0 = Healthy, 1 = Mild condition, 2 = Serious condition
    diagnoses = np.array([2, 0, 1, 2, 0, 2, 0, 2])
    
    # New patient
    new_patient = np.array([[55, 38.2, 138, 88]])
    
    # Normalize features (important for medical data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    patients_scaled = scaler.fit_transform(patients)
    new_patient_scaled = scaler.transform(new_patient)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(patients_scaled, diagnoses)
    
    # Make diagnosis
    prediction = knn.predict(new_patient_scaled)[0]
    probabilities = knn.predict_proba(new_patient_scaled)[0]
    
    diagnosis_labels = ['Healthy', 'Mild Condition', 'Serious Condition']
    
    print("Medical Diagnosis System:")
    print(f"Patient profile: Age={new_patient[0,0]}, Temp={new_patient[0,1]}°C, "
          f"BP={new_patient[0,2]}, HR={new_patient[0,3]}")
    print(f"Diagnosis: {diagnosis_labels[prediction]}")
    print(f"Confidence: {max(probabilities):.2%}")
    print("\nAll probabilities:")
    for i, label in enumerate(diagnosis_labels):
        print(f"  {label}: {probabilities[i]:.2%}")

# Run medical diagnosis example
medical_diagnosis_example()
```

## ⚖️ The Curse of Dimensionality

As the number of features increases, KNN faces a significant challenge:

### Understanding the Problem

```python
def curse_of_dimensionality_demo():
    """
    Demonstrate how distance becomes less meaningful in high dimensions
    """
    from sklearn.datasets import make_classification
    
    dimensions = [2, 10, 50, 100, 500]
    distance_ratios = []
    
    for dim in dimensions:
        # Generate random data in different dimensions
        X, _ = make_classification(n_samples=1000, n_features=dim, 
                                  n_redundant=0, random_state=42)
        
        # Calculate distances from first point to all others
        distances = [euclidean_distance(X[0], X[i]) for i in range(1, len(X))]
        
        # Calculate ratio of max to min distance
        min_dist = min(distances)
        max_dist = max(distances)
        ratio = max_dist / min_dist if min_dist > 0 else float('inf')
        
        distance_ratios.append(ratio)
        print(f"Dimension {dim:3d}: Min distance = {min_dist:.3f}, "
              f"Max distance = {max_dist:.3f}, Ratio = {ratio:.3f}")
    
    # Plot the effect
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, distance_ratios, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Max Distance / Min Distance')
    plt.title('Curse of Dimensionality: Distance Becomes Less Discriminative')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nAs dimensions increase, all points become roughly equidistant!")
    print("This makes 'nearest' neighbors less meaningful.")

# Run curse of dimensionality demo
curse_of_dimensionality_demo()
```

### Solutions to High-Dimensional Problems

1. **Dimensionality Reduction**: PCA, t-SNE, UMAP
2. **Feature Selection**: Choose most relevant features
3. **Distance Metric Learning**: Learn optimal distance functions
4. **Locality Sensitive Hashing**: Approximate nearest neighbors

## 📊 Comprehensive Pros and Cons Analysis

### ✅ Advantages

**Simplicity and Intuition:**
- **Easy to understand**: No complex mathematics or assumptions
- **No training required**: "Lazy learning" - just store the data
- **Naturally handles multi-class**: Works with any number of classes
- **Non-parametric**: No assumptions about data distribution

**Flexibility and Adaptability:**
- **Adapts to new data**: Performance improves with more training data
- **Local decision boundaries**: Can capture complex, non-linear patterns
- **Handles any data type**: Works with numerical, categorical, mixed data
- **Robust to outliers**: Single outliers don't affect global model

**Practical Benefits:**
- **Memory of all examples**: Maintains complete training information
- **Good baseline**: Often works well without tuning
- **Probabilistic outputs**: Provides confidence estimates
- **Interpretable predictions**: Can examine neighbor cases

### ❌ Disadvantages

**Computational Challenges:**
- **Slow prediction**: Must calculate distances to all training points
- **Memory intensive**: Stores entire training dataset
- **Doesn't scale well**: Performance degrades with large datasets
- **No model compression**: Cannot reduce model size

**Sensitivity Issues:**
- **Feature scaling dependent**: Requires normalized features
- **Sensitive to irrelevant features**: Noise features hurt performance
- **Curse of dimensionality**: Fails in very high-dimensional spaces
- **Imbalanced data problems**: Majority classes dominate

**Parameter Dependence:**
- **K selection crucial**: Wrong K can drastically hurt performance
- **Distance metric choice**: Different metrics give different results
- **Local region shape**: Assumes spherical neighborhoods
- **Boundary effects**: Performance varies across feature space

### 🎯 Choosing the Right K Value

The choice of K is crucial for KNN performance:

```python
def k_selection_analysis():
    """
    Systematic analysis of different K values
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different K values
    k_values = range(1, 31)
    mean_scores = []
    std_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5)
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.errorbar(k_values, mean_scores, yerr=std_scores, 
                capsize=3, capthick=1, linewidth=2)
    plt.xlabel('K Value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN Performance vs K Value')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal K
    optimal_k = k_values[np.argmax(mean_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.show()
    
    print(f"Optimal K value: {optimal_k}")
    print(f"Best accuracy: {max(mean_scores):.3f} ± {std_scores[optimal_k-1]:.3f}")
    
    # Guidelines for K selection
    print("\nK Selection Guidelines:")
    print("• K=1: Very flexible, prone to overfitting")
    print("• K=sqrt(n): Common rule of thumb")
    print("• Odd K: Avoids ties in binary classification")
    print("• Cross-validation: Systematic approach for optimal K")

# Run K selection analysis
k_selection_analysis()
```

## 🎯 When to Use KNN

### ✅ Perfect Scenarios

**Small to Medium Datasets:**
- Training data < 10,000 samples
- Real-time prediction not critical
- Memory usage not a constraint

**High-Quality, Relevant Features:**
- Features are meaningful and well-engineered
- Low-dimensional data (< 50 features)
- Features have similar scales or can be normalized

**Local Pattern Recognition:**
- Non-linear decision boundaries
- Local clusters and neighborhoods matter
- Pattern recognition in image/signal processing

**Recommendation Systems:**
- User-based collaborative filtering
- Content-based recommendations
- "Similar users/items" scenarios

**Specific Business Cases:**
- Medical diagnosis (finding similar cases)
- Quality control (detecting similar defects)
- Market research (customer segmentation)
- Text classification (document similarity)

### ❌ Avoid When

**Large-Scale Applications:**
- Big data scenarios (> 100,000 samples)
- Real-time prediction requirements
- Limited memory/storage constraints
- Distributed computing needs

**High-Dimensional Data:**
- Text data with thousands of features
- Image data without feature extraction
- Genomic data with thousands of genes
- Any dataset with > 100 features

**Specific Data Characteristics:**
- Severely imbalanced datasets
- Mostly irrelevant or noisy features
- Time series with temporal dependencies
- Data with clear global patterns

## 🛠️ Advanced Implementation Techniques

### Efficient KNN with Ball Tree and KD-Tree

```python
def efficient_knn_methods():
    """
    Compare different KNN algorithms for efficiency
    """
    from sklearn.neighbors import NearestNeighbors
    from time import time
    
    # Generate larger dataset
    X, _ = make_classification(n_samples=10000, n_features=10, random_state=42)
    query_points = X[:100]  # Use first 100 points as queries
    
    algorithms = ['brute', 'ball_tree', 'kd_tree', 'auto']
    
    print("Efficiency Comparison for Different KNN Algorithms:")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Fit Time':<12} {'Query Time':<12} {'Memory':<12}")
    print("-" * 60)
    
    for algorithm in algorithms:
        try:
            # Time the fitting
            start_time = time()
            nn = NearestNeighbors(n_neighbors=5, algorithm=algorithm)
            nn.fit(X)
            fit_time = time() - start_time
            
            # Time the querying
            start_time = time()
            distances, indices = nn.kneighbors(query_points)
            query_time = time() - start_time
            
            print(f"{algorithm:<12} {fit_time:<12.4f} {query_time:<12.4f} {'Variable':<12}")
            
        except Exception as e:
            print(f"{algorithm:<12} {'Error':<12} {'Error':<12} {'Error':<12}")
    
    print("\nAlgorithm Guide:")
    print("• Brute force: Always works, slow for large data")
    print("• KD-Tree: Fast for low dimensions (< 20)")
    print("• Ball Tree: Better for higher dimensions")
    print("• Auto: Automatically chooses best algorithm")

# Run efficiency comparison
efficient_knn_methods()
```

### Custom Distance Metrics

```python
def custom_distance_example():
    """
    Implement custom distance metrics for specific domains
    """
    from sklearn.neighbors import DistanceMetric
    
    # Example: Hamming distance for categorical data
    def hamming_distance(x, y):
        """Distance for categorical features"""
        return np.sum(x != y) / len(x)
    
    # Example: Weighted Euclidean distance
    def weighted_euclidean(x, y, weights):
        """Euclidean distance with feature weights"""
        return np.sqrt(np.sum(weights * (x - y)**2))
    
    # Custom distance for mixed data types
    def mixed_distance(x, y, categorical_mask, weights=None):
        """
        Distance for mixed numerical/categorical data
        """
        if weights is None:
            weights = np.ones(len(x))
        
        # Numerical features: weighted Euclidean
        num_dist = np.sum(weights[~categorical_mask] * 
                         (x[~categorical_mask] - y[~categorical_mask])**2)
        
        # Categorical features: weighted Hamming
        cat_dist = np.sum(weights[categorical_mask] * 
                         (x[categorical_mask] != y[categorical_mask]))
        
        return np.sqrt(num_dist) + cat_dist
    
    print("Custom Distance Metrics:")
    print("• Hamming: For categorical/binary features")
    print("• Weighted Euclidean: Different feature importance")
    print("• Mixed: Combination for heterogeneous data")
    print("• Domain-specific: Tailored to specific problems")

# Run custom distance example
custom_distance_example()
```

## 🚀 Advanced Applications and Extensions

### KNN for Regression

```python
def knn_regression_example():
    """
    Demonstrate KNN for regression problems
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Generate regression data
    X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
    
    # Train different KNN regressors
    k_values = [1, 5, 10, 20]
    
    plt.figure(figsize=(16, 4))
    
    for idx, k in enumerate(k_values):
        plt.subplot(1, 4, idx + 1)
        
        # Train KNN regressor
        knn_reg = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn_reg.fit(X, y)
        
        # Create smooth prediction line
        X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_plot = knn_reg.predict(X_plot)
        
        # Plot
        plt.scatter(X, y, alpha=0.6, s=30, label='Data')
        plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'KNN (K={k})')
        plt.title(f'KNN Regression (K={k})')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run KNN regression example
knn_regression_example()
```

### Outlier Detection with KNN

```python
def knn_outlier_detection():
    """
    Use KNN for outlier detection
    """
    from sklearn.neighbors import LocalOutlierFactor
    
    # Generate data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    X = np.vstack([normal_data, outliers])
    
    # Detect outliers using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outlier_labels = lof.fit_predict(X)
    outlier_scores = lof.negative_outlier_factor_
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['blue' if label == 1 else 'red' for label in outlier_labels]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
    plt.title('Outlier Detection using LOF')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.hist(outlier_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Local Outlier Factor Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Outlier Scores')
    plt.axvline(x=-1, color='red', linestyle='--', label='Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected {np.sum(outlier_labels == -1)} outliers out of {len(X)} points")

# Run outlier detection example
knn_outlier_detection()
```

## 🎓 Learning Path and Next Steps

### Immediate Practice

1. **Basic implementation**: Code KNN from scratch using NumPy
2. **Distance metrics**: Experiment with different distance functions
3. **K selection**: Use cross-validation to find optimal K
4. **Real datasets**: Apply KNN to UCI Machine Learning Repository datasets

### Advanced Understanding

1. **Efficiency algorithms**: Study Ball Tree, KD-Tree implementations
2. **Approximate methods**: Learn about Locality Sensitive Hashing
3. **Distance learning**: Explore metric learning techniques
4. **Ensemble methods**: Combine multiple KNN models

### Related Algorithms to Explore

- **Nearest Centroid**: Simpler alternative to KNN
- **Local Linear Embedding**: Dimensionality reduction using neighborhoods
- **DBSCAN**: Density-based clustering using neighborhoods
- **Collaborative Filtering**: KNN for recommendation systems
- **Kernel Methods**: Using similarity functions in other algorithms

### Real-World Applications

- **Recommendation Systems**: Build collaborative filtering systems
- **Image Recognition**: Implement simple face recognition
- **Anomaly Detection**: Create fraud detection systems
- **Text Classification**: Build document classification systems

---

**Remember**: KNN is often underestimated because of its simplicity, but it's incredibly powerful when applied correctly. Many successful production systems use KNN as a core component, especially in recommendation systems and similarity search applications. Master the fundamentals, understand its limitations, and you'll have a versatile tool for many machine learning problems! 🏘️🎯
