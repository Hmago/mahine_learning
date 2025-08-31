# Clustering Algorithms: Hands-On Exercises

## 🎯 Exercise Overview

These exercises progress from basic concept understanding to real-world applications. Each exercise includes:
- **Learning objective**
- **Dataset description**
- **Step-by-step tasks**
- **Expected insights**
- **Business applications**

## 🚀 Exercise 1: Customer Segmentation (Beginner)

### Objective
Learn K-Means fundamentals by segmenting customers for targeted marketing.

### Dataset
E-commerce customer data with purchasing behavior:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate realistic customer data
np.random.seed(42)
n_customers = 500

# Create customer segments with realistic patterns
# Segment 1: Young professionals (high tech spending)
young_prof = {
    'age': np.random.normal(28, 4, 150),
    'income': np.random.normal(65000, 15000, 150),
    'tech_spending': np.random.normal(1200, 300, 150),
    'grocery_spending': np.random.normal(300, 50, 150),
    'entertainment_spending': np.random.normal(200, 80, 150)
}

# Segment 2: Families (high grocery, moderate tech)
families = {
    'age': np.random.normal(38, 6, 200),
    'income': np.random.normal(75000, 20000, 200),
    'tech_spending': np.random.normal(600, 200, 200),
    'grocery_spending': np.random.normal(600, 100, 200),
    'entertainment_spending': np.random.normal(400, 150, 200)
}

# Segment 3: Retirees (low tech, moderate grocery)
retirees = {
    'age': np.random.normal(68, 8, 150),
    'income': np.random.normal(45000, 12000, 150),
    'tech_spending': np.random.normal(200, 100, 150),
    'grocery_spending': np.random.normal(400, 80, 150),
    'entertainment_spending': np.random.normal(150, 60, 150)
}

# Combine all segments
customer_data = pd.DataFrame({
    'age': np.concatenate([young_prof['age'], families['age'], retirees['age']]),
    'income': np.concatenate([young_prof['income'], families['income'], retirees['income']]),
    'tech_spending': np.concatenate([young_prof['tech_spending'], families['tech_spending'], retirees['tech_spending']]),
    'grocery_spending': np.concatenate([young_prof['grocery_spending'], families['grocery_spending'], retirees['grocery_spending']]),
    'entertainment_spending': np.concatenate([young_prof['entertainment_spending'], families['entertainment_spending'], retirees['entertainment_spending']])
})

# Add customer IDs
customer_data['customer_id'] = range(1, len(customer_data) + 1)

print("Customer Dataset Created!")
print(f"Shape: {customer_data.shape}")
print("\nFirst 5 customers:")
print(customer_data.head())
```

### Your Tasks

#### Task 1: Exploratory Data Analysis
```python
# 1.1 Create visualizations to understand the data
def explore_customer_data(df):
    """Create comprehensive EDA of customer data"""
    
    # TODO: Create the following plots:
    # - Distribution of each feature (histograms)
    # - Correlation matrix heatmap
    # - Scatter plots of key feature pairs
    # - Summary statistics
    
    # Your code here
    pass

# 1.2 Answer these questions:
questions = [
    "What's the age range of customers?",
    "Which spending category has the highest variance?", 
    "Are there obvious correlations between features?",
    "Do you see any obvious outliers?"
]
```

#### Task 2: Apply K-Means Clustering
```python
# 2.1 Prepare the data
def prepare_clustering_data(df):
    """Prepare data for clustering"""
    
    # TODO: 
    # - Select features for clustering (exclude customer_id)
    # - Handle any missing values
    # - Standardize the features (very important!)
    
    # Your code here
    pass

# 2.2 Find optimal number of clusters
def find_optimal_clusters(data, max_clusters=10):
    """Use elbow method to find optimal K"""
    
    # TODO:
    # - Try K from 1 to max_clusters
    # - Calculate inertia for each K
    # - Plot elbow curve
    # - Identify the optimal K
    
    # Your code here
    pass

# 2.3 Apply K-Means with optimal K
def apply_kmeans_clustering(data, k):
    """Apply K-Means and analyze results"""
    
    # TODO:
    # - Apply K-Means with k clusters
    # - Add cluster labels to original dataframe
    # - Calculate cluster centers
    # - Visualize clusters
    
    # Your code here
    pass
```

#### Task 3: Interpret and Validate Results
```python
# 3.1 Analyze each cluster
def analyze_clusters(df_with_clusters):
    """Analyze characteristics of each cluster"""
    
    # TODO:
    # - Calculate mean values for each cluster
    # - Create cluster profiles
    # - Name each cluster based on characteristics
    # - Visualize cluster differences
    
    # Your code here
    pass

# 3.2 Business recommendations
def generate_business_insights(cluster_analysis):
    """Generate actionable business recommendations"""
    
    # TODO:
    # - Suggest marketing strategies for each segment
    # - Recommend product development opportunities
    # - Identify cross-selling opportunities
    # - Estimate market size for each segment
    
    # Your code here
    pass
```

### Expected Results
- **3-4 distinct customer segments**
- **Clear business interpretation** for each segment
- **Actionable marketing strategies**
- **Understanding of K-Means strengths and limitations**

## 🔬 Exercise 2: DBSCAN for Fraud Detection (Intermediate)

### Objective
Master density-based clustering by detecting fraudulent transactions.

### Dataset
Credit card transactions with normal and fraudulent patterns:

```python
# Generate synthetic credit card transaction data
def generate_fraud_dataset():
    """Create realistic fraud detection dataset"""
    
    np.random.seed(42)
    
    # Normal transactions (95% of data)
    n_normal = 9500
    normal_transactions = {
        'amount': np.random.lognormal(3, 1, n_normal),
        'time_of_day': np.random.normal(12, 4, n_normal) % 24,
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], n_normal),
        'location_risk': np.random.beta(2, 8, n_normal),  # Low risk
        'frequency_last_week': np.random.poisson(3, n_normal)
    }
    
    # Fraudulent transactions (5% of data)  
    n_fraud = 500
    fraud_transactions = {
        'amount': np.random.lognormal(6, 1.5, n_fraud),  # Larger amounts
        'time_of_day': np.random.choice([2, 3, 4, 22, 23], n_fraud),  # Unusual times
        'merchant_category': np.random.choice(['online', 'cash_advance', 'luxury'], n_fraud),
        'location_risk': np.random.beta(8, 2, n_fraud),  # High risk
        'frequency_last_week': np.random.choice([0, 1, 15, 20], n_fraud)  # Very low or very high
    }
    
    # Combine and shuffle
    all_data = pd.DataFrame({
        'amount': np.concatenate([normal_transactions['amount'], fraud_transactions['amount']]),
        'time_of_day': np.concatenate([normal_transactions['time_of_day'], fraud_transactions['time_of_day']]),
        'merchant_category': np.concatenate([normal_transactions['merchant_category'], fraud_transactions['merchant_category']]),
        'location_risk': np.concatenate([normal_transactions['location_risk'], fraud_transactions['location_risk']]),
        'frequency_last_week': np.concatenate([normal_transactions['frequency_last_week'], fraud_transactions['frequency_last_week']])
    })
    
    # True labels (for validation)
    true_labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(all_data))
    all_data = all_data.iloc[shuffle_idx].reset_index(drop=True)
    true_labels = true_labels[shuffle_idx]
    
    return all_data, true_labels

fraud_data, true_fraud_labels = generate_fraud_dataset()
print(f"Dataset shape: {fraud_data.shape}")
print(f"Fraud rate: {true_fraud_labels.mean():.3%}")
```

### Your Tasks

#### Task 1: Data Exploration and Preprocessing
```python
# 1.1 Analyze fraud vs normal patterns
def analyze_fraud_patterns(data, labels):
    """Compare normal vs fraudulent transaction patterns"""
    
    # TODO:
    # - Compare distributions of each feature for normal vs fraud
    # - Identify which features best separate fraud from normal
    # - Create visualizations showing the differences
    # - Handle categorical variables (merchant_category)
    
    # Your code here
    pass

# 1.2 Prepare data for DBSCAN
def prepare_fraud_data(data):
    """Prepare data for DBSCAN clustering"""
    
    # TODO:
    # - Encode categorical variables (one-hot encoding)
    # - Standardize numerical features
    # - Handle any data quality issues
    
    # Your code here
    pass
```

#### Task 2: Apply DBSCAN
```python
# 2.1 Parameter tuning
def tune_dbscan_parameters(data):
    """Find optimal epsilon and min_samples for DBSCAN"""
    
    # TODO:
    # - Create k-distance plot to find optimal epsilon
    # - Try different min_samples values
    # - Evaluate clustering quality for each combination
    # - Consider business constraints (false positive rate)
    
    # Your code here
    pass

# 2.2 Apply DBSCAN and analyze results
def apply_dbscan_fraud_detection(data, eps, min_samples):
    """Apply DBSCAN for fraud detection"""
    
    # TODO:
    # - Apply DBSCAN with chosen parameters
    # - Analyze cluster sizes and characteristics
    # - Identify which cluster(s) represent normal transactions
    # - Classify remaining points as potential fraud
    
    # Your code here
    pass
```

#### Task 3: Evaluation and Business Impact
```python
# 3.1 Compare with true fraud labels
def evaluate_fraud_detection(predicted_anomalies, true_labels):
    """Evaluate fraud detection performance"""
    
    # TODO:
    # - Calculate precision, recall, F1-score
    # - Create confusion matrix
    # - Analyze false positives and false negatives
    # - Calculate business impact (money saved vs investigation costs)
    
    # Your code here
    pass

# 3.2 Improve the system
def improve_fraud_detection(data, initial_results):
    """Iterate to improve fraud detection"""
    
    # TODO:
    # - Analyze misclassified cases
    # - Try ensemble methods (combine with other algorithms)
    # - Adjust parameters based on business priorities
    # - Design real-time implementation strategy
    
    # Your code here
    pass
```

### Expected Results
- **Understanding of DBSCAN parameter tuning**
- **Practical fraud detection system**
- **Business impact analysis**
- **Comparison with other anomaly detection methods**

## 🌟 Exercise 3: Hierarchical Clustering for Gene Analysis (Advanced)

### Objective
Use hierarchical clustering to discover gene expression patterns and biological pathways.

### Dataset
Gene expression data across different conditions:

```python
# Generate synthetic gene expression data
def generate_gene_expression_data():
    """Create realistic gene expression dataset"""
    
    np.random.seed(42)
    
    # Define gene groups (biological pathways)
    pathway_1_genes = 20  # Cell division pathway
    pathway_2_genes = 15  # Immune response pathway  
    pathway_3_genes = 10  # Metabolic pathway
    housekeeping_genes = 10  # Always expressed
    
    total_genes = pathway_1_genes + pathway_2_genes + pathway_3_genes + housekeeping_genes
    conditions = ['healthy', 'diseased', 'treated', 'control', 'stressed']
    
    # Generate expression patterns for each pathway
    expression_data = np.zeros((total_genes, len(conditions)))
    
    # Pathway 1: Upregulated in diseased condition
    expression_data[0:pathway_1_genes, :] = np.random.normal(1, 0.3, (pathway_1_genes, len(conditions)))
    expression_data[0:pathway_1_genes, 1] += 2  # Higher in diseased condition
    
    # Pathway 2: Upregulated in treated condition
    expression_data[pathway_1_genes:pathway_1_genes+pathway_2_genes, :] = np.random.normal(1, 0.3, (pathway_2_genes, len(conditions)))
    expression_data[pathway_1_genes:pathway_1_genes+pathway_2_genes, 2] += 1.5  # Higher in treated
    
    # Pathway 3: Downregulated in stressed condition
    expression_data[pathway_1_genes+pathway_2_genes:pathway_1_genes+pathway_2_genes+pathway_3_genes, :] = np.random.normal(2, 0.3, (pathway_3_genes, len(conditions)))
    expression_data[pathway_1_genes+pathway_2_genes:pathway_1_genes+pathway_2_genes+pathway_3_genes, 4] -= 1  # Lower in stressed
    
    # Housekeeping genes: Stable expression
    expression_data[pathway_1_genes+pathway_2_genes+pathway_3_genes:, :] = np.random.normal(1.5, 0.1, (housekeeping_genes, len(conditions)))
    
    # Create gene names
    gene_names = []
    gene_names.extend([f'CellDiv_Gene_{i+1}' for i in range(pathway_1_genes)])
    gene_names.extend([f'Immune_Gene_{i+1}' for i in range(pathway_2_genes)])
    gene_names.extend([f'Metabolic_Gene_{i+1}' for i in range(pathway_3_genes)])
    gene_names.extend([f'Housekeeping_Gene_{i+1}' for i in range(housekeeping_genes)])
    
    # Create true pathway labels (for validation)
    true_pathways = ['CellDivision'] * pathway_1_genes + ['ImmuneResponse'] * pathway_2_genes + ['Metabolic'] * pathway_3_genes + ['Housekeeping'] * housekeeping_genes
    
    gene_df = pd.DataFrame(expression_data, index=gene_names, columns=conditions)
    
    return gene_df, true_pathways

gene_expression, true_pathways = generate_gene_expression_data()
print(f"Gene expression dataset shape: {gene_expression.shape}")
print(f"True pathways: {set(true_pathways)}")
```

### Your Tasks

#### Task 1: Gene Expression Analysis
```python
# 1.1 Explore the gene expression patterns
def explore_gene_expression(gene_df):
    """Analyze gene expression patterns across conditions"""
    
    # TODO:
    # - Create heatmap of gene expression
    # - Calculate correlation between conditions
    # - Identify genes with highest variance
    # - Find condition-specific expression patterns
    
    # Your code here
    pass

# 1.2 Preprocess for clustering
def preprocess_gene_data(gene_df):
    """Prepare gene expression data for clustering"""
    
    # TODO:
    # - Decide whether to cluster genes or conditions
    # - Standardize expression values (z-score normalization)
    # - Handle any missing values
    # - Consider log transformation if needed
    
    # Your code here
    pass
```

#### Task 2: Hierarchical Clustering
```python
# 2.1 Apply hierarchical clustering
def cluster_genes_hierarchically(gene_data):
    """Perform hierarchical clustering on genes"""
    
    # TODO:
    # - Try different linkage methods (ward, complete, average)
    # - Create dendrograms for each method
    # - Compare results and choose best method
    # - Determine optimal number of clusters
    
    # Your code here
    pass

# 2.2 Create enhanced visualizations
def create_clustered_heatmap(gene_df, cluster_labels):
    """Create heatmap ordered by clusters"""
    
    # TODO:
    # - Reorder genes by cluster assignment
    # - Create heatmap with cluster boundaries
    # - Add cluster labels and colors
    # - Annotate with biological insights
    
    # Your code here
    pass
```

#### Task 3: Biological Interpretation
```python
# 3.1 Validate against known pathways
def validate_gene_clusters(predicted_clusters, true_pathways):
    """Compare discovered clusters with known biological pathways"""
    
    # TODO:
    # - Calculate adjusted rand index
    # - Create confusion matrix of clusters vs pathways
    # - Identify which clusters correspond to which pathways
    # - Find genes that might be misclassified
    
    # Your code here
    pass

# 3.2 Generate biological hypotheses
def generate_biological_insights(gene_clusters, expression_patterns):
    """Generate testable biological hypotheses"""
    
    # TODO:
    # - Identify co-expressed gene groups
    # - Suggest functional relationships
    # - Propose experimental validations
    # - Prioritize genes for further study
    
    # Your code here
    pass
```

### Expected Insights
- **4 main gene clusters** corresponding to biological pathways
- **Condition-specific expression patterns**
- **Candidate genes** for experimental validation
- **Hypotheses about gene function** and regulation

## 🏢 Exercise 4: Multi-Algorithm Comparison Project (Expert)

### Objective
Compare all clustering algorithms on the same dataset to understand their strengths and limitations.

### Dataset
Complex customer behavior data with multiple cluster shapes:

```python
def generate_complex_clustering_dataset():
    """Create dataset with different cluster shapes for algorithm comparison"""
    
    np.random.seed(42)
    
    # Cluster 1: Spherical (good for K-Means)
    spherical = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 150)
    
    # Cluster 2: Elongated (challenging for K-Means)
    elongated = np.random.multivariate_normal([8, 8], [[3, 2.5], [2.5, 3]], 100)
    
    # Cluster 3: Crescent shape (impossible for K-Means)
    theta = np.random.uniform(0, np.pi, 120)
    r = 3 + np.random.normal(0, 0.3, 120)
    crescent_x = r * np.cos(theta) + 6
    crescent_y = r * np.sin(theta) + 2
    crescent = np.column_stack([crescent_x, crescent_y])
    
    # Cluster 4: Dense core with sparse outliers
    core = np.random.multivariate_normal([12, 6], [[0.5, 0], [0, 0.5]], 80)
    outliers = np.random.multivariate_normal([12, 6], [[3, 0], [0, 3]], 20)
    dense_cluster = np.vstack([core, outliers])
    
    # Noise points
    noise = np.random.uniform([0, 0], [15, 12], (30, 2))
    
    # Combine all data
    all_data = np.vstack([spherical, elongated, crescent, dense_cluster, noise])
    
    # True labels (for validation)
    true_labels = np.concatenate([
        np.zeros(150),      # Spherical
        np.ones(100),       # Elongated
        np.full(120, 2),    # Crescent
        np.full(100, 3),    # Dense
        np.full(30, -1)     # Noise
    ])
    
    return all_data, true_labels

complex_data, complex_true_labels = generate_complex_clustering_dataset()
```

### Your Tasks

#### Task 1: Visual Analysis
```python
def visualize_complex_dataset(data, true_labels):
    """Visualize the complex dataset"""
    
    # TODO:
    # - Create scatter plot with true cluster colors
    # - Analyze cluster shapes and densities
    # - Identify challenges for different algorithms
    # - Predict which algorithms will work best
    
    # Your code here
    pass
```

#### Task 2: Algorithm Comparison
```python
def compare_clustering_algorithms(data):
    """Compare K-Means, DBSCAN, and Hierarchical clustering"""
    
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    # TODO:
    # - Apply K-Means with different K values
    # - Apply DBSCAN with tuned parameters  
    # - Apply Hierarchical clustering with different linkages
    # - Calculate evaluation metrics for each
    # - Visualize results side by side
    
    # Your code here
    pass
```

#### Task 3: Algorithm Selection Framework
```python
def create_algorithm_selection_guide(comparison_results):
    """Create practical guide for choosing clustering algorithms"""
    
    # TODO:
    # - Analyze which algorithms work best for which cluster types
    # - Create decision tree for algorithm selection
    # - Document computational performance differences
    # - Recommend best practices for each algorithm
    
    # Your code here
    pass
```

## 🎯 Mini-Challenges

### Challenge 1: Real-Time Customer Segmentation (30 minutes)
```python
# Scenario: Streaming customer behavior data
# Goal: Update customer segments in real-time as new data arrives

def streaming_clustering_challenge():
    """Implement incremental clustering for streaming data"""
    
    # TODO:
    # - Simulate streaming customer data
    # - Implement incremental K-Means
    # - Handle concept drift (changing customer behavior)
    # - Maintain cluster quality over time
    
    pass
```

### Challenge 2: Multi-Modal Clustering (45 minutes)
```python
# Scenario: Customer data with both numerical and categorical features
# Goal: Cluster customers using mixed data types

def mixed_data_clustering_challenge():
    """Handle numerical + categorical data in clustering"""
    
    # TODO:
    # - Create dataset with mixed data types
    # - Implement appropriate distance metrics
    # - Compare different encoding strategies
    # - Validate results with business logic
    
    pass
```

### Challenge 3: Scalable Clustering (60 minutes)
```python
# Scenario: 1 million customer records
# Goal: Efficient clustering of large datasets

def scalable_clustering_challenge():
    """Implement clustering for large datasets"""
    
    # TODO:
    # - Use Mini-Batch K-Means for large data
    # - Implement sampling strategies
    # - Compare accuracy vs speed trade-offs
    # - Design distributed clustering approach
    
    pass
```

## 🏆 Success Criteria

### For Each Exercise:
- [ ] **Code Quality**: Clean, well-commented, modular code
- [ ] **Visualization**: Clear, informative plots with proper labels
- [ ] **Analysis**: Thorough interpretation of results
- [ ] **Business Value**: Actionable insights and recommendations
- [ ] **Validation**: Proper evaluation using appropriate metrics

### Advanced Criteria:
- [ ] **Scalability**: Consider computational efficiency
- [ ] **Robustness**: Handle edge cases and data quality issues
- [ ] **Generalizability**: Solutions work on similar but different datasets
- [ ] **Innovation**: Creative approaches to challenging problems

## 💡 Tips for Success

### 1. **Start Simple, Then Add Complexity**
- Begin with basic K-Means on clean data
- Gradually introduce challenges (noise, different shapes, mixed data types)
- Compare results at each step

### 2. **Always Validate Your Assumptions**
- Check data distributions before choosing algorithms
- Validate cluster stability with different random seeds
- Compare multiple algorithms on the same data

### 3. **Think Like a Business Stakeholder**
- What decisions will be made based on these clusters?
- How confident are you in the results?
- What are the costs of being wrong?

### 4. **Document Your Process**
- Keep track of parameter choices and their rationale
- Document insights and surprises
- Create reproducible analysis pipelines

## 🚀 Bonus Challenges

### 1. **Create an Interactive Clustering Dashboard**
- Use Plotly or Streamlit
- Allow parameter adjustment in real-time
- Show multiple algorithm results simultaneously

### 2. **Implement Custom Distance Metrics**
- Design domain-specific distance functions
- Compare with standard Euclidean distance
- Validate improvement on real problems

### 3. **Build a Clustering API**
- RESTful API for clustering services
- Handle different data formats
- Include automatic algorithm selection

## 💭 Reflection After Exercises

After completing these exercises, reflect on:

1. **Which algorithm worked best for which type of data?** Why do you think that was?

2. **What was the most challenging aspect of clustering?** Parameter tuning? Interpretation? Validation?

3. **How would you explain your clustering results to a non-technical stakeholder?**

4. **What additional data would have made clustering more effective?**

5. **How would you deploy these clustering solutions in production?**

Remember: The goal isn't just to run algorithms, but to gain genuine insights that drive business value. Focus on understanding the "why" behind each result!
