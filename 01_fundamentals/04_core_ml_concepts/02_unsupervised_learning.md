## Unsupervised Learning

### What is Unsupervised Learning?
Unsupervised learning is a type of machine learning where the model is trained on data without labeled responses. The goal is to identify patterns, groupings, or structures within the data. Unlike supervised learning, where the model learns from labeled data, unsupervised learning deals with input data that has no corresponding output labels.

### Why Does This Matter?
Unsupervised learning is crucial for exploratory data analysis, allowing us to uncover hidden patterns in data. It can help in customer segmentation, anomaly detection, and feature extraction, making it a powerful tool in various applications, from marketing to fraud detection.

### Key Concepts

1. **Clustering**: This is the process of grouping similar data points together. Common algorithms include:
   - **K-Means**: Partitions data into K distinct clusters based on feature similarity.
   - **Hierarchical Clustering**: Builds a tree of clusters by either merging or splitting them based on distance metrics.

   **Example**: In customer segmentation, K-Means can be used to group customers based on purchasing behavior, helping businesses tailor their marketing strategies.

2. **Dimensionality Reduction**: This technique reduces the number of features in a dataset while preserving its essential characteristics. Common methods include:
   - **Principal Component Analysis (PCA)**: Transforms the data into a new coordinate system, where the greatest variance by any projection lies on the first coordinate (principal component).
   - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: A technique for visualizing high-dimensional data by reducing it to two or three dimensions.

   **Example**: PCA can be used in image compression, where high-dimensional image data is reduced to a lower-dimensional representation while retaining important features.

3. **Association Rules**: This method identifies interesting relationships between variables in large databases. A common example is market basket analysis, which uncovers patterns in customer purchases.

   **Example**: If customers who buy bread often buy butter, a supermarket can place these items closer together to increase sales.

### Practical Applications
- **Market Segmentation**: Businesses can use unsupervised learning to identify distinct customer segments based on purchasing behavior, enabling targeted marketing strategies.
- **Anomaly Detection**: In cybersecurity, unsupervised learning can help identify unusual patterns that may indicate fraudulent activity.
- **Recommendation Systems**: By clustering users based on their preferences, companies can recommend products that similar users liked.

### Thought Experiment
Imagine you have a dataset of various fruits with features like weight, color, and sweetness. Without any labels, how would you group these fruits? Would you cluster them based on their similarities? What patterns might emerge?

### Conclusion
Unsupervised learning is a powerful approach for discovering hidden structures in data. By understanding and applying these techniques, you can gain valuable insights and make data-driven decisions in various fields.

## Mathematical Foundation

### Key Formulas

**K-Means Clustering:**

**Objective Function:**
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $C_i$ = cluster $i$
- $\mu_i$ = centroid of cluster $i$
- $k$ = number of clusters

**Centroid Update:**
$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

**Principal Component Analysis (PCA):**

**Covariance Matrix:**
$$C = \frac{1}{n-1} X^T X$$

**Principal Components:**
Eigenvectors of covariance matrix ordered by eigenvalue magnitude

**Variance Explained:**
$$\text{Variance Explained by PC}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

**Dimensionality Reduction:**
$$Y = XW$$
Where $W$ contains the first $k$ principal components

### Solved Examples

#### Example 1: K-Means Clustering Step-by-Step

Given: Data points $A(1,1)$, $B(2,1)$, $C(4,3)$, $D(5,4)$ with $k=2$

Find: Final cluster assignments

Solution:
Step 1: Initialize centroids randomly
$\mu_1 = (1.5, 1)$, $\mu_2 = (4.5, 3.5)$

Step 2: Assign points to nearest centroid (Iteration 1)
Distance calculations:
- $A$ to $\mu_1$: $\sqrt{(1-1.5)^2 + (1-1)^2} = 0.5$ → Cluster 1
- $A$ to $\mu_2$: $\sqrt{(1-4.5)^2 + (1-3.5)^2} = \sqrt{18.5} = 4.3$
- $B$ to $\mu_1$: $\sqrt{(2-1.5)^2 + (1-1)^2} = 0.5$ → Cluster 1
- $C$ to $\mu_2$: $\sqrt{(4-4.5)^2 + (3-3.5)^2} = \sqrt{0.5} = 0.7$ → Cluster 2
- $D$ to $\mu_2$: $\sqrt{(5-4.5)^2 + (4-3.5)^2} = \sqrt{0.5} = 0.7$ → Cluster 2

Assignments: Cluster 1: $\{A, B\}$, Cluster 2: $\{C, D\}$

Step 3: Update centroids
$$\mu_1 = \frac{(1,1) + (2,1)}{2} = (1.5, 1)$$
$$\mu_2 = \frac{(4,3) + (5,4)}{2} = (4.5, 3.5)$$

Result: Centroids unchanged → Convergence achieved!

#### Example 2: PCA Dimensionality Reduction

Given: Data matrix $X = \begin{bmatrix} 2 & 4 \\ 3 & 5 \\ 4 & 6 \\ 5 & 7 \end{bmatrix}$ (4 samples, 2 features)

Find: First principal component and reduced representation

Solution:
Step 1: Center the data
$$\bar{x} = \begin{bmatrix} 3.5 \\ 5.5 \end{bmatrix}$$

$$X_{centered} = \begin{bmatrix} -1.5 & -1.5 \\ -0.5 & -0.5 \\ 0.5 & 0.5 \\ 1.5 & 1.5 \end{bmatrix}$$

Step 2: Calculate covariance matrix
$$C = \frac{1}{3}X_{centered}^T X_{centered} = \frac{1}{3}\begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix} = \begin{bmatrix} 1.33 & 1.33 \\ 1.33 & 1.33 \end{bmatrix}$$

Step 3: Find eigenvalues and eigenvectors
Characteristic equation: $\det(C - \lambda I) = (1.33 - \lambda)^2 - 1.33^2 = 0$
Eigenvalues: $\lambda_1 = 2.67$, $\lambda_2 = 0$

First eigenvector: $v_1 = \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}$ (normalized)

Step 4: Project data onto first principal component
$$Y = X_{centered} \cdot v_1 = \begin{bmatrix} -2.12 \\ -0.71 \\ 0.71 \\ 2.12 \end{bmatrix}$$

Result: 2D data reduced to 1D while preserving 100% of variance.

#### Example 3: Hierarchical Clustering with Distance Matrix

Given: Points $P_1(0,0)$, $P_2(1,0)$, $P_3(0,1)$, $P_4(10,10)$

Find: Hierarchical clustering using single linkage

Solution:
Step 1: Calculate distance matrix
$$D = \begin{bmatrix} 0 & 1 & 1 & 14.14 \\ 1 & 0 & 1.41 & 14.04 \\ 1 & 1.41 & 0 & 14.14 \\ 14.14 & 14.04 & 14.14 & 0 \end{bmatrix}$$

Step 2: Merge closest points iteratively
Iteration 1: Merge $P_1$ and $P_2$ (distance = 1)
Iteration 2: Merge $\{P_1, P_2\}$ and $P_3$ (min distance = 1)
Iteration 3: Merge $\{P_1, P_2, P_3\}$ and $P_4$ (min distance = 14.04)

Step 3: Build dendrogram
```
    P4 ---|
          |--- 14.04
P1-P2 ---|
    |
    P3 ---|--- 1.0
```

Result: Two main clusters identified: $\{P_1, P_2, P_3\}$ and $\{P_4\}$ with clear separation.

### Suggested Exercises
- Implement K-Means clustering on a dataset of your choice and visualize the clusters.
- Use PCA to reduce the dimensionality of a dataset and observe how it affects data representation.
- Conduct a market basket analysis on a retail dataset to find association rules.