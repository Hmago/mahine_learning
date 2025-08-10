# 📚 Linear Algebra Practice Guide

## 🎯 Practice 1: Implement Matrix Operations from Scratch

### Learning Objectives
- Understand matrix operations at a fundamental level
- Implement core operations without relying on NumPy's built-in functions
- Gain intuition about computational complexity

### Exercises

#### 1.1 Vector Operations
```python
def vector_add(v1, v2):
    """Add two vectors element-wise"""
    # TODO: Implement vector addition
    pass

def vector_scalar_mult(scalar, vector):
    """Multiply vector by scalar"""
    # TODO: Implement scalar multiplication
    pass

def dot_product(v1, v2):
    """Compute dot product of two vectors"""
    # TODO: Implement dot product
    pass

def vector_norm(vector, p=2):
    """Compute L-p norm of vector"""
    # TODO: Implement vector norm (L1, L2, L-infinity)
    pass
```

#### 1.2 Matrix Operations
```python
def matrix_multiply(A, B):
    """Multiply two matrices A and B"""
    # TODO: Check dimension compatibility
    # TODO: Implement matrix multiplication
    # Hint: C[i,j] = sum(A[i,k] * B[k,j] for k in range(A.shape[1]))
    pass

def matrix_transpose(A):
    """Transpose matrix A"""
    # TODO: Implement transpose
    pass

def matrix_determinant_2x2(A):
    """Compute determinant of 2x2 matrix"""
    # TODO: Implement det(A) = ad - bc
    pass

def matrix_inverse_2x2(A):
    """Compute inverse of 2x2 matrix"""
    # TODO: Check if matrix is invertible
    # TODO: Use formula: A^(-1) = (1/det(A)) * [[d, -b], [-c, a]]
    pass
```

#### 1.3 Performance Comparison
```python
import time
import numpy as np

def benchmark_operations():
    """Compare custom implementations with NumPy"""
    # Create test matrices
    size = 100
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    
    # Benchmark your implementation vs NumPy
    # Time matrix multiplication, transpose, etc.
    pass
```

---

## 🎯 Practice 2: Visualize Vector Operations in 2D/3D

### Learning Objectives
- Develop geometric intuition for vector operations
- Create interactive visualizations
- Understand transformations geometrically

### Exercises

#### 2.1 2D Vector Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_vector_addition():
    """Create interactive visualization of vector addition"""
    # TODO: Plot two vectors and their sum
    # TODO: Show head-to-tail method
    # TODO: Show parallelogram method
    pass

def visualize_dot_product():
    """Visualize dot product and angle between vectors"""
    # TODO: Plot two vectors
    # TODO: Show angle between them
    # TODO: Display dot product value
    # TODO: Show how dot product changes as angle changes
    pass

def visualize_transformations():
    """Visualize 2D linear transformations"""
    transformations = {
        'identity': np.array([[1, 0], [0, 1]]),
        'scaling': np.array([[2, 0], [0, 0.5]]),
        'rotation': np.array([[0, -1], [1, 0]]),  # 90 degrees
        'shear': np.array([[1, 1], [0, 1]]),
        'reflection': np.array([[1, 0], [0, -1]])
    }
    
    # TODO: For each transformation:
    # - Apply to unit square
    # - Show before/after
    # - Display determinant
    pass
```

#### 2.2 3D Vector Visualization
```python
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_vectors():
    """Visualize vectors in 3D space"""
    # TODO: Plot 3D vectors
    # TODO: Show cross product result
    # TODO: Demonstrate right-hand rule
    pass

def visualize_3d_transformations():
    """Visualize 3D linear transformations"""
    # TODO: Apply 3D transformations to cube
    # TODO: Show rotations around different axes
    pass
```

---

## 🎯 Practice 3: Build PCA Algorithm Manually

### Learning Objectives
- Implement PCA from mathematical foundations
- Understand the role of eigenvalues/eigenvectors
- Compare with sklearn implementation

### Exercises

#### 3.1 PCA Implementation
```python
def manual_pca(data, n_components=None):
    """
    Implement PCA from scratch
    
    Steps:
    1. Center the data
    2. Compute covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Sort by eigenvalue magnitude
    5. Transform data to PCA space
    """
    # TODO: Implement each step
    
    # Step 1: Center data
    # data_centered = ?
    
    # Step 2: Covariance matrix
    # cov_matrix = ?
    
    # Step 3: Eigendecomposition
    # eigenvals, eigenvecs = ?
    
    # Step 4: Sort components
    # Sort by eigenvalue (descending)
    
    # Step 5: Transform data
    # Select top n_components
    # transformed_data = ?
    
    return {
        'transformed_data': None,
        'principal_components': None,
        'explained_variance': None,
        'explained_variance_ratio': None
    }

def compare_with_sklearn():
    """Compare manual PCA with sklearn"""
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    
    # Load data
    data = load_iris().data
    
    # Manual PCA
    manual_result = manual_pca(data, n_components=2)
    
    # Sklearn PCA
    sklearn_pca = PCA(n_components=2)
    sklearn_result = sklearn_pca.fit_transform(data)
    
    # Compare results (they might differ by sign)
    # TODO: Compare explained variance ratios
    # TODO: Visualize both results
    pass
```

#### 3.2 PCA Visualization
```python
def visualize_pca_process():
    """Visualize each step of PCA"""
    # TODO: Create 2D correlated data
    # TODO: Show original data
    # TODO: Show centered data
    # TODO: Show principal components
    # TODO: Show data in PCA space
    # TODO: Show reconstruction
    pass

def pca_dimensionality_reduction():
    """Demonstrate dimensionality reduction with PCA"""
    # TODO: Use high-dimensional dataset
    # TODO: Apply PCA with different n_components
    # TODO: Show cumulative explained variance
    # TODO: Find elbow point for optimal components
    pass
```

---

## 🎯 Practice 4: Image Compression using SVD

### Learning Objectives
- Apply SVD to real-world problem
- Understand rank approximation
- Balance compression vs quality

### Project Overview
Singular Value Decomposition (SVD) can compress images by keeping only the most important singular values.

For any matrix A: **A = UΣV^T**
- **U**: Left singular vectors
- **Σ**: Singular values (diagonal)
- **V^T**: Right singular vectors

### Exercises

#### 4.1 SVD Implementation
```python
def compress_image_svd(image, k):
    """
    Compress image using SVD with k singular values
    
    Args:
        image: 2D numpy array (grayscale) or 3D (RGB)
        k: number of singular values to keep
    
    Returns:
        compressed_image: reconstructed image
        compression_ratio: original_size / compressed_size
    """
    # TODO: Handle grayscale vs RGB
    # TODO: Apply SVD to each channel
    # TODO: Keep only top k singular values
    # TODO: Reconstruct image
    # TODO: Calculate compression ratio
    pass

def analyze_compression_quality():
    """Analyze quality vs compression tradeoff"""
    # TODO: Load test image
    # TODO: Try different values of k
    # TODO: Calculate MSE, PSNR for each k
    # TODO: Plot quality metrics vs compression ratio
    pass
```

#### 4.2 Visualization and Analysis
```python
def visualize_svd_compression():
    """Visualize SVD compression process"""
    # TODO: Show original image
    # TODO: Show compressed versions with different k
    # TODO: Plot singular values (spectrum)
    # TODO: Show cumulative energy preserved
    pass

def interactive_compression_demo():
    """Create interactive demo of SVD compression"""
    # TODO: Use ipywidgets for interactive k selection
    # TODO: Real-time compression visualization
    # TODO: Display compression statistics
    pass
```

#### 4.3 Advanced Analysis
```python
def compare_compression_methods():
    """Compare SVD with other compression methods"""
    # TODO: Compare with JPEG compression
    # TODO: Analyze different image types (natural, synthetic)
    # TODO: Study effect of image content on compression
    pass

def svd_for_color_images():
    """Handle RGB images with SVD"""
    # TODO: Apply SVD to each color channel
    # TODO: Compare with applying SVD to combined data
    # TODO: Analyze color preservation
    pass
```

---

## 🔬 Challenge Projects

### 1. **Face Recognition with Eigenfaces**
- Use PCA on face images to create "eigenfaces"
- Implement face recognition using PCA projection
- Compare different numbers of components

### 2. **Recommendation System with Matrix Factorization**
- Implement collaborative filtering using SVD
- Handle missing ratings in user-item matrix
- Compare with modern recommendation algorithms

### 3. **Data Visualization with t-SNE and PCA**
- Compare PCA vs t-SNE for high-dimensional data
- Implement dimensionality reduction pipeline
- Visualize clusters in reduced space

### 4. **Linear Regression from Linear Algebra**
- Derive normal equations using matrix calculus
- Implement ridge regression with matrix operations
- Understand geometric interpretation of regression

---

## 📋 Self-Assessment Checklist

### Vectors ✅
- [ ] Can implement vector operations from scratch
- [ ] Understand geometric meaning of dot/cross products
- [ ] Can visualize vectors in 2D/3D
- [ ] Know when vectors are orthogonal/parallel

### Matrices ✅
- [ ] Can multiply matrices by hand and code
- [ ] Understand transpose, inverse, determinant
- [ ] Can visualize matrix transformations
- [ ] Know conditions for matrix invertibility

### Eigenvalues/Eigenvectors ✅
- [ ] Understand eigenvector geometric meaning
- [ ] Can implement PCA from scratch
- [ ] Know applications in dimensionality reduction
- [ ] Can interpret principal components

### Vector Spaces ✅
- [ ] Understand linear independence concept
- [ ] Can check if vectors form a basis
- [ ] Know what span means geometrically
- [ ] Can work with different coordinate systems

### Applications ✅
- [ ] Implemented SVD image compression
- [ ] Built PCA for real datasets
- [ ] Can choose appropriate linear algebra tools
- [ ] Understand computational complexity

## 🎓 Next Steps

1. **Practice Daily**: Implement one concept per day
2. **Real Projects**: Apply to actual datasets
3. **Advanced Topics**: Learn about tensor operations, matrix calculus
4. **ML Integration**: See how these concepts appear in neural networks
5. **Optimization**: Study how linear algebra enables efficient ML algorithms

Remember: **Linear algebra is the language of machine learning**. The more intuitive these concepts become, the better you'll understand ML algorithms!
