# 📋 Linear Algebra Cheat Sheet for ML

## 🔢 Vector Operations

| Operation | Formula | Code | ML Use |
|-----------|---------|------|--------|
| **Addition** | $\vec{u} + \vec{v}$ | `u + v` | Combining features |
| **Scalar Mult** | $c\vec{v}$ | `c * v` | Learning rates |
| **Dot Product** | $\vec{u} \cdot \vec{v} = \sum u_i v_i$ | `np.dot(u, v)` | Similarity, NN activation |
| **Cross Product** | $\vec{u} \times \vec{v}$ | `np.cross(u, v)` | Normal vectors |
| **L2 Norm** | $\|\|\vec{v}\|\|_2 = \sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` | Distance, regularization |
| **L1 Norm** | $\|\|\vec{v}\|\|_1 = \sum \|v_i\|$ | `np.linalg.norm(v, 1)` | Sparsity (Lasso) |

## 🔢 Matrix Operations

| Operation | Formula | Code | ML Use |
|-----------|---------|------|--------|
| **Multiply** | $C_{ij} = \sum A_{ik}B_{kj}$ | `A @ B` | Forward pass, transformations |
| **Transpose** | $(A^T)_{ij} = A_{ji}$ | `A.T` | Gradients, covariance |
| **Inverse** | $AA^{-1} = I$ | `np.linalg.inv(A)` | Normal equations (avoid!) |
| **Determinant** | $\det(A)$ | `np.linalg.det(A)` | Invertibility check |
| **Trace** | $\text{tr}(A) = \sum A_{ii}$ | `np.trace(A)` | Sum of eigenvalues |

## 🔢 Eigenvalues & Eigenvectors

| Concept | Formula | Code | ML Use |
|---------|---------|------|--------|
| **Definition** | $A\vec{v} = \lambda\vec{v}$ | `np.linalg.eig(A)` | PCA, spectral methods |
| **Characteristic** | $\det(A - \lambda I) = 0$ | - | Finding eigenvalues |
| **PCA Steps** | $C = \frac{1}{n}X^TX$ | `sklearn.decomposition.PCA` | Dimensionality reduction |

## 🧮 Quick Formulas

### **PCA in 4 Steps**
1. **Center**: `X_centered = X - X.mean(axis=0)`
2. **Covariance**: `C = np.cov(X_centered.T)`
3. **Eigendecomp**: `eigenvals, eigenvecs = np.linalg.eig(C)`
4. **Transform**: `X_pca = X_centered @ eigenvecs`

### **Linear Regression (Normal Equations)**
- **Solution**: $\hat{w} = (X^TX)^{-1}X^Ty$
- **Code**: `w = np.linalg.solve(X.T @ X, X.T @ y)`

### **Distance Metrics**
- **Euclidean**: $d = \|\|x - y\|\|_2$
- **Manhattan**: $d = \|\|x - y\|\|_1$
- **Cosine**: $\text{sim} = \frac{x \cdot y}{\|\|x\|\| \|\|y\|\|}$

## 🔍 Key Properties

### **Matrix Multiplication**
- **Not commutative**: $AB \neq BA$
- **Associative**: $(AB)C = A(BC)$
- **Distributive**: $A(B + C) = AB + AC$

### **Transpose**
- $(A^T)^T = A$
- $(AB)^T = B^TA^T$
- $(A + B)^T = A^T + B^T$

### **Determinant**
- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = 1/\det(A)$

## 🎯 ML Applications Quick Reference

| ML Algorithm | Linear Algebra Core |
|--------------|-------------------|
| **Linear Regression** | Matrix multiplication, transpose, inverse |
| **PCA** | Eigendecomposition, covariance matrix |
| **Neural Networks** | Matrix multiplication (layers), gradients |
| **SVM** | Dot products, kernel functions |
| **k-NN** | Distance metrics, norms |
| **Clustering** | Distance matrices, centroids |
| **Recommendation** | Matrix factorization (SVD) |
| **Image Processing** | Convolution (matrix operations) |

## ⚡ NumPy Quick Commands

```python
# Vector operations
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot_product = np.dot(u, v)          # or u @ v
norm = np.linalg.norm(u)            # L2 norm
unit_vector = u / np.linalg.norm(u) # Normalize

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B                           # Matrix multiply
A_T = A.T                           # Transpose
det_A = np.linalg.det(A)           # Determinant
inv_A = np.linalg.inv(A)           # Inverse (use solve() instead)

# Eigenvalues/eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)

# Linear system solving
x = np.linalg.solve(A, b)          # Solve Ax = b (preferred)
x = np.linalg.inv(A) @ b           # Using inverse (avoid)

# SVD
U, s, Vt = np.linalg.svd(A)

# Special matrices
I = np.eye(3)                      # 3x3 identity
zeros = np.zeros((2, 3))           # 2x3 zero matrix
ones = np.ones((3, 2))             # 3x2 ones matrix
```

## 🚨 Common Mistakes

| ❌ Wrong | ✅ Correct | Why |
|----------|------------|-----|
| `np.linalg.inv(A) @ b` | `np.linalg.solve(A, b)` | Numerical stability |
| `A * B` | `A @ B` | Element-wise vs matrix multiply |
| `np.transpose(A)` | `A.T` | Simpler syntax |
| Ignore dimensions | Check `A.shape` | Avoid dimension errors |

## 📊 When to Use What

### **Eigendecomposition vs SVD**
- **Eigendecomposition**: Square matrices, PCA on covariance
- **SVD**: Any matrix, more numerically stable

### **Norms**
- **L2**: Most common, smooth gradients
- **L1**: Sparsity, robust to outliers
- **L∞**: Uniform constraints

### **Distance Metrics**
- **Euclidean**: Continuous features, spherical clusters
- **Manhattan**: Discrete features, robust to outliers
- **Cosine**: Text data, high dimensions

## 🎓 Memory Aids

### **Matrix Dimensions**
- **(m×n) × (n×p) = (m×p)** - "Inner dimensions must match"
- **Row-Column Rule**: "Row of first × Column of second"

### **Eigenvalues**
- **Large eigenvalue** = **Important direction**
- **Zero eigenvalue** = **Null space**
- **Negative eigenvalue** = **Unstable direction**

### **PCA**
- **Eigenvectors** = **Principal components** (directions)
- **Eigenvalues** = **Variance explained** (importance)
- **First PC** = **Maximum variance direction**

This cheat sheet covers the essential linear algebra needed for 90% of ML applications!
