# Calculus & Optimization Quick Reference 📋

## 🧮 Essential Formulas

### Derivatives
```
Basic Rules:
d/dx[x^n] = nx^(n-1)              # Power rule
d/dx[f(x)g(x)] = f'g + fg'        # Product rule  
d/dx[f(g(x))] = f'(g(x)) × g'(x)  # Chain rule

Common Functions:
d/dx[sin(x)] = cos(x)
d/dx[cos(x)] = -sin(x)  
d/dx[e^x] = e^x
d/dx[ln(x)] = 1/x
d/dx[1/(1+e^(-x))] = σ(x)(1-σ(x)) # Sigmoid derivative
```

### Partial Derivatives & Gradients
```
∂f/∂x = lim[h→0] (f(x+h,y) - f(x,y))/h

Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Chain Rule (multivariable):
∂z/∂t = (∂z/∂x)(∂x/∂t) + (∂z/∂y)(∂y/∂t)
```

### Optimization
```
Gradient Descent: x_{k+1} = x_k - α∇f(x_k)

Newton's Method: x_{k+1} = x_k - H⁻¹(x_k)∇f(x_k)

Lagrange Multipliers: ∇f = λ∇g (at constrained optimum)
```

---

## 📊 Key Concepts

### Critical Points
| Type | Condition (1D) | Condition (2D) | Visualization |
|------|----------------|----------------|---------------|
| **Local Min** | f'(x) = 0, f''(x) > 0 | ∇f = 0, det(H) > 0, fₓₓ > 0 | Bowl shape ⌒ |
| **Local Max** | f'(x) = 0, f''(x) < 0 | ∇f = 0, det(H) > 0, fₓₓ < 0 | Dome shape ⌒ |
| **Saddle** | N/A | ∇f = 0, det(H) < 0 | Horse saddle |

### Convexity Check
```python
# 1D: f''(x) ≥ 0 for all x
# 2D: Hessian H is positive semidefinite
# General: All eigenvalues of H ≥ 0
```

---

## 💻 Python Code Snippets

### Numerical Derivatives
```python
def numerical_derivative(f, x, h=1e-5):
    """Compute derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-5):
    """Compute gradient for multivariable function"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad
```

### Gradient Descent Implementation
```python
def gradient_descent(f, df, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """Basic gradient descent algorithm"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = df(x)
        x_new = x - lr * grad
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)
```

### Common Loss Functions
```python
# Mean Squared Error
def mse_loss(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def mse_gradient(X, y, w):
    y_pred = X @ w
    return X.T @ (y_pred - y) / len(y)

# Logistic Loss  
def logistic_loss(y_true, y_pred):
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
```

### Hessian Computation
```python
def numerical_hessian(f, x, h=1e-5):
    """Compute Hessian matrix numerically"""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Compute second partial derivative
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h  
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
    
    return H
```

---

## 🎯 ML Applications Quick Guide

### Backpropagation
```python
# Forward pass: z = Wx + b, a = σ(z)
# Backward pass: 
# dL/dW = dL/da × da/dz × dz/dW = dL/da × σ'(z) × x^T
# dL/db = dL/da × da/dz × dz/db = dL/da × σ'(z)
```

### Linear Regression Gradients
```python
# Model: y = Xw + b
# Loss: L = ||Xw + b - y||²/(2m)
# Gradients:
# ∂L/∂w = X^T(Xw + b - y)/m  
# ∂L/∂b = sum(Xw + b - y)/m
```

### Common Activation Derivatives
```python
def relu_derivative(x):
    return (x > 0).astype(float)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

---

## 🔧 Debugging Optimization

### Common Issues & Solutions

| Problem | Symptoms | Solutions |
|---------|----------|-----------|
| **Learning rate too high** | Loss oscillates/explodes | Reduce α by factor of 10 |
| **Learning rate too low** | Very slow convergence | Increase α gradually |
| **Poor conditioning** | Slow convergence despite good lr | Use momentum, adaptive methods |
| **Saddle points** | Training stalls | Add noise, use second-order methods |
| **Vanishing gradients** | Deep network doesn't learn | Batch norm, skip connections, better init |

### Gradient Checking
```python
def gradient_check(f, df, x, epsilon=1e-7):
    """Verify analytical gradients against numerical"""
    analytical_grad = df(x)
    numerical_grad = numerical_gradient(f, x, epsilon)
    
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    relative_error = diff / (np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad))
    
    print(f"Absolute difference: {diff:.2e}")
    print(f"Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-5:
        print("✅ Gradient check passed!")
    else:
        print("❌ Gradient check failed!")
```

---

## 📈 Optimization Algorithm Comparison

| Algorithm | Convergence Rate | Memory | Best For |
|-----------|------------------|---------|----------|
| **Gradient Descent** | O(1/k) convex, O(ρᵏ) strongly convex | O(n) | Simple problems |
| **SGD** | O(1/√k) | O(n) | Large datasets |
| **Momentum** | O(1/k²) with acceleration | O(n) | Ill-conditioned problems |
| **Adam** | O(1/√k) empirically | O(n) | Most deep learning |
| **Newton** | Quadratic near minimum | O(n²) | Small-medium problems |
| **L-BFGS** | Superlinear | O(mn) | Medium problems |

---

## 🎨 Visualization Code Templates

### Loss Surface Plotting
```python
def plot_loss_surface(loss_func, x_range, y_range):
    """Create 3D loss surface plot"""
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_func(X, Y)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.set_title('Loss Surface')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Contour Plot')
    
    plt.tight_layout()
    plt.show()
```

### Optimization Path Visualization
```python
def plot_optimization_path(loss_func, path, x_range, y_range):
    """Plot optimization path on loss surface"""
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_func(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='End')
    plt.legend()
    plt.title('Optimization Path')
    plt.show()
```

---

## 🎯 Study Checklist

### Theoretical Understanding
- [ ] Can explain what a derivative represents geometrically
- [ ] Understand chain rule and its connection to backpropagation  
- [ ] Know when gradient descent converges
- [ ] Can classify critical points using second derivatives
- [ ] Understand convexity and its importance

### Practical Skills
- [ ] Can implement gradient descent from scratch
- [ ] Know how to compute gradients numerically
- [ ] Can debug optimization problems
- [ ] Understand common optimization algorithms
- [ ] Can visualize loss landscapes

### ML Applications
- [ ] Know how backpropagation uses chain rule
- [ ] Can derive gradients for common loss functions
- [ ] Understand optimization challenges in deep learning
- [ ] Know when to use different optimization algorithms

---

## 🔗 Quick References

### SymPy for Symbolic Math
```python
import sympy as sp
x, y = sp.symbols('x y')
f = x**2 + 2*x*y + y**2

# Compute derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# Hessian
hessian = sp.Matrix([[sp.diff(f, x, 2), sp.diff(f, x, y)],
                     [sp.diff(f, y, x), sp.diff(f, y, 2)]])
```

### Common Numpy Operations
```python
# Vector operations
np.dot(a, b)              # Dot product
np.linalg.norm(v)         # Vector norm
np.gradient(f, x)         # Numerical gradient

# Matrix operations  
np.linalg.det(A)          # Determinant
np.linalg.eigvals(A)      # Eigenvalues
np.linalg.inv(A)          # Matrix inverse
```

Keep this reference handy while working through the practice projects! 🚀
