# Calculus & Optimization Practice Guide 🚀

This guide provides hands-on exercises to master calculus concepts essential for ML/AI.

## 🎯 Practice Projects Overview

### 1. **Implement Gradient Descent from Scratch** ⭐⭐⭐
**Goal**: Build gradient descent algorithm without using libraries
**Skills**: Derivatives, optimization, algorithm implementation
**Time**: 2-3 hours

### 2. **Visualize Loss Function Surfaces** ⭐⭐
**Goal**: Create 3D visualizations of optimization landscapes  
**Skills**: Multivariable calculus, data visualization
**Time**: 1-2 hours

### 3. **Find Optimal Parameters for Simple Functions** ⭐⭐
**Goal**: Use calculus to solve optimization problems analytically
**Skills**: Critical points, second derivatives, optimization theory
**Time**: 1-2 hours

### 4. **Project: Linear Regression with Manual Gradient Descent** ⭐⭐⭐⭐
**Goal**: Build complete ML model from mathematical foundations
**Skills**: All calculus concepts + real ML application
**Time**: 3-4 hours

---

## 📋 Project 1: Implement Gradient Descent from Scratch

### Learning Objectives
- Understand how gradient descent works mathematically
- Implement numerical differentiation
- Handle different step sizes and convergence criteria
- Visualize optimization path

### Implementation Steps

#### Step 1: Basic Gradient Descent Function
```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, df, x0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Implement gradient descent algorithm
    
    Parameters:
    f: function to minimize
    df: derivative of function  
    x0: starting point
    learning_rate: step size
    max_iterations: maximum number of iterations
    tolerance: convergence threshold
    
    Returns:
    x_history: array of x values during optimization
    f_history: array of function values during optimization
    """
    # TODO: Implement this function
    # Hints:
    # 1. Initialize arrays to store history
    # 2. Loop until convergence or max iterations
    # 3. Update: x_new = x_old - learning_rate * df(x_old)
    # 4. Check convergence: |x_new - x_old| < tolerance
    
    pass
```

#### Step 2: Test Functions
Create and test your implementation on these functions:

```python
# Function 1: Simple quadratic
def f1(x):
    return x**2 + 4*x + 3

def df1(x):
    return 2*x + 4

# Function 2: More complex function
def f2(x):
    return x**4 - 3*x**3 + 2*x**2 + x + 1

def df2(x):
    return 4*x**3 - 9*x**2 + 4*x + 1

# Function 3: Non-convex function
def f3(x):
    return x*np.sin(x) + 0.1*x**2

def df3(x):
    return np.sin(x) + x*np.cos(x) + 0.2*x
```

#### Step 3: Visualization Function
```python
def visualize_gradient_descent(f, df, x0, learning_rate, x_range):
    """
    Visualize the gradient descent optimization path
    
    TODO: Implement visualization showing:
    1. Function curve
    2. Optimization path (red dots)
    3. Starting point and final point
    4. Gradient vectors at key points
    """
    pass
```

#### Step 4: Learning Rate Experiments
```python
# TODO: Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.3, 0.5]

# For each learning rate:
# 1. Run gradient descent
# 2. Plot convergence curves
# 3. Analyze convergence speed and stability
```

### 🎯 Challenge Extensions
1. **Adaptive Learning Rate**: Implement learning rate scheduling
2. **Momentum**: Add momentum to gradient descent
3. **Multiple Starting Points**: Test convergence from different initial points
4. **2D Functions**: Extend to functions of two variables

### ✅ Success Criteria
- [ ] Gradient descent converges to correct minimum
- [ ] Works with different learning rates
- [ ] Proper visualization of optimization path
- [ ] Handles edge cases (very small/large learning rates)

---

## 📋 Project 2: Visualize Loss Function Surfaces

### Learning Objectives
- Understand loss function landscapes in ML
- Create informative 3D visualizations
- Explore different types of optimization challenges

### Implementation Tasks

#### Task 1: Create Loss Function Visualizations
```python
def create_loss_surface(loss_type='mse'):
    """
    Create 3D visualization of different loss functions
    
    Types to implement:
    - 'mse': Mean squared error
    - 'cross_entropy': Cross-entropy loss
    - 'huber': Huber loss
    - 'custom': Design your own challenging landscape
    """
    # TODO: Implement different loss functions
    # Create meshgrid for parameters
    # Compute loss values
    # Create 3D surface plot with contours
    pass
```

#### Task 2: Optimization Path Visualization
```python
def visualize_optimization_path(loss_func, grad_func, start_points, algorithms):
    """
    Compare different optimization algorithms on same loss surface
    
    algorithms: ['gradient_descent', 'momentum', 'adam']
    start_points: List of (x0, y0) starting points
    """
    # TODO: 
    # 1. Run each algorithm from each starting point
    # 2. Plot 3D surface with optimization paths
    # 3. Use different colors for different algorithms
    # 4. Show convergence comparison
    pass
```

#### Task 3: Interactive Exploration
```python
# TODO: Create interactive plots where users can:
# 1. Click to set starting point
# 2. Adjust learning rate with slider
# 3. See real-time optimization path
# 4. Compare algorithm performance
```

### 🎯 Specific Functions to Visualize

1. **Simple Convex Bowl**
   ```python
   f(x, y) = x² + y² + xy
   ```

2. **Rosenbrock Function** (Banana function)
   ```python
   f(x, y) = (1-x)² + 100(y-x²)²
   ```

3. **Beale Function**
   ```python
   f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
   ```

4. **Neural Network Loss Landscape**
   ```python
   # Simulate loss surface for simple 2-parameter neural network
   # Include local minima and saddle points
   ```

### ✅ Success Criteria
- [ ] Clear 3D surface plots with contours
- [ ] Multiple optimization paths visualized
- [ ] Comparison of different algorithms
- [ ] Interactive elements working
- [ ] Insights about optimization challenges documented

---

## 📋 Project 3: Find Optimal Parameters Analytically

### Learning Objectives
- Apply calculus to solve optimization problems exactly
- Understand relationship between analytical and numerical solutions
- Practice critical point analysis

### Mathematical Problems to Solve

#### Problem 1: Portfolio Optimization
```python
"""
A simple portfolio optimization problem:

You have two assets with expected returns r1, r2 and risks σ1, σ2.
Portfolio return: R = w*r1 + (1-w)*r2
Portfolio risk: Risk = w²*σ1² + (1-w)²*σ2² + 2*w*(1-w)*σ1*σ2*ρ

Find weight w that maximizes: Utility = R - λ*Risk
where λ is risk aversion parameter.

TODO: 
1. Set up the optimization problem
2. Find critical points analytically  
3. Verify with numerical optimization
4. Visualize utility function vs weight
"""

def solve_portfolio_optimization(r1, r2, sigma1, sigma2, rho, lambda_risk):
    # TODO: Implement analytical solution
    pass
```

#### Problem 2: Machine Learning Regularization
```python
"""
Find optimal regularization parameter for ridge regression:

Loss(w) = ||Xw - y||² + λ||w||²

Given X, y, find λ that minimizes cross-validation error.

TODO:
1. Derive analytical solution for w given λ
2. Set up cross-validation error as function of λ  
3. Find optimal λ analytically or numerically
4. Compare with sklearn implementation
"""

def solve_ridge_regularization(X, y):
    # TODO: Implement solution
    pass
```

#### Problem 3: Economic Optimization
```python
"""
Profit maximization problem:

Revenue: R(q) = p*q where p = 100 - 0.5*q (demand curve)
Cost: C(q) = 20 + 10*q + 0.1*q²

Find quantity q that maximizes profit = R(q) - C(q)

TODO:
1. Set up profit function
2. Find critical points
3. Verify it's a maximum (second derivative test)
4. Calculate maximum profit
"""

def solve_profit_maximization():
    # TODO: Implement solution
    pass
```

### 🎯 Advanced Challenges
1. **Constrained Optimization**: Add constraints and use Lagrange multipliers
2. **Multiple Variables**: Extend to functions of 3+ variables
3. **Numerical Verification**: Compare analytical vs numerical solutions
4. **Sensitivity Analysis**: How do optimal values change with parameters?

### ✅ Success Criteria
- [ ] Correct analytical solutions derived
- [ ] Solutions verified numerically
- [ ] Clear mathematical steps documented
- [ ] Practical insights explained
- [ ] Code implementations working

---

## 📋 Project 4: Linear Regression with Manual Gradient Descent

### Learning Objectives
- Apply all calculus concepts to real ML problem
- Implement complete ML pipeline from scratch
- Understand mathematical foundations of linear regression

### Project Overview
Build linear regression entirely from mathematical foundations:
1. Derive cost function and gradients analytically
2. Implement gradient descent optimization
3. Add regularization and analyze effects
4. Compare with sklearn implementation
5. Visualize training process and results

### Implementation Plan

#### Phase 1: Mathematical Foundation
```python
"""
Linear Regression Setup:
- Model: y = X*w + b
- Cost function: J(w,b) = (1/2m) * ||Xw + b - y||²
- Gradients: ∂J/∂w, ∂J/∂b

TODO: Derive gradients analytically first!
"""

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _compute_cost(self, X, y):
        """Compute mean squared error cost"""
        # TODO: Implement MSE cost function
        pass
    
    def _compute_gradients(self, X, y):
        """Compute gradients of cost function"""
        # TODO: Implement gradient computation
        # Return: dw, db
        pass
    
    def fit(self, X, y):
        """Train model using gradient descent"""
        # TODO: Implement training loop
        # 1. Initialize weights and bias
        # 2. For each iteration:
        #    - Compute cost
        #    - Compute gradients  
        #    - Update parameters
        #    - Store cost for plotting
        pass
    
    def predict(self, X):
        """Make predictions"""
        # TODO: Implement prediction
        pass
```

#### Phase 2: Data Generation and Preprocessing
```python
def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=42):
    """
    Generate synthetic regression data for testing
    
    TODO: Create datasets with:
    1. Simple linear relationship
    2. Polynomial relationship (feature engineering)
    3. Multiple features
    4. Different noise levels
    """
    pass

def preprocess_data(X, y):
    """
    Preprocess data for training
    
    TODO: Implement:
    1. Feature normalization/standardization
    2. Train/test split
    3. Add polynomial features if needed
    """
    pass
```

#### Phase 3: Training and Visualization
```python
def train_and_visualize(X, y, learning_rates=[0.001, 0.01, 0.1]):
    """
    Train model with different learning rates and visualize results
    
    TODO: Create visualizations showing:
    1. Training data and fitted line
    2. Cost function convergence
    3. Parameter evolution during training
    4. Effect of different learning rates
    """
    pass

def compare_with_sklearn(X, y):
    """
    Compare custom implementation with sklearn
    
    TODO: 
    1. Train both models
    2. Compare predictions
    3. Compare parameter values
    4. Analyze differences
    """
    pass
```

#### Phase 4: Regularization Extension
```python
class RidgeRegressionFromScratch(LinearRegressionFromScratch):
    def __init__(self, learning_rate=0.01, max_iterations=1000, alpha=1.0):
        super().__init__(learning_rate, max_iterations)
        self.alpha = alpha  # Regularization strength
    
    def _compute_cost(self, X, y):
        """MSE cost + L2 regularization"""
        # TODO: Add regularization term
        pass
    
    def _compute_gradients(self, X, y):
        """Gradients with regularization"""
        # TODO: Add regularization to gradients
        pass
```

### 🎯 Advanced Features to Implement

1. **Feature Engineering**
   ```python
   # TODO: Add polynomial features and analyze overfitting
   ```

2. **Learning Rate Scheduling**
   ```python
   # TODO: Implement adaptive learning rate
   ```

3. **Batch vs Stochastic Gradient Descent**
   ```python
   # TODO: Implement mini-batch gradient descent
   ```

4. **Convergence Analysis**
   ```python
   # TODO: Implement convergence criteria and early stopping
   ```

### 📊 Experiments to Run

1. **Learning Rate Impact**
   - Test rates: [0.001, 0.01, 0.1, 0.5, 1.0]
   - Plot convergence curves
   - Identify optimal range

2. **Regularization Analysis**
   - Test α values: [0, 0.1, 1.0, 10.0, 100.0]
   - Plot training vs validation error
   - Find optimal regularization

3. **Data Size Effects**
   - Train on datasets: [50, 100, 500, 1000, 5000] samples
   - Analyze convergence speed and final accuracy

4. **Feature Engineering Impact**
   - Compare linear vs polynomial features
   - Analyze bias-variance tradeoff

### ✅ Final Deliverables
- [ ] Complete working implementation
- [ ] Comprehensive visualizations
- [ ] Comparison with sklearn (< 1% difference)
- [ ] Regularization analysis
- [ ] Performance benchmarking
- [ ] Written analysis of results

---

## 🎓 Learning Path Recommendations

### Week 1: Foundation Building
1. **Day 1-2**: Complete derivative and gradient exercises in notebook
2. **Day 3-4**: Start Project 1 (Gradient Descent Implementation)
3. **Day 5-7**: Finish Project 1 and document learnings

### Week 2: Visualization and Analysis  
1. **Day 1-3**: Complete Project 2 (Loss Surface Visualization)
2. **Day 4-5**: Work on Project 3 (Analytical Solutions)
3. **Day 6-7**: Review and consolidate understanding

### Week 3: Capstone Application
1. **Day 1-4**: Complete Project 4 (Linear Regression from Scratch)
2. **Day 5-6**: Extensions and advanced features
3. **Day 7**: Final review and portfolio preparation

## 🔧 Setup Requirements

```bash
# Required packages
pip install numpy matplotlib scipy sympy scikit-learn seaborn jupyter

# Optional for interactive plots
pip install plotly ipywidgets
```

## 📚 Additional Resources

### Mathematical References
- **Khan Academy Calculus**: Visual explanations
- **3Blue1Brown Essence of Calculus**: Intuitive understanding
- **Paul's Online Math Notes**: Comprehensive calculus reference

### Coding Resources
- **NumPy Documentation**: Array operations and broadcasting
- **Matplotlib Gallery**: Visualization examples
- **SciPy Optimize**: Professional optimization tools

### ML Context
- **Andrew Ng's Course**: Mathematical foundations
- **Elements of Statistical Learning**: Theoretical background
- **Hands-On ML**: Practical applications

Remember: The goal is deep understanding, not just working code. Take time to understand the mathematics behind each implementation! 🚀
