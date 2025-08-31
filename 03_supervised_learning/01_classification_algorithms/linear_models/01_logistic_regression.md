# Logistic Regression: Your First Classification Algorithm 📈

## What is Logistic Regression? 🤔

Think of logistic regression as a smart way to draw a line that separates two groups. Imagine you're a bouncer at a club deciding who gets in based on age and dress code score. You'd draw a line on a graph where people above the line get in, and people below don't. That's essentially what logistic regression does!

**The Simple Truth:** Despite its name containing "regression," logistic regression is actually a **classification** algorithm. It predicts the **probability** that something belongs to a particular category.

## Why Not Just Use a Straight Line? 🚫📏

Regular linear regression gives us a straight line that can go anywhere from negative infinity to positive infinity. But for classification, we need:
- Outputs between 0 and 1 (probabilities)
- A way to make yes/no decisions
- A smooth transition between categories

**Enter the Sigmoid Function!** 🌊

The sigmoid function takes any number and squashes it between 0 and 1:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Let's see what this looks like
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Input (x)')
plt.ylabel('Probability')
plt.title('The Sigmoid Function: Turning Lines into Probabilities')
plt.grid(True, alpha=0.3)
plt.show()
```

## How Does It Work? ⚙️

### Step 1: Linear Combination
Just like linear regression, we start with a linear combination:
```
z = b₀ + b₁×x₁ + b₂×x₂ + ... + bₙ×xₙ
```

Where:
- `b₀` is the intercept (bias)
- `b₁, b₂, ..., bₙ` are the coefficients (weights)
- `x₁, x₂, ..., xₙ` are the features

### Step 2: Apply Sigmoid
Transform the linear output into a probability:
```
P(y=1) = sigmoid(z) = 1 / (1 + e^(-z))
```

### Step 3: Make Decision
- If P(y=1) ≥ 0.5 → Predict Class 1
- If P(y=1) < 0.5 → Predict Class 0

## Real-World Example: Email Spam Detection 📧

Let's say we want to detect spam emails based on two features:
- Number of exclamation marks
- Number of CAPS words

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data: [exclamation_marks, caps_words]
X = np.array([
    [0, 1],   # Normal email
    [1, 2],   # Normal email  
    [5, 8],   # Spam email
    [7, 12],  # Spam email
    [2, 3],   # Normal email
    [8, 15],  # Spam email
    [1, 1],   # Normal email
    [6, 10]   # Spam email
])

# Labels: 0 = Normal, 1 = Spam
y = np.array([0, 0, 1, 1, 0, 1, 0, 1])

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
new_email = [[3, 5]]  # 3 exclamation marks, 5 caps words
probability = model.predict_proba(new_email)[0][1]  # Probability of spam
prediction = model.predict(new_email)[0]

print(f"Probability of spam: {probability:.2f}")
print(f"Prediction: {'Spam' if prediction == 1 else 'Normal'}")
```

## The Math Behind the Magic 🧮

Don't worry - you don't need to memorize formulas, but understanding the intuition helps!

### Cost Function (Log-Likelihood)
Logistic regression finds the best line by minimizing the "log loss":

```python
def log_loss(y_true, y_pred):
    """
    Calculates how wrong our predictions are
    Lower is better!
    """
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### Finding the Best Parameters
We use gradient descent (like rolling a ball down a hill to find the bottom):

```python
def gradient_descent_step(X, y, weights, learning_rate=0.01):
    """
    One step of gradient descent for logistic regression
    """
    m = len(y)
    
    # Forward pass
    z = X.dot(weights)
    predictions = sigmoid(z)
    
    # Calculate gradients
    gradients = X.T.dot(predictions - y) / m
    
    # Update weights
    weights = weights - learning_rate * gradients
    
    return weights
```

## Simple Implementation from Scratch 💻

Here's a complete logistic regression implementation:

```python
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Add bias term (intercept)
        X = np.column_stack([np.ones(len(X)), X])
        
        # Initialize weights randomly
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X.dot(self.weights)
            predictions = self.sigmoid(z)
            
            # Calculate cost
            cost = -np.mean(y * np.log(predictions + 1e-15) + 
                          (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Calculate gradients
            gradients = X.T.dot(predictions - y) / len(y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict_proba(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return self.sigmoid(X.dot(self.weights))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Example usage
model = SimpleLogisticRegression()
model.fit(X, y)
```

## Advantages & Disadvantages 📊

### ✅ Advantages
- **Fast**: Quick to train and predict
- **Interpretable**: Easy to understand coefficients
- **Probabilistic**: Gives confidence in predictions
- **No assumptions**: About data distribution
- **Baseline**: Great starting point for any classification problem

### ❌ Disadvantages
- **Linear only**: Can't capture complex non-linear relationships
- **Sensitive to outliers**: Extreme values can skew results
- **Feature scaling**: Requires normalization for best results
- **Large datasets**: Can struggle with millions of features

## When to Use Logistic Regression? 🎯

**Perfect for:**
- Quick prototypes and baselines
- When you need to explain predictions
- Linear relationships in your data
- Real-time predictions (very fast)
- Small to medium datasets

**Avoid when:**
- Complex non-linear relationships
- Image or text data (use deep learning instead)
- Very large datasets (consider other algorithms)

## Common Mistakes to Avoid ⚠️

1. **Forgetting to scale features**: Different scales can bias results
2. **Ignoring multicollinearity**: Highly correlated features cause problems
3. **Not checking assumptions**: Linear relationship between features and log-odds
4. **Overfitting with too many features**: Use regularization!

## Next Steps 🚀

Now that you understand logistic regression:
1. Try the interactive notebook: `../../notebooks/01_logistic_regression_lab.ipynb`
2. Read about regularization: `02_regularization.md`
3. Learn about Support Vector Machines: `03_svm.md`
4. Practice with the exercises in `../../exercises/`

## Quick Quiz 🧠

1. What does the sigmoid function do?
2. When would you choose logistic regression over a decision tree?
3. What's the difference between predict() and predict_proba()?
4. Why can't we use regular linear regression for classification?

*Answers are in the exercises folder!*
