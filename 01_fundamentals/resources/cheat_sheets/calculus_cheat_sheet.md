# Contents for the file: /01_fundamentals/resources/cheat_sheets/calculus_cheat_sheet.md

# Calculus Cheat Sheet

## Key Concepts

### 1. Derivatives
- **Definition**: The derivative of a function measures how the function's output changes as its input changes. It represents the slope of the tangent line to the function at a given point.
- **Notation**: 
  - f'(x) or df/dx
- **Rules**:
  - **Power Rule**: d/dx [x^n] = n*x^(n-1)
  - **Product Rule**: d/dx [u*v] = u'v + uv'
  - **Quotient Rule**: d/dx [u/v] = (u'v - uv')/v^2
  - **Chain Rule**: d/dx [f(g(x))] = f'(g(x)) * g'(x)

### 2. Gradients
- **Definition**: The gradient is a vector that contains all the partial derivatives of a function. It points in the direction of the steepest ascent.
- **Notation**: ∇f(x, y) = [∂f/∂x, ∂f/∂y]

### 3. Optimization
- **Objective**: Finding the maximum or minimum values of a function.
- **Critical Points**: Points where the derivative is zero or undefined.
- **Second Derivative Test**:
  - If f''(x) > 0, the function has a local minimum at x.
  - If f''(x) < 0, the function has a local maximum at x.

### 4. Multivariable Calculus
- **Partial Derivatives**: Derivatives of functions with multiple variables, holding other variables constant.
- **Example**: For f(x, y), the partial derivatives are ∂f/∂x and ∂f/∂y.

### 5. Gradient Descent
- **Definition**: An optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent.
- **Update Rule**: 
  - θ = θ - α * ∇J(θ)
  - Where θ represents parameters, α is the learning rate, and J(θ) is the cost function.

## Practical Applications
- **Machine Learning**: Derivatives and gradients are used in training models, particularly in optimization algorithms like gradient descent.
- **Physics**: Calculus is used to model motion, change, and rates of change in physical systems.

## Why Does This Matter?
Understanding calculus is crucial for grasping how machine learning algorithms work, especially in optimization and model training. It provides the mathematical foundation for adjusting model parameters to minimize error and improve predictions.

## Exercises
- Calculate the derivative of the function f(x) = 3x^3 - 5x^2 + 2.
- Use the gradient descent algorithm to find the minimum of the function f(x) = x^2 + 4x + 4, starting from an initial guess of x = 0.