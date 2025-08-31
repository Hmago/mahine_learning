# Derivatives and Gradients

## Introduction to Derivatives

In calculus, a derivative represents the rate at which a function is changing at any given point. Think of it as a way to measure how steep a curve is at a specific point. If you imagine driving a car along a hilly road, the derivative at any point tells you how steep the hill is at that moment. 

### Why Does This Matter?
Understanding derivatives is crucial in machine learning because they help us optimize models. When we want to minimize a loss function (which measures how well our model is performing), we need to know how to adjust our parameters. Derivatives provide the necessary information to make these adjustments.

### Practical Example
Consider a simple function: 
f(x) = x². 

The derivative of this function, denoted as f'(x), tells us how f(x) changes as x changes. For f(x) = x², the derivative is:
f'(x) = 2x.

This means that at x = 3, the slope of the function is 6 (2 * 3). If we were to plot this, we would see that the curve is steepening as x increases.

## Understanding Gradients

When dealing with functions of multiple variables, we use the concept of gradients. The gradient is a vector that contains all the partial derivatives of a function. It points in the direction of the steepest ascent of the function.

### Why Does This Matter?
In machine learning, we often work with functions that depend on multiple parameters (like weights in a neural network). The gradient helps us understand how to change all parameters simultaneously to minimize our loss function.

### Practical Example
For a function f(x, y) = x² + y², the gradient is:
∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y].

This means that if we are at the point (1, 1), the gradient is (2, 2). This tells us that to decrease the function value, we should move in the opposite direction of the gradient.

## Visual Analogy

Imagine you are standing on a mountain and want to find the quickest way down. The slope of the mountain at your feet represents the derivative. If you look around, the steepest path down is the direction of the gradient. By following this path, you can reach the bottom of the mountain (minimum point) most efficiently.

## Conclusion

Derivatives and gradients are foundational concepts in calculus that play a vital role in optimization problems in machine learning. By understanding how to calculate and interpret them, you can effectively minimize loss functions and improve model performance.

## Practical Exercises

1. **Calculate Derivatives**: Given the function f(x) = 3x³ - 5x² + 2, calculate the derivative f'(x) and evaluate it at x = 2.
   
2. **Gradient Calculation**: For the function f(x, y) = x² + 4y², compute the gradient ∇f at the point (1, 2).

3. **Visualize**: Plot the function f(x) = x² and its derivative on the same graph to see how they relate.

This file serves as an introduction to derivatives and gradients, providing the necessary theoretical background and practical applications relevant to machine learning.

## Mathematical Foundation

### Key Formulas

**Derivative Definition:**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Common Derivatives:**

- Power rule: $\frac{d}{dx}[x^n] = nx^{n-1}$
- Sum rule: $\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$
- Product rule: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
- Chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$

**Partial Derivatives:**
For function $f(x,y)$:
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h,y) - f(x,y)}{h}$$

**Gradient:**
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Solved Examples

#### Example 1: Basic Derivative Calculation

Given: $f(x) = 3x^4 - 5x^2 + 2x - 1$

Find: $f'(x)$ and evaluate at $x = 2$

Solution:
Step 1: Apply power rule to each term
$$f'(x) = \frac{d}{dx}[3x^4] - \frac{d}{dx}[5x^2] + \frac{d}{dx}[2x] - \frac{d}{dx}[1]$$
$$f'(x) = 3(4x^3) - 5(2x) + 2 - 0$$
$$f'(x) = 12x^3 - 10x + 2$$

Step 2: Evaluate at $x = 2$
$$f'(2) = 12(2)^3 - 10(2) + 2 = 12(8) - 20 + 2 = 96 - 20 + 2 = 78$$

Result: The slope of the function at $x = 2$ is 78.

#### Example 2: Gradient Calculation for Multivariable Function

Given: $f(x,y) = x^2y + 3xy^2 - 2y^3$

Find: $\nabla f$ and evaluate at point $(1, 2)$

Solution:
Step 1: Calculate partial derivative with respect to $x$
$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

Step 2: Calculate partial derivative with respect to $y$
$$\frac{\partial f}{\partial y} = x^2 + 6xy - 6y^2$$

Step 3: Form gradient vector
$$\nabla f = \begin{bmatrix} 2xy + 3y^2 \\ x^2 + 6xy - 6y^2 \end{bmatrix}$$

Step 4: Evaluate at $(1, 2)$
$$\nabla f(1,2) = \begin{bmatrix} 2(1)(2) + 3(2)^2 \\ (1)^2 + 6(1)(2) - 6(2)^2 \end{bmatrix} = \begin{bmatrix} 4 + 12 \\ 1 + 12 - 24 \end{bmatrix} = \begin{bmatrix} 16 \\ -11 \end{bmatrix}$$

#### Example 3: Chain Rule Application (Neural Network Context)

Given: Composite function $f(x) = \sigma(wx + b)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$ (sigmoid function)

Find: $\frac{df}{dx}$ (useful for backpropagation)

Solution:
Step 1: Identify inner and outer functions
- Outer function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Inner function: $z = wx + b$

Step 2: Calculate derivative of outer function
$$\sigma'(z) = \frac{d}{dz}\left[\frac{1}{1 + e^{-z}}\right] = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1-\sigma(z))$$

Step 3: Calculate derivative of inner function
$$\frac{dz}{dx} = w$$

Step 4: Apply chain rule
$$\frac{df}{dx} = \sigma'(wx + b) \cdot w = \sigma(wx + b)(1-\sigma(wx + b)) \cdot w$$

Result: This formula is fundamental in training neural networks through gradient descent.