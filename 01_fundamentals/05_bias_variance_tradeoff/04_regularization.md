# File: /01_fundamentals/05_bias_variance_tradeoff/04_regularization.md

## Regularization in Machine Learning

### What is Regularization?

Regularization is a technique used in machine learning to prevent overfitting, which occurs when a model learns the noise in the training data instead of the underlying pattern. By adding a penalty for complexity to the loss function, regularization helps to keep the model simpler and more generalizable to unseen data.

### Why Does This Matter?

In machine learning, the goal is to create models that perform well not just on the training data but also on new, unseen data. Overfitting can lead to poor performance in real-world applications, making regularization a crucial step in model training. It helps ensure that the model captures the essential patterns without being overly complex.

### Types of Regularization

1. **L1 Regularization (Lasso Regression)**:
   - Adds a penalty equal to the absolute value of the magnitude of coefficients.
   - Encourages sparsity in the model, meaning it can reduce some coefficients to zero, effectively selecting a simpler model.
   - Useful for feature selection.

   **Example**: If you have a dataset with many features, L1 regularization can help identify the most important features by shrinking the less important ones to zero.

2. **L2 Regularization (Ridge Regression)**:
   - Adds a penalty equal to the square of the magnitude of coefficients.
   - Helps to keep all coefficients small but does not necessarily reduce them to zero.
   - Useful when you want to keep all features but reduce their impact.

   **Example**: In a dataset where all features are potentially important, L2 regularization can help prevent any single feature from dominating the model.

3. **Elastic Net**:
   - Combines both L1 and L2 regularization.
   - Useful when there are multiple features that are correlated with each other.

   **Example**: If you have a dataset with many correlated features, Elastic Net can help balance the benefits of both L1 and L2 regularization.

### Practical Example

Imagine you are building a model to predict house prices based on various features like size, location, and number of bedrooms. Without regularization, your model might fit the training data perfectly but fail to predict prices accurately for new houses. By applying L1 or L2 regularization, you can create a model that generalizes better, leading to more accurate predictions.

### Visual Analogy

Think of regularization like a gardener pruning a tree. If the tree grows too wild and unkempt, it may not bear fruit effectively. By trimming back the branches (regularization), the gardener ensures that the tree focuses its energy on producing fruit rather than growing excessively. Similarly, regularization helps the model focus on the most important features, improving its performance on new data.

### Conclusion

Regularization is a vital concept in machine learning that helps prevent overfitting by adding a penalty for complexity. Understanding and applying regularization techniques can significantly enhance the performance and reliability of your models.

### Mathematical Foundation

#### Key Formulas

**L1 Regularization (Lasso):**
$$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|$$

**L2 Regularization (Ridge):**
$$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$$

**Elastic Net:**
$$J(\theta) = MSE(\theta) + \alpha_1 \sum_{i=1}^{n} |\theta_i| + \alpha_2 \sum_{i=1}^{n} \theta_i^2$$

**Ridge Regression Closed Form:**
$$\hat{\theta} = (X^T X + \alpha I)^{-1} X^T y$$

**Lasso Soft Thresholding:**
$$\theta_i = \begin{cases}
\frac{\theta_i^{OLS} - \alpha/2}{1 + \alpha} & \text{if } \theta_i^{OLS} > \alpha/2 \\
0 & \text{if } |\theta_i^{OLS}| \leq \alpha/2 \\
\frac{\theta_i^{OLS} + \alpha/2}{1 + \alpha} & \text{if } \theta_i^{OLS} < -\alpha/2
\end{cases}$$

#### Solved Examples

##### Example 1: Ridge Regression Calculation

Given: Linear regression problem with features $X$ and target $y$
$$X = \begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix}, \quad y = \begin{pmatrix} 3 \\ 5 \\ 7 \end{pmatrix}$$

Find: Ridge regression coefficients with $\alpha = 0.1$

Solution:
Step 1: Calculate $X^T X$
$$X^T X = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 9 \\ 9 & 29 \end{pmatrix}$$

Step 2: Add regularization term
$$X^T X + \alpha I = \begin{pmatrix} 3 & 9 \\ 9 & 29 \end{pmatrix} + 0.1 \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3.1 & 9 \\ 9 & 29.1 \end{pmatrix}$$

Step 3: Calculate $X^T y$
$$X^T y = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 3 & 4 \end{pmatrix} \begin{pmatrix} 3 \\ 5 \\ 7 \end{pmatrix} = \begin{pmatrix} 15 \\ 49 \end{pmatrix}$$

Step 4: Solve for coefficients
$$\hat{\theta} = \begin{pmatrix} 3.1 & 9 \\ 9 & 29.1 \end{pmatrix}^{-1} \begin{pmatrix} 15 \\ 49 \end{pmatrix}$$

Computing the inverse:
$$\hat{\theta} = \begin{pmatrix} 1.02 \\ 1.98 \end{pmatrix}$$

Result: Ridge coefficients are $\theta_0 = 1.02$, $\theta_1 = 1.98$

##### Example 2: Regularization Strength Effect

Compare models with different $\alpha$ values:
- $\alpha = 0$: Coefficients = [1.0, 2.0, 1.5, 0.8]
- $\alpha = 0.1$: Coefficients = [0.9, 1.8, 1.3, 0.7]  
- $\alpha = 1.0$: Coefficients = [0.5, 0.9, 0.6, 0.3]

Find: Effect of increasing regularization

Solution:
Step 1: Calculate L2 norm for each case
$$||\theta||_2^{(\alpha=0)} = \sqrt{1.0^2 + 2.0^2 + 1.5^2 + 0.8^2} = \sqrt{7.89} = 2.81$$
$$||\theta||_2^{(\alpha=0.1)} = \sqrt{0.9^2 + 1.8^2 + 1.3^2 + 0.7^2} = \sqrt{6.23} = 2.50$$
$$||\theta||_2^{(\alpha=1.0)} = \sqrt{0.5^2 + 0.9^2 + 0.6^2 + 0.3^2} = \sqrt{1.51} = 1.23$$

Step 2: Observe shrinkage effect
- 10× increase in $\alpha$ (0.1 to 1.0) leads to ~50% reduction in coefficient magnitude
- All coefficients shrink proportionally
- Model complexity decreases with higher $\alpha$

Result: Higher regularization leads to smaller coefficients and reduced model complexity.

##### Example 3: Lasso Feature Selection

Given: Dataset with 5 features, Lasso regression with $\alpha = 0.5$
Original coefficients (unregularized): [2.1, 0.3, 1.8, 0.1, 1.2]

Find: Which features survive Lasso selection

Solution:
Step 1: Apply soft thresholding rule
For each coefficient $\theta_i^{OLS}$, apply:
$$\theta_i^{Lasso} = \text{sign}(\theta_i^{OLS}) \cdot \max(0, |\theta_i^{OLS}| - \alpha/2)$$

Step 2: Calculate thresholded coefficients
- $\theta_1$: $\text{sign}(2.1) \cdot \max(0, |2.1| - 0.25) = 1 \cdot 1.85 = 1.85$
- $\theta_2$: $\text{sign}(0.3) \cdot \max(0, |0.3| - 0.25) = 1 \cdot 0.05 = 0.05$
- $\theta_3$: $\text{sign}(1.8) \cdot \max(0, |1.8| - 0.25) = 1 \cdot 1.55 = 1.55$
- $\theta_4$: $\text{sign}(0.1) \cdot \max(0, |0.1| - 0.25) = 1 \cdot 0 = 0$ (eliminated)
- $\theta_5$: $\text{sign}(1.2) \cdot \max(0, |1.2| - 0.25) = 1 \cdot 0.95 = 0.95$

Result: Lasso selects features 1, 2, 3, and 5, eliminating feature 4 due to small coefficient.

**Key Insight:** Lasso automatically performs feature selection by setting small coefficients to zero.

### Practical Exercise

- Experiment with L1 and L2 regularization on a regression problem using a dataset of your choice. Compare the performance of models with and without regularization to see the impact on overfitting.