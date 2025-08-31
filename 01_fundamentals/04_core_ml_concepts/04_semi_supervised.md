# Semi-Supervised Learning

## Introduction
Semi-supervised learning is a machine learning approach that combines both labeled and unlabeled data to improve learning accuracy. It is particularly useful when acquiring a fully labeled dataset is expensive or time-consuming, while unlabeled data is abundant.

## Why Does This Matter?
In many real-world scenarios, obtaining labeled data can be challenging. For example, in medical imaging, expert annotations are required to label images accurately, which can be costly and time-consuming. Semi-supervised learning allows us to leverage the vast amounts of unlabeled data available, improving model performance without the need for extensive labeling.

## Key Concepts

### 1. Labeled vs. Unlabeled Data
- **Labeled Data**: Data that has been tagged with the correct output. For instance, an image of a cat labeled as "cat."
- **Unlabeled Data**: Data that has no associated output. For example, a collection of images without any labels.

### 2. Learning Process
Semi-supervised learning typically involves two main phases:
- **Pre-training**: The model is initially trained on the labeled data to learn basic patterns.
- **Fine-tuning**: The model is then refined using both labeled and unlabeled data, allowing it to generalize better.

### 3. Techniques
Several techniques are commonly used in semi-supervised learning:
- **Self-training**: The model is trained on labeled data, then used to predict labels for the unlabeled data. The most confident predictions are added to the training set.
- **Co-training**: Two models are trained simultaneously on different views of the data. Each model helps label the unlabeled data for the other.
- **Graph-based methods**: These methods use the relationships between data points to propagate labels from labeled to unlabeled data.

## Practical Example
Consider a scenario where you want to classify emails as "spam" or "not spam." You have a small set of labeled emails (e.g., 100 labeled emails) and a large set of unlabeled emails (e.g., 10,000 emails). By applying semi-supervised learning, you can use the labeled emails to train an initial model and then use that model to label the unlabeled emails. This expanded dataset can significantly improve the model's performance.

## Visual Analogy
Think of semi-supervised learning like a teacher helping a student. The teacher (labeled data) provides guidance on certain topics, while the student (unlabeled data) explores other areas independently. The student learns more effectively by combining both the teacher's guidance and their own exploration.

## Conclusion
Semi-supervised learning is a powerful approach that allows us to make the most of both labeled and unlabeled data. By understanding and applying this technique, we can build more robust models, especially in situations where labeled data is scarce.

## Mathematical Foundation

### Key Formulas

**Semi-Supervised Learning Setup:**
Dataset: $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{l} \cup \{x_j\}_{j=l+1}^{l+u}$

Where:
- $l$ = number of labeled examples
- $u$ = number of unlabeled examples (typically $u \gg l$)

**Self-Training Algorithm:**
1. Train classifier $f$ on labeled data $\mathcal{D}_l$
2. Predict labels for unlabeled data: $\hat{y}_j = f(x_j)$
3. Select confident predictions: $\mathcal{D}_{confident} = \{(x_j, \hat{y}_j) : P(\hat{y}_j|x_j) > \tau\}$
4. Update training set: $\mathcal{D}_l \leftarrow \mathcal{D}_l \cup \mathcal{D}_{confident}$

**Graph-Based SSL (Label Propagation):**
$$\min_{F} \frac{1}{2} \sum_{i,j=1}^{n} w_{ij} ||f_i - f_j||^2$$

Where:
- $f_i$ = predicted label for node $i$
- $w_{ij}$ = edge weight between nodes $i$ and $j$

**Consistency Regularization:**
$$\mathcal{L}_{total} = \mathcal{L}_{supervised} + \lambda \mathcal{L}_{consistency}$$
$$\mathcal{L}_{consistency} = \sum_{i=1}^{u} ||f(x_i) - f(\tilde{x}_i)||^2$$

Where $\tilde{x}_i$ is augmented version of $x_i$.

### Solved Examples

#### Example 1: Self-Training for Text Classification

Given: 
- Labeled data: 50 documents (25 sports, 25 politics)
- Unlabeled data: 500 documents
- Initial classifier accuracy: 85%

Find: Self-training improvement calculation

Solution:
Step 1: Train initial classifier on 50 labeled documents
Assume logistic regression with confidence threshold $\tau = 0.9$

Step 2: Predict on unlabeled data
Out of 500 documents, assume:
- 150 predictions with confidence > 0.9
- Predicted labels: 80 sports, 70 politics

Step 3: Calculate label quality
Assume true accuracy of high-confidence predictions = 92%
Expected correct labels: $150 \times 0.92 = 138$
Expected incorrect labels: $150 \times 0.08 = 12$

Step 4: Update training set
New labeled set size: $50 + 150 = 200$ documents
Label noise rate: $\frac{12}{200} = 6\%$

Step 5: Retrain and evaluate
Expected accuracy improvement: $85\% \rightarrow 89\%$ due to increased training data

#### Example 2: Graph-Based Label Propagation

Given: Graph with 4 nodes, adjacency matrix:
$$W = \begin{bmatrix} 0 & 0.8 & 0.3 & 0 \\ 0.8 & 0 & 0.5 & 0.2 \\ 0.3 & 0.5 & 0 & 0.9 \\ 0 & 0.2 & 0.9 & 0 \end{bmatrix}$$

Labels: Node 1 = Class A (1), Node 4 = Class B (0), Nodes 2,3 = unlabeled

Find: Propagated labels for nodes 2 and 3

Solution:
Step 1: Set up label propagation equations
$$f_i = \frac{\sum_{j} w_{ij} f_j}{\sum_{j} w_{ij}}$$

For unlabeled nodes (keeping labeled nodes fixed):
$$f_2 = \frac{0.8 f_1 + 0.5 f_3 + 0.2 f_4}{0.8 + 0.5 + 0.2} = \frac{0.8(1) + 0.5 f_3 + 0.2(0)}{1.5}$$
$$f_3 = \frac{0.3 f_1 + 0.5 f_2 + 0.9 f_4}{0.3 + 0.5 + 0.9} = \frac{0.3(1) + 0.5 f_2 + 0.9(0)}{1.7}$$

Step 2: Solve iteratively
Initial: $f_2^{(0)} = f_3^{(0)} = 0.5$

Iteration 1:
$$f_2^{(1)} = \frac{0.8 + 0.5(0.5)}{1.5} = \frac{1.05}{1.5} = 0.7$$
$$f_3^{(1)} = \frac{0.3 + 0.5(0.5)}{1.7} = \frac{0.55}{1.7} = 0.324$$

Iteration 2:
$$f_2^{(2)} = \frac{0.8 + 0.5(0.324)}{1.5} = 0.641$$
$$f_3^{(2)} = \frac{0.3 + 0.5(0.7)}{1.7} = 0.382$$

After convergence: $f_2 \approx 0.65$, $f_3 \approx 0.35$

Result: Node 2 likely Class A (65%), Node 3 likely Class B (35%).

#### Example 3: Consistency Regularization Loss

Given: Batch of 4 samples (2 labeled, 2 unlabeled)
- Labeled: $(x_1, y_1 = 1)$, $(x_2, y_2 = 0)$
- Unlabeled: $x_3$, $x_4$
- Predictions: $f(x_1) = 0.9$, $f(x_2) = 0.1$, $f(x_3) = 0.7$, $f(x_4) = 0.3$
- Augmented predictions: $f(\tilde{x_3}) = 0.65$, $f(\tilde{x_4}) = 0.35$

Find: Total loss with $\lambda = 0.5$

Solution:
Step 1: Calculate supervised loss (cross-entropy)
$$\mathcal{L}_{supervised} = -[y_1 \log f(x_1) + (1-y_1) \log(1-f(x_1)) + y_2 \log f(x_2) + (1-y_2) \log(1-f(x_2))]$$
$$\mathcal{L}_{supervised} = -[1 \log(0.9) + 0 \log(0.1) + 0 \log(0.1) + 1 \log(0.9)]$$
$$\mathcal{L}_{supervised} = -2 \log(0.9) = -2(-0.046) = 0.092$$

Step 2: Calculate consistency loss
$$\mathcal{L}_{consistency} = ||f(x_3) - f(\tilde{x_3})||^2 + ||f(x_4) - f(\tilde{x_4})||^2$$
$$\mathcal{L}_{consistency} = (0.7 - 0.65)^2 + (0.3 - 0.35)^2 = 0.0025 + 0.0025 = 0.005$$

Step 3: Calculate total loss
$$\mathcal{L}_{total} = \mathcal{L}_{supervised} + \lambda \mathcal{L}_{consistency} = 0.092 + 0.5(0.005) = 0.0945$$

Result: Total loss incorporates both supervised accuracy and prediction consistency.

### Suggested Exercises
1. **Thought Experiment**: Imagine you have a dataset of images of animals. How would you approach labeling them if you only had a few labeled examples?
2. **Research Task**: Look into real-world applications of semi-supervised learning in fields like healthcare or natural language processing. What challenges do they face, and how do they overcome them?