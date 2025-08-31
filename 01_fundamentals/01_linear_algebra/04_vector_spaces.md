# Vector Spaces

## Introduction to Vector Spaces

A vector space is a fundamental concept in linear algebra that provides a framework for understanding how vectors can be combined and manipulated. It consists of a set of vectors, which can be added together and multiplied by scalars, satisfying certain properties. Understanding vector spaces is crucial for many applications in machine learning, as they form the basis for representing data and performing operations on it.

## Key Properties of Vector Spaces

1. **Closure**: If you take any two vectors in a vector space and add them together, the result is also a vector in that space. Similarly, if you multiply a vector by a scalar, the result is still within the space.

2. **Associativity**: Vector addition is associative. This means that for any vectors **u**, **v**, and **w**, the equation (**u** + **v**) + **w** = **u** + (**v** + **w**) holds true.

3. **Commutativity**: Vector addition is commutative. For any vectors **u** and **v**, **u** + **v** = **v** + **u**.

4. **Identity Element**: There exists a zero vector (denoted as **0**) in the vector space such that for any vector **v**, **v** + **0** = **v**.

5. **Inverse Element**: For every vector **v**, there exists a vector **-v** such that **v** + **(-v)** = **0**.

6. **Distributive Property**: Scalar multiplication distributes over vector addition. For any scalar **c** and vectors **u** and **v**, **c**(**u** + **v**) = **cu** + **cv**.

7. **Scalar Multiplication**: Scalar multiplication is associative and has an identity element. For any scalar **c** and **1**, **c**(1**v**) = **v**.

## Real-World Applications

Vector spaces are used extensively in machine learning and data science. Here are a few examples:

- **Data Representation**: In machine learning, data points are often represented as vectors in a high-dimensional space. Each feature of the data corresponds to a dimension in this space.

- **Dimensionality Reduction**: Techniques like Principal Component Analysis (PCA) rely on the concept of vector spaces to reduce the number of dimensions while preserving the variance in the data.

- **Linear Transformations**: Many algorithms, such as linear regression, involve transforming data points in vector spaces to find relationships between variables.

## Why Does This Matter?

Understanding vector spaces is essential for grasping more complex concepts in linear algebra and machine learning. It helps you visualize and manipulate data effectively, which is crucial for building and optimizing machine learning models.

## Practical Exercise

1. **Thought Experiment**: Imagine you have a dataset with two features: height and weight. Visualize this data as points in a 2D space. What would the vector representation of a point look like? How would you represent the average height and weight?

2. **Explore**: Take a simple dataset and plot it in a 2D space. Identify the vectors representing each data point and practice adding and scaling these vectors.

## Conclusion

Vector spaces provide a powerful framework for understanding and manipulating data in machine learning. By mastering this concept, you will be better equipped to tackle more advanced topics in linear algebra and apply them to real-world problems.

## Mathematical Foundation

### Key Formulas

**Vector Space Definition:**
A set $V$ with operations addition $(+)$ and scalar multiplication $(\cdot)$ is a vector space if it satisfies:

**Closure Properties:**
- $\vec{u} + \vec{v} \in V$ for all $\vec{u}, \vec{v} \in V$
- $c\vec{v} \in V$ for all $c \in \mathbb{R}$ and $\vec{v} \in V$

**Vector Space Axioms:**
1. Commutativity: $\vec{u} + \vec{v} = \vec{v} + \vec{u}$
2. Associativity: $(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$
3. Zero vector: $\vec{v} + \vec{0} = \vec{v}$
4. Additive inverse: $\vec{v} + (-\vec{v}) = \vec{0}$
5. Scalar distributivity: $c(\vec{u} + \vec{v}) = c\vec{u} + c\vec{v}$
6. Vector distributivity: $(c + d)\vec{v} = c\vec{v} + d\vec{v}$
7. Scalar associativity: $c(d\vec{v}) = (cd)\vec{v}$
8. Scalar identity: $1\vec{v} = \vec{v}$

**Linear Independence:**
Vectors $\vec{v_1}, \vec{v_2}, \ldots, \vec{v_n}$ are linearly independent if:
$$c_1\vec{v_1} + c_2\vec{v_2} + \cdots + c_n\vec{v_n} = \vec{0}$$
implies $c_1 = c_2 = \cdots = c_n = 0$

**Basis and Dimension:**
- A basis is a linearly independent set that spans the vector space
- Dimension = number of vectors in any basis

### Solved Examples

#### Example 1: Verifying Vector Space Properties

Given: Set $V = \mathbb{R}^2$ with standard addition and scalar multiplication

Verify: Closure under addition

Solution:
Let $\vec{u} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$ be any two vectors in $\mathbb{R}^2$

Step 1: Perform addition
$$\vec{u} + \vec{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

Step 2: Check if result is in $\mathbb{R}^2$
Since $u_1 + v_1 \in \mathbb{R}$ and $u_2 + v_2 \in \mathbb{R}$, we have $\vec{u} + \vec{v} \in \mathbb{R}^2$ ✓

#### Example 2: Linear Independence Test

Given: Vectors $\vec{v_1} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $\vec{v_2} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$, $\vec{v_3} = \begin{bmatrix} 5 \\ 10 \end{bmatrix}$

Determine: Whether these vectors are linearly independent

Solution:
Step 1: Set up linear combination equation
$$c_1\vec{v_1} + c_2\vec{v_2} + c_3\vec{v_3} = \vec{0}$$
$$c_1\begin{bmatrix} 1 \\ 2 \end{bmatrix} + c_2\begin{bmatrix} 3 \\ 4 \end{bmatrix} + c_3\begin{bmatrix} 5 \\ 10 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Step 2: Form system of equations
$$c_1 + 3c_2 + 5c_3 = 0$$
$$2c_1 + 4c_2 + 10c_3 = 0$$

Step 3: Solve system
From equation 1: $c_1 = -3c_2 - 5c_3$
Substituting into equation 2: $2(-3c_2 - 5c_3) + 4c_2 + 10c_3 = 0$
This gives: $-6c_2 - 10c_3 + 4c_2 + 10c_3 = 0$
Simplifying: $-2c_2 = 0$, so $c_2 = 0$

Therefore: $c_1 = -5c_3$ and $c_2 = 0$

Since we can choose $c_3 = 1$, giving $c_1 = -5$, $c_2 = 0$, $c_3 = 1$ (non-trivial solution), the vectors are **linearly dependent**.

#### Example 3: Finding Basis and Dimension

Given: Subspace $W = \text{span}\{\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}\}$

Find: Basis and dimension of $W$

Solution:
Step 1: Form matrix with vectors as columns
$$A = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & -1 \end{bmatrix}$$

Step 2: Row reduce to find linearly independent columns
$$\begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & -1 \end{bmatrix} \rightarrow \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

Step 3: Identify pivot columns
Pivots in columns 1 and 2, so the first two vectors form a basis.

Result: 
- Basis: $\{\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}\}$
- Dimension: $\dim(W) = 2$