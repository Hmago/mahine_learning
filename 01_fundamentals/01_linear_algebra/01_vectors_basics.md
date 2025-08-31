# Vectors Basics

## What is a Vector?

A vector is a mathematical object that has both a magnitude (length) and a direction. In machine learning and data science, vectors are used to represent data points in a multi-dimensional space. You can think of a vector as an arrow pointing from one point to another in space.

### Why Does This Matter?

Vectors are fundamental in machine learning because they allow us to represent complex data in a way that algorithms can process. For example, a vector can represent features of an object, such as height, weight, and age, which can then be used to make predictions.

## Types of Vectors

1. **Row Vector**: A row vector is a 1 x n matrix, which means it has one row and multiple columns. For example, [2, 3, 5] is a row vector with three elements.

2. **Column Vector**: A column vector is an n x 1 matrix, which means it has multiple rows and one column. For example, 
   ```
   [2]
   [3]
   [5]
   ```
   is a column vector with three elements.

### Visual Analogy

Imagine you are standing at the origin of a coordinate system (0,0) and you want to describe your position. If you move 3 units to the right and 4 units up, you can represent your position as a vector (3, 4). This vector not only tells you how far to move but also in which direction.

## Vector Operations

### 1. Addition

Vectors can be added together. If you have two vectors A and B, their sum C is calculated by adding their corresponding components.

**Example**:
If A = [1, 2] and B = [3, 4], then:
C = A + B = [1 + 3, 2 + 4] = [4, 6]

### 2. Scalar Multiplication

You can multiply a vector by a scalar (a single number). This operation scales the vector's magnitude without changing its direction.

**Example**:
If A = [2, 3] and you multiply it by 2, you get:
B = 2 * A = [2 * 2, 2 * 3] = [4, 6]

### 3. Dot Product

The dot product is a way to multiply two vectors, resulting in a single number. It is calculated by multiplying corresponding components and then summing those products.

**Example**:
If A = [1, 2] and B = [3, 4], then:
Dot Product = 1*3 + 2*4 = 3 + 8 = 11

### 4. Magnitude

The magnitude (or length) of a vector is calculated using the Pythagorean theorem. For a vector A = [x, y], the magnitude is given by:
Magnitude = √(x² + y²)

**Example**:
For A = [3, 4], the magnitude is:
Magnitude = √(3² + 4²) = √(9 + 16) = √25 = 5

## Practical Applications

- **Data Representation**: In machine learning, each data point can be represented as a vector. For instance, a house can be represented by a vector containing its features like size, number of rooms, and price.

- **Distance Calculation**: Vectors are used to calculate distances between points, which is crucial in clustering algorithms.

## Thought Experiment

Imagine you are trying to classify fruits based on their features like weight, color, and sweetness. Each fruit can be represented as a vector in a multi-dimensional space. How would you visualize the distance between different fruits? What does it mean for two fruits to be "close" together in this space?

## Conclusion

Understanding vectors is essential for grasping more complex concepts in machine learning. They serve as the building blocks for representing data and performing operations that are crucial for algorithm development.

## Mathematical Foundation

### Key Formulas

**Vector Notation:**
- Vector $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$ (column vector)
- Vector $\vec{v} = [v_1, v_2, \ldots, v_n]$ (row vector)

**Vector Operations:**
- Addition: $\vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$
- Scalar multiplication: $k\vec{v} = \begin{bmatrix} kv_1 \\ kv_2 \\ \vdots \\ kv_n \end{bmatrix}$
- Dot product: $\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \ldots + a_nb_n$
- Magnitude: $||\vec{v}|| = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2}$

**Unit Vector:**
$$\hat{v} = \frac{\vec{v}}{||\vec{v}||}$$

### Solved Examples

**Example 1: Vector Addition and Scalar Multiplication**
- Given: $\vec{a} = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}$ and $\vec{b} = \begin{bmatrix} 4 \\ 5 \\ -2 \end{bmatrix}$
- Find: $3\vec{a} + 2\vec{b}$

Solution:
Step 1: Calculate $3\vec{a}$
$$3\vec{a} = 3 \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix} = \begin{bmatrix} 6 \\ -3 \\ 9 \end{bmatrix}$$

Step 2: Calculate $2\vec{b}$
$$2\vec{b} = 2 \begin{bmatrix} 4 \\ 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 8 \\ 10 \\ -4 \end{bmatrix}$$

Step 3: Add the results
$$3\vec{a} + 2\vec{b} = \begin{bmatrix} 6 \\ -3 \\ 9 \end{bmatrix} + \begin{bmatrix} 8 \\ 10 \\ -4 \end{bmatrix} = \begin{bmatrix} 14 \\ 7 \\ 5 \end{bmatrix}$$

**Example 2: Dot Product and Angle Between Vectors**
- Given: $\vec{u} = [3, 4]$ and $\vec{v} = [1, 2]$
- Find: Dot product and angle between vectors

Solution:
Step 1: Calculate dot product
$$\vec{u} \cdot \vec{v} = 3(1) + 4(2) = 3 + 8 = 11$$

Step 2: Calculate magnitudes
$$||\vec{u}|| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
$$||\vec{v}|| = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236$$

Step 3: Calculate angle using $\cos \theta = \frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| \cdot ||\vec{v}||}$
$$\cos \theta = \frac{11}{5 \times \sqrt{5}} = \frac{11}{5\sqrt{5}} = \frac{11\sqrt{5}}{25} \approx 0.983$$
$$\theta = \arccos(0.983) \approx 10.8°$$

**Example 3: Unit Vector Calculation**
- Given: $\vec{w} = [6, -8]$
- Find: Unit vector in the direction of $\vec{w}$

Solution:
Step 1: Calculate magnitude
$$||\vec{w}|| = \sqrt{6^2 + (-8)^2} = \sqrt{36 + 64} = \sqrt{100} = 10$$

Step 2: Calculate unit vector
$$\hat{w} = \frac{\vec{w}}{||\vec{w}||} = \frac{[6, -8]}{10} = [0.6, -0.8]$$

Verification: $||\hat{w}|| = \sqrt{0.6^2 + (-0.8)^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$ ✓

---

This file provides a comprehensive introduction to vectors, their operations, and their significance in machine learning, tailored for beginners.