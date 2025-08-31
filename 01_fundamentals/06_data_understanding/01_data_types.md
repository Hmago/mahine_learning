# Data Types in Machine Learning

## Introduction
In machine learning, understanding different data types is crucial because it influences how we process, analyze, and model the data. Each type of data has its own characteristics and requires specific techniques for handling and analysis.

## Why Does This Matter?
Data types determine the kind of operations we can perform on the data and the algorithms we can use. For example, numerical data can be used in mathematical operations, while categorical data may require encoding before it can be used in models. Understanding data types helps in making informed decisions about data preprocessing and feature engineering.

## Overview of Data Types

### 1. Numerical Data
Numerical data represents measurable quantities and can be divided into two subtypes:
- **Continuous Data**: Can take any value within a range (e.g., height, weight, temperature).
- **Discrete Data**: Can only take specific values (e.g., number of students in a class, number of cars in a parking lot).

**Example**: 
- Continuous: The height of individuals measured in centimeters.
- Discrete: The number of pets owned by a family.

### 2. Categorical Data
Categorical data represents categories or groups and can be further classified into:
- **Nominal Data**: Categories without a specific order (e.g., colors, types of fruits).
- **Ordinal Data**: Categories with a defined order (e.g., ratings from 1 to 5).

**Example**: 
- Nominal: Types of fruits (apple, banana, orange).
- Ordinal: Customer satisfaction ratings (poor, fair, good, excellent).

### 3. Text Data
Text data consists of unstructured data in the form of words or sentences. It requires special techniques for processing, such as tokenization and vectorization.

**Example**: 
- Customer reviews or social media posts.

### 4. Time Series Data
Time series data is a sequence of data points collected or recorded at specific time intervals. It is often used in forecasting and trend analysis.

**Example**: 
- Daily stock prices or monthly sales figures.

### 5. Image Data
Image data consists of pixel values that represent visual information. It is typically processed using techniques from computer vision.

**Example**: 
- Photographs or medical imaging scans.

## Practical Examples
- **Numerical Data**: In a dataset of house prices, the price and square footage are numerical features.
- **Categorical Data**: In a customer dataset, the 'gender' and 'country' columns are categorical features.
- **Text Data**: Analyzing customer feedback to determine sentiment.
- **Time Series Data**: Analyzing temperature readings over a year to identify seasonal trends.
- **Image Data**: Classifying images of animals into different species.

## Conclusion
Understanding data types is fundamental in machine learning as it guides the preprocessing steps and the choice of algorithms. By recognizing the nature of your data, you can apply the appropriate techniques to extract meaningful insights and build effective models.

## Mathematical Foundation

### Key Formulas

**Numerical Data Standardization:**
$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation.

**Categorical Data Encoding:**

**One-Hot Encoding:** For categorical variable with $k$ categories:
$$x_i = \begin{cases} 1 & \text{if category } = i \\ 0 & \text{otherwise} \end{cases}$$

**Label Encoding:** For ordinal data:
$$x_{encoded} = \{0, 1, 2, ..., k-1\}$$

**Text Vectorization (TF-IDF):**
$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

Where:
$$TF(t,d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in } d}$$
$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

**Image Data Normalization:**
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

For pixel values typically in range [0, 255].

### Solved Examples

#### Example 1: Numerical Data Standardization

Given: House prices dataset with the following values (in thousands):
Prices: [200, 350, 180, 420, 290, 310, 250]

Find: Standardized values

Solution:
Step 1: Calculate mean
$$\mu = \frac{200 + 350 + 180 + 420 + 290 + 310 + 250}{7} = \frac{2000}{7} = 285.71$$

Step 2: Calculate standard deviation
$$\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{n-1}}$$
$$\sigma = \sqrt{\frac{(200-285.71)^2 + (350-285.71)^2 + ... + (250-285.71)^2}{6}}$$
$$\sigma = \sqrt{\frac{7345.24 + 4137.53 + 11148.08 + 18076.82 + 18.37 + 592.65 + 1275.51}{6}} = 85.92$$

Step 3: Apply standardization formula
- $z_1 = \frac{200 - 285.71}{85.92} = -0.998$
- $z_2 = \frac{350 - 285.71}{85.92} = 0.748$
- $z_3 = \frac{180 - 285.71}{85.92} = -1.231$
- $z_4 = \frac{420 - 285.71}{85.92} = 1.563$
- $z_5 = \frac{290 - 285.71}{85.92} = 0.050$
- $z_6 = \frac{310 - 285.71}{85.92} = 0.283$
- $z_7 = \frac{250 - 285.71}{85.92} = -0.416$

Result: Standardized prices have mean 0 and standard deviation 1.

#### Example 2: One-Hot Encoding for Categorical Data

Given: Color feature with values: ["Red", "Blue", "Green", "Red", "Blue"]

Find: One-hot encoded representation

Solution:
Step 1: Identify unique categories
Categories: ["Red", "Blue", "Green"] → 3 categories

Step 2: Create binary vectors
- "Red" → [1, 0, 0]
- "Blue" → [0, 1, 0]
- "Green" → [0, 0, 1]

Step 3: Apply encoding to dataset
$$\begin{array}{c|ccc}
\text{Original} & \text{Red} & \text{Blue} & \text{Green} \\
\hline
\text{Red} & 1 & 0 & 0 \\
\text{Blue} & 0 & 1 & 0 \\
\text{Green} & 0 & 0 & 1 \\
\text{Red} & 1 & 0 & 0 \\
\text{Blue} & 0 & 1 & 0
\end{array}$$

Result: 3 binary features replace 1 categorical feature.

#### Example 3: TF-IDF Calculation

Given: Two documents:
- Doc 1: "machine learning is fun"
- Doc 2: "learning algorithms are fun"

Find: TF-IDF vectors for both documents

Solution:
Step 1: Create vocabulary
Vocabulary: ["machine", "learning", "is", "fun", "algorithms", "are"]

Step 2: Calculate TF for each document
Doc 1 TF: machine=0.25, learning=0.25, is=0.25, fun=0.25, algorithms=0, are=0
Doc 2 TF: machine=0, learning=0.25, is=0, fun=0.25, algorithms=0.25, are=0.25

Step 3: Calculate IDF for each term
- $IDF(\text{machine}) = \log\left(\frac{2}{1}\right) = \log(2) = 0.693$
- $IDF(\text{learning}) = \log\left(\frac{2}{2}\right) = \log(1) = 0$
- $IDF(\text{is}) = \log\left(\frac{2}{1}\right) = 0.693$
- $IDF(\text{fun}) = \log\left(\frac{2}{2}\right) = 0$
- $IDF(\text{algorithms}) = \log\left(\frac{2}{1}\right) = 0.693$
- $IDF(\text{are}) = \log\left(\frac{2}{1}\right) = 0.693$

Step 4: Calculate TF-IDF vectors
Doc 1: [0.25×0.693, 0.25×0, 0.25×0.693, 0.25×0, 0, 0] = [0.173, 0, 0.173, 0, 0, 0]
Doc 2: [0, 0.25×0, 0, 0.25×0, 0.25×0.693, 0.25×0.693] = [0, 0, 0, 0, 0.173, 0.173]

Result: TF-IDF vectors capture the importance of words relative to the corpus.

**Data Type Selection Guidelines:**
1. **Numerical**: Use for measurements, counts, ratios
2. **Categorical**: Use for groupings, classifications
3. **Text**: Requires tokenization and vectorization
4. **Time Series**: Requires temporal ordering consideration
5. **Images**: Requires pixel normalization and possibly dimensionality reduction

## Exercises
1. Identify the data types in a given dataset and explain why they are classified as such.
2. Create a small dataset with examples of each data type and describe how you would handle each type in a machine learning project.