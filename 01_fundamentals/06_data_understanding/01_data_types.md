# Contents for the file: /01_fundamentals/06_data_understanding/01_data_types.md

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

## Exercises
1. Identify the data types in a given dataset and explain why they are classified as such.
2. Create a small dataset with examples of each data type and describe how you would handle each type in a machine learning project.