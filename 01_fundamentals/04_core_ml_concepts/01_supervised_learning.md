# Contents for the file: /01_fundamentals/04_core_ml_concepts/01_supervised_learning.md

# Supervised Learning

## What is Supervised Learning?

Supervised learning is a type of machine learning where an algorithm is trained on a labeled dataset. This means that the input data is paired with the correct output, allowing the model to learn the relationship between the two. The goal is to make predictions or decisions based on new, unseen data.

### Why Does This Matter?

Supervised learning is fundamental in many real-world applications, such as:
- **Email Filtering**: Classifying emails as spam or not spam.
- **Credit Scoring**: Predicting whether a loan applicant is likely to default.
- **Medical Diagnosis**: Assisting doctors in diagnosing diseases based on patient data.

## Key Concepts

### 1. Training and Testing Data

In supervised learning, the dataset is typically divided into two parts:
- **Training Data**: Used to train the model. It contains input-output pairs.
- **Testing Data**: Used to evaluate the model's performance. It contains inputs only, and the model's predictions are compared against the actual outputs.

### 2. Algorithms

Several algorithms can be used for supervised learning, including:
- **Linear Regression**: Used for predicting continuous values (e.g., predicting house prices).
- **Logistic Regression**: Used for binary classification problems (e.g., predicting whether an email is spam).
- **Decision Trees**: A flowchart-like structure used for both classification and regression tasks.
- **Support Vector Machines (SVM)**: Effective for high-dimensional spaces and used for classification tasks.

### 3. Evaluation Metrics

To assess the performance of a supervised learning model, various metrics can be used:
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1 Score**: The harmonic mean of precision and recall, useful for imbalanced datasets.

## Practical Example

### Predicting House Prices

Imagine you want to predict house prices based on features like size, location, and number of bedrooms. You would:
1. **Collect Data**: Gather historical data of houses sold, including their features and prices.
2. **Split Data**: Divide the data into training and testing sets.
3. **Choose an Algorithm**: Use linear regression to model the relationship between features and prices.
4. **Train the Model**: Fit the model using the training data.
5. **Evaluate the Model**: Use the testing data to see how well the model predicts prices.

## Thought Experiment

Consider a scenario where you are trying to predict whether a student will pass or fail an exam based on their study hours and attendance. 
- What features would you consider important?
- How would you collect the data?
- What algorithm might you choose for this task?

## Conclusion

Supervised learning is a powerful tool in machine learning that allows us to make informed predictions based on historical data. By understanding the key concepts and algorithms, you can apply supervised learning techniques to a wide range of problems in various fields.