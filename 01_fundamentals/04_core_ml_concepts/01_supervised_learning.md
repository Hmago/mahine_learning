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

## Mathematical Foundation

### Key Formulas

**Linear Regression:**
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n + \epsilon$$

**Cost Function (Mean Squared Error):**
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

**Logistic Regression:**
$$h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}$$

**Logistic Cost Function:**
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

**Evaluation Metrics:**

- **Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$
- **F1-Score**: $\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### Solved Examples

#### Example 1: Linear Regression - House Price Prediction

Given: Dataset with house features
- House 1: Size = 1000 sq ft, Price = $150,000
- House 2: Size = 1500 sq ft, Price = $200,000  
- House 3: Size = 2000 sq ft, Price = $250,000

Find: Linear regression model and predict price for 1800 sq ft house

Solution:
Step 1: Set up linear model
$$\text{Price} = \beta_0 + \beta_1 \times \text{Size}$$

Step 2: Calculate coefficients using normal equations
Mean values: $\bar{x} = 1500, \bar{y} = 200000$

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

Calculations:
$(1000-1500)(150000-200000) + (1500-1500)(200000-200000) + (2000-1500)(250000-200000)$
$= (-500)(-50000) + (0)(0) + (500)(50000) = 25000000 + 0 + 25000000 = 50000000$

$\sum(x_i - \bar{x})^2 = (-500)^2 + 0^2 + 500^2 = 500000$

$$\beta_1 = \frac{50000000}{500000} = 100$$

$$\beta_0 = \bar{y} - \beta_1\bar{x} = 200000 - 100(1500) = 50000$$

Model: $\text{Price} = 50000 + 100 \times \text{Size}$

Step 3: Predict for 1800 sq ft
$$\text{Price} = 50000 + 100(1800) = 230000$$

Result: Predicted price is $230,000.

#### Example 2: Binary Classification Metrics

Given: Email spam classification results
- True Positives (TP): 85 (correctly identified spam)
- False Positives (FP): 15 (ham classified as spam)  
- True Negatives (TN): 180 (correctly identified ham)
- False Negatives (FN): 20 (spam classified as ham)

Find: All classification metrics

Solution:
Step 1: Calculate Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{85 + 180}{85 + 180 + 15 + 20} = \frac{265}{300} = 0.883$$

Step 2: Calculate Precision
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{85}{85 + 15} = \frac{85}{100} = 0.85$$

Step 3: Calculate Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{85}{85 + 20} = \frac{85}{105} = 0.810$$

Step 4: Calculate F1-Score
$$\text{F1} = 2 \cdot \frac{0.85 \times 0.810}{0.85 + 0.810} = 2 \cdot \frac{0.6885}{1.66} = 0.829$$

Result: Accuracy = 88.3%, Precision = 85%, Recall = 81%, F1 = 82.9%

#### Example 3: Logistic Regression Probability

Given: Logistic regression model for loan approval
$$h_\theta(x) = \frac{1}{1 + e^{-(0.5 + 0.3 \times \text{income} - 0.1 \times \text{debt})}}$$

Find: Approval probability for applicant with income = $50K, debt = $10K

Solution:
Step 1: Calculate linear combination
$$z = 0.5 + 0.3(50) - 0.1(10) = 0.5 + 15 - 1 = 14.5$$

Step 2: Apply sigmoid function
$$h_\theta(x) = \frac{1}{1 + e^{-14.5}} = \frac{1}{1 + e^{-14.5}} \approx \frac{1}{1 + 5.01 \times 10^{-7}} \approx 1.0$$

Result: Approval probability ≈ 100% (very high chance of loan approval).

**Decision Rule**: If $h_\theta(x) \geq 0.5$, approve loan; otherwise, reject.