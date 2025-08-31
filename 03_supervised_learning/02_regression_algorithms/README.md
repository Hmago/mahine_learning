# Regression Algorithms: Predicting Numbers 📊

## What is Regression? 🤔

Imagine you're a real estate agent trying to predict house prices. You look at factors like:
- Size of the house
- Number of bedrooms  
- Location quality
- Age of the house

Regression helps you find the mathematical relationship between these **input features** and the **target value** (price). Instead of predicting categories like classification, regression predicts **continuous numbers**.

## Regression vs Classification: What's the Difference? 🔄

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Output** | Categories (Dog, Cat, Bird) | Numbers (Price, Temperature, Score) |
| **Examples** | Email spam detection | House price prediction |
| **Evaluation** | Accuracy, Precision, Recall | MSE, MAE, R² |
| **Decision Boundary** | Separates classes | Fits a curve through data |

## Types of Regression Problems 📈

### 1. Simple Linear Regression
One input feature predicting one output:
```
House Price = a × House Size + b
```

### 2. Multiple Linear Regression  
Multiple input features predicting one output:
```
House Price = a₁×Size + a₂×Bedrooms + a₃×Location + a₄×Age + b
```

### 3. Polynomial Regression
Non-linear relationships:
```
House Price = a₁×Size + a₂×Size² + a₃×Size³ + b
```

### 4. Non-linear Regression
Complex relationships that can't be expressed as simple equations

## Linear Regression: The Foundation 🏗️

Linear regression finds the **best line** through your data points. But what does "best" mean?

### The Least Squares Approach

Imagine drawing a line through scattered points. Some points will be above the line, some below. Linear regression finds the line that minimizes the **sum of squared errors**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple example: Ice cream sales vs Temperature
temperature = np.array([20, 25, 30, 35, 40, 45]).reshape(-1, 1)
ice_cream_sales = np.array([50, 80, 120, 150, 200, 250])

# Train linear regression
model = LinearRegression()
model.fit(temperature, ice_cream_sales)

# Make predictions
temp_range = np.linspace(15, 50, 100).reshape(-1, 1)
predictions = model.predict(temp_range)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(temperature, ice_cream_sales, color='red', s=100, label='Actual Sales')
plt.plot(temp_range, predictions, color='blue', linewidth=2, label='Regression Line')

# Show the "errors" (residuals)
train_predictions = model.predict(temperature)
for i in range(len(temperature)):
    plt.plot([temperature[i], temperature[i]], 
             [ice_cream_sales[i], train_predictions[i]], 
             'gray', linestyle='--', alpha=0.7)

plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales ($)')
plt.title('Linear Regression: Ice Cream Sales vs Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

# Print the equation
print(f"Equation: Sales = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}")
print(f"R² Score: {model.score(temperature, ice_cream_sales):.3f}")

plt.show()
```

## Understanding Linear Regression Mathematically 🧮

### The Normal Equation (Analytical Solution)

For simple problems, we can solve linear regression exactly:

```python
def linear_regression_normal_equation(X, y):
    """
    Solve linear regression using the normal equation
    θ = (X^T × X)^(-1) × X^T × y
    """
    # Add bias term (column of ones)
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation
    XtX = X_with_bias.T.dot(X_with_bias)
    Xty = X_with_bias.T.dot(y)
    theta = np.linalg.inv(XtX).dot(Xty)
    
    return theta

# Test our implementation
theta = linear_regression_normal_equation(temperature.flatten(), ice_cream_sales)
print(f"Our implementation - Intercept: {theta[0]:.2f}, Slope: {theta[1]:.2f}")
print(f"Sklearn implementation - Intercept: {model.intercept_:.2f}, Slope: {model.coef_[0]:.2f}")
```

### Gradient Descent (Iterative Solution)

For larger problems, we use gradient descent:

```python
def linear_regression_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Solve linear regression using gradient descent
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Initialize weights randomly
    theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
    m = len(y)
    
    costs = []
    
    for i in range(iterations):
        # Forward pass
        predictions = X_with_bias.dot(theta)
        
        # Calculate cost (Mean Squared Error)
        cost = np.mean((predictions - y) ** 2)
        costs.append(cost)
        
        # Calculate gradients
        gradients = (2/m) * X_with_bias.T.dot(predictions - y)
        
        # Update weights
        theta -= learning_rate * gradients
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.2f}")
    
    return theta, costs

# Train using gradient descent
theta_gd, costs = linear_regression_gradient_descent(
    temperature.flatten(), ice_cream_sales, learning_rate=0.0001, iterations=1000
)

# Plot cost function
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent: Cost Function Minimization')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Gradient descent result - Intercept: {theta_gd[0]:.2f}, Slope: {theta_gd[1]:.2f}")
```

## Multiple Linear Regression: Real Estate Example 🏠

Let's predict house prices using multiple features:

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create realistic house data
np.random.seed(42)
n_houses = 200

house_data = {
    'Size_SqFt': np.random.normal(2000, 500, n_houses),
    'Bedrooms': np.random.randint(1, 6, n_houses),
    'Bathrooms': np.random.normal(2.5, 0.8, n_houses),
    'Age_Years': np.random.randint(0, 50, n_houses),
    'Garage_Cars': np.random.randint(0, 4, n_houses),
    'Distance_to_City': np.random.normal(15, 8, n_houses)
}

# Create realistic price based on features
house_data['Price'] = (
    house_data['Size_SqFt'] * 150 +
    house_data['Bedrooms'] * 10000 +
    house_data['Bathrooms'] * 15000 -
    house_data['Age_Years'] * 1000 +
    house_data['Garage_Cars'] * 8000 -
    house_data['Distance_to_City'] * 2000 +
    np.random.normal(0, 20000, n_houses)  # Add noise
)

house_df = pd.DataFrame(house_data)

# Prepare data
X_houses = house_df.drop('Price', axis=1)
y_houses = house_df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X_houses, y_houses, test_size=0.2, random_state=42
)

# Train multiple linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: ${mse:,.0f}")
print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"R² Score: {r2:.3f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_houses.columns,
    'Coefficient': lr.coef_,
    'Abs_Coefficient': np.abs(lr.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance)

# Visualize predictions vs actual
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Predictions vs Actual (R² = {r2:.3f})')

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

## Polynomial Regression: Capturing Curves 📈

Sometimes the relationship isn't a straight line:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate curved data
np.random.seed(42)
X_curve = np.linspace(0, 4, 50).reshape(-1, 1)
y_curve = 2 * X_curve.flatten()**2 - 3 * X_curve.flatten() + 1 + np.random.normal(0, 2, 50)

# Compare different polynomial degrees
degrees = [1, 2, 3, 5, 8]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

X_plot = np.linspace(0, 4, 100).reshape(-1, 1)

for i, degree in enumerate(degrees):
    # Create polynomial features
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(X_curve, y_curve)
    y_plot = poly_model.predict(X_plot)
    
    # Plot
    axes[i].scatter(X_curve, y_curve, alpha=0.7, label='Data')
    axes[i].plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {degree}')
    axes[i].set_title(f'Polynomial Degree {degree}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot
axes[5].remove()
plt.tight_layout()
plt.show()

# Show the danger of overfitting
degrees_test = range(1, 16)
train_scores = []
val_scores = []

for degree in degrees_test:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Training score
    poly_model.fit(X_curve, y_curve)
    train_score = poly_model.score(X_curve, y_curve)
    train_scores.append(train_score)
    
    # Validation score (using same data for simplicity)
    val_score = r2_score(y_curve, poly_model.predict(X_curve))
    val_scores.append(val_score)

plt.figure(figsize=(10, 6))
plt.plot(degrees_test, train_scores, 'o-', label='Training R²')
plt.plot(degrees_test, val_scores, 'o-', label='Validation R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Overfitting in Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Regularized Regression: Preventing Overfitting 🛡️

### Ridge Regression (L2 Regularization)

Ridge regression adds a penalty for large coefficients:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Create data with many features (some irrelevant)
X_many_features, y_target = make_regression(
    n_samples=100, n_features=20, n_informative=5, 
    noise=10, random_state=42
)

# Split and scale
X_train_mf, X_test_mf, y_train_mf, y_test_mf = train_test_split(
    X_many_features, y_target, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mf)
X_test_scaled = scaler.transform(X_test_mf)

# Compare regular vs Ridge regression
regular_lr = LinearRegression()
ridge_lr = Ridge(alpha=1.0)

regular_lr.fit(X_train_scaled, y_train_mf)
ridge_lr.fit(X_train_scaled, y_train_mf)

# Evaluate both
reg_train_score = regular_lr.score(X_train_scaled, y_train_mf)
reg_test_score = regular_lr.score(X_test_scaled, y_test_mf)

ridge_train_score = ridge_lr.score(X_train_scaled, y_train_mf)
ridge_test_score = ridge_lr.score(X_test_scaled, y_test_mf)

print("Regular Linear Regression:")
print(f"  Training R²: {reg_train_score:.3f}")
print(f"  Test R²: {reg_test_score:.3f}")
print(f"  Difference: {reg_train_score - reg_test_score:.3f}")

print("\nRidge Regression:")
print(f"  Training R²: {ridge_train_score:.3f}")
print(f"  Test R²: {ridge_test_score:.3f}")
print(f"  Difference: {ridge_train_score - ridge_test_score:.3f}")

# Visualize coefficient shrinkage
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(regular_lr.coef_)), regular_lr.coef_)
plt.title('Regular Linear Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.subplot(1, 2, 2)
plt.bar(range(len(ridge_lr.coef_)), ridge_lr.coef_)
plt.title('Ridge Regression Coefficients (Shrunk)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.show()
```

### Lasso Regression (L1 Regularization)

Lasso can set some coefficients to exactly zero, effectively selecting features:

```python
from sklearn.linear_model import Lasso

# Train Lasso regression
lasso_lr = Lasso(alpha=1.0)
lasso_lr.fit(X_train_scaled, y_train_mf)

lasso_train_score = lasso_lr.score(X_train_scaled, y_train_mf)
lasso_test_score = lasso_lr.score(X_test_scaled, y_test_mf)

print("Lasso Regression:")
print(f"  Training R²: {lasso_train_score:.3f}")
print(f"  Test R²: {lasso_test_score:.3f}")
print(f"  Features selected: {np.sum(lasso_lr.coef_ != 0)} out of {len(lasso_lr.coef_)}")

# Compare all three approaches
methods = ['Regular', 'Ridge', 'Lasso']
train_scores = [reg_train_score, ridge_train_score, lasso_train_score]
test_scores = [reg_test_score, ridge_test_score, lasso_test_score]

plt.figure(figsize=(10, 6))
x_pos = np.arange(len(methods))
width = 0.35

plt.bar(x_pos - width/2, train_scores, width, label='Training R²', alpha=0.8)
plt.bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.8)

plt.xlabel('Regression Method')
plt.ylabel('R² Score')
plt.title('Comparison of Regularization Methods')
plt.xticks(x_pos, methods)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (train, test) in enumerate(zip(train_scores, test_scores)):
    plt.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center')
    plt.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center')

plt.show()
```

## Model Evaluation: How Good is Good? 📏

### Key Regression Metrics

```python
def regression_metrics_explained(y_true, y_pred):
    """
    Calculate and explain regression metrics
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error  
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"  → Penalizes large errors heavily")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  → Same units as target variable")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  → Average absolute difference")
    print(f"R² Score: {r2:.3f}")
    print(f"  → Fraction of variance explained (higher = better)")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Test on our house price predictions
metrics = regression_metrics_explained(y_test, y_pred)
```

### Interpreting R² Score 📊

R² tells you what fraction of the variance in your target variable is explained by your model:

- **R² = 1.0**: Perfect predictions (very rare in real world)
- **R² = 0.8**: Explains 80% of variance (excellent)
- **R² = 0.5**: Explains 50% of variance (decent)
- **R² = 0.0**: No better than predicting the mean
- **R² < 0**: Worse than predicting the mean (very bad!)

```python
# Visualize what different R² values mean
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

r2_examples = [0.95, 0.7, 0.3, 0.05]

for i, target_r2 in enumerate(r2_examples):
    # Generate data with specific R² 
    X_example = np.random.randn(50, 1)
    y_true = 2 * X_example.flatten() + 1
    
    # Add noise to achieve target R²
    noise_level = np.sqrt((1 - target_r2) / target_r2) * np.std(y_true)
    y_noisy = y_true + np.random.normal(0, noise_level, len(y_true))
    
    # Fit model
    lr_example = LinearRegression()
    lr_example.fit(X_example, y_noisy)
    y_pred_example = lr_example.predict(X_example)
    actual_r2 = lr_example.score(X_example, y_noisy)
    
    # Plot
    axes[i].scatter(X_example, y_noisy, alpha=0.7, label='Data')
    axes[i].plot(X_example, y_pred_example, 'r-', linewidth=2, label='Fit')
    axes[i].set_title(f'R² = {actual_r2:.2f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Advanced Regression Techniques 🚀

### 1. Support Vector Regression (SVR)

SVR uses the same idea as SVM but for regression:

```python
from sklearn.svm import SVR

# SVR with different kernels
svr_linear = SVR(kernel='linear', C=1.0)
svr_rbf = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_poly = SVR(kernel='poly', C=1.0, degree=2)

models = [
    ('Linear SVR', svr_linear),
    ('RBF SVR', svr_rbf), 
    ('Polynomial SVR', svr_poly)
]

for name, model in models:
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f"{name}: R² = {score:.3f}")
```

### 2. Random Forest Regression

Trees aren't just for classification!

```python
from sklearn.ensemble import RandomForestRegressor

# Random Forest for regression
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_regressor.fit(X_train, y_train)  # No scaling needed for trees!
rf_score = rf_regressor.score(X_test, y_test)

print(f"Random Forest R²: {rf_score:.3f}")

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': X_houses.columns,
    'Importance': rf_regressor.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)
```

### 3. Gradient Boosting Regression

```python
from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting
gb_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_regressor.fit(X_train, y_train)
gb_score = gb_regressor.score(X_test, y_test)

print(f"Gradient Boosting R²: {gb_score:.3f}")
```

## Regression Algorithm Comparison 🏁

```python
# Compare all regression methods
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=1.0)
}

comparison_results = {}

for name, model in regressors.items():
    if name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR (RBF)']:
        # These need scaled features
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        # Tree-based models don't need scaling
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    comparison_results[name] = scores
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Visualize comparison
plt.figure(figsize=(12, 8))
model_names = list(comparison_results.keys())
bp = plt.boxplot([comparison_results[name] for name in model_names], 
                labels=model_names, patch_artist=True)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('R² Score')
plt.title('Regression Algorithm Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Feature Engineering for Regression 🛠️

Good features can make a huge difference in regression performance:

```python
def engineer_regression_features(df):
    """
    Create new features that might help regression models
    """
    df_new = df.copy()
    
    # 1. Interaction features
    if 'Size_SqFt' in df.columns and 'Bedrooms' in df.columns:
        df_new['Size_per_Bedroom'] = df['Size_SqFt'] / (df['Bedrooms'] + 1)
    
    # 2. Polynomial features for key variables
    if 'Size_SqFt' in df.columns:
        df_new['Size_Squared'] = df['Size_SqFt'] ** 2
        df_new['Size_Log'] = np.log(df['Size_SqFt'] + 1)
    
    # 3. Binning continuous variables
    if 'Age_Years' in df.columns:
        df_new['Age_Category'] = pd.cut(df['Age_Years'], 
                                       bins=[0, 5, 15, 30, 50], 
                                       labels=['New', 'Recent', 'Mature', 'Old'])
        # Convert to dummy variables
        age_dummies = pd.get_dummies(df_new['Age_Category'], prefix='Age')
        df_new = pd.concat([df_new, age_dummies], axis=1)
        df_new = df_new.drop('Age_Category', axis=1)
    
    # 4. Ratios and derived features
    if 'Bathrooms' in df.columns and 'Bedrooms' in df.columns:
        df_new['Bath_to_Bed_Ratio'] = df['Bathrooms'] / (df['Bedrooms'] + 1)
    
    return df_new

# Apply feature engineering
X_engineered = engineer_regression_features(house_df.drop('Price', axis=1))
print(f"Original features: {X_houses.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")

# Test improvement
lr_engineered = LinearRegression()
scores_original = cross_val_score(lr, X_houses, y_houses, cv=5, scoring='r2')
scores_engineered = cross_val_score(lr_engineered, X_engineered, y_houses, cv=5, scoring='r2')

print(f"Original features R²: {scores_original.mean():.3f}")
print(f"Engineered features R²: {scores_engineered.mean():.3f}")
print(f"Improvement: {scores_engineered.mean() - scores_original.mean():.3f}")
```

## Handling Real-World Regression Challenges 🌍

### 1. Non-Linear Relationships

```python
# When linear models fail, try:

# Option 1: Polynomial features
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# Option 2: Tree-based models
tree_regressor = RandomForestRegressor()

# Option 3: Kernel methods  
svr_regressor = SVR(kernel='rbf')
```

### 2. Outliers

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

# Robust regression methods
huber = HuberRegressor(epsilon=1.35)  # Less sensitive to outliers
ransac = RANSACRegressor(random_state=42)  # Ignores outliers completely

# Compare on data with outliers
X_outliers = X_train.copy()
y_outliers = y_train.copy()

# Add some outliers
outlier_indices = np.random.choice(len(y_outliers), 5)
y_outliers.iloc[outlier_indices] *= 3  # Make some prices 3x higher

models_robust = {
    'Linear': LinearRegression(),
    'Huber': huber,
    'RANSAC': ransac
}

for name, model in models_robust.items():
    model.fit(X_outliers, y_outliers)
    score = model.score(X_test, y_test)  # Test on clean data
    print(f"{name} (with outliers): R² = {score:.3f}")
```

### 3. Multicollinearity

When features are highly correlated:

```python
# Check correlation matrix
correlation_matrix = X_houses.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# High correlation (> 0.8) can cause problems
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                correlation_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print("Highly correlated feature pairs:")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} - {feat2}: {corr:.3f}")
else:
    print("No highly correlated features found")
```

## Assumptions of Linear Regression 📋

Linear regression makes several assumptions. Let's check them:

### 1. Linearity
```python
# Check if relationship is linear
from scipy.stats import pearsonr

for feature in X_houses.columns:
    corr, p_value = pearsonr(X_houses[feature], y_houses)
    print(f"{feature}: correlation = {corr:.3f}, p-value = {p_value:.3f}")
```

### 2. Independence of Errors
```python
# Check if residuals are independent (no patterns)
residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=20, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

# Q-Q plot for normality
from scipy import stats
plt.subplot(1, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normal Distribution)')

plt.tight_layout()
plt.show()
```

### 3. Homoscedasticity (Constant Variance)
```python
# Residuals should have constant variance across predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, np.abs(residuals), alpha=0.7)
plt.xlabel('Predicted Values')
plt.ylabel('Absolute Residuals')
plt.title('Checking Homoscedasticity')
plt.grid(True, alpha=0.3)
plt.show()

# If variance increases with predictions, consider:
# - Log transformation of target
# - Weighted least squares
# - Different model altogether
```

## When to Use Different Regression Algorithms 🎯

### Linear Regression
**Use when:**
- Relationship is approximately linear
- You need interpretability
- Quick baseline model
- Small to medium datasets

### Ridge Regression  
**Use when:**
- Many features with multicollinearity
- Want to keep all features but shrink coefficients
- Regularization is needed

### Lasso Regression
**Use when:**
- Many features, some irrelevant
- Want automatic feature selection
- Sparse solutions desired

### Tree-Based Regression
**Use when:**
- Non-linear relationships
- Feature interactions important
- Mixed data types
- No assumptions about distribution

### SVR
**Use when:**
- Complex non-linear relationships
- Robust to outliers needed
- High-dimensional data

## Complete Regression Project Example 🏗️

Let's build an end-to-end regression system:

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

# Define multiple scoring metrics
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

# Create preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Identify feature types
numeric_features = X_houses.select_dtypes(include=[np.number]).columns
categorical_features = X_houses.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create full pipeline
regression_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Evaluate with cross-validation
cv_results = cross_validate(
    regression_pipeline, X_houses, y_houses, 
    cv=5, scoring=scoring, return_train_score=True
)

print("Cross-validation results:")
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric}:")
    print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std()*2:.3f})")
    print(f"  Test:  {test_scores.mean():.3f} (+/- {test_scores.std()*2:.3f})")
```

## Key Takeaways 🎯

1. **Linear regression finds the best line** through data points
2. **Multiple features** can be combined linearly
3. **Polynomial features** capture non-linear relationships
4. **Regularization** (Ridge/Lasso) prevents overfitting
5. **Feature scaling** important for regularized methods
6. **R² score** measures fraction of variance explained
7. **Residual analysis** helps validate model assumptions
8. **Tree-based methods** handle non-linearity naturally

## Next Steps 🚀

1. **Practice**: Work through `../../notebooks/07_regression_lab.ipynb`
2. **Learn evaluation**: Deep dive into metrics `../03_model_evaluation/`
3. **Try advanced methods**: Explore gradient boosting and neural networks
4. **Real project**: Apply regression to a dataset you care about

## Quick Challenge 💪

Build a regression model that can predict:
- **Student exam scores** based on study hours, previous grades, and attendance
- **Stock prices** based on company fundamentals and market indicators
- **Website traffic** based on content type, posting time, and social media engagement

Which algorithm would you choose for each problem and why?

*Detailed solutions and analysis in the exercises folder!*
