# Feature Engineering: The Art of Creating Better Data for Machine Learning

## What is Feature Engineering?

Imagine you're a chef preparing ingredients for a complex dish. You don't just throw raw vegetables into the pot – you wash, peel, chop, season, and combine them in specific ways to create the perfect flavor. Feature engineering is exactly like this, but for machine learning. It's the art and science of transforming raw data into meaningful representations that help algorithms understand patterns better.

**Simple Definition:** Feature engineering is the process of creating, selecting, and transforming the input variables (features) that your machine learning model will use to make predictions.

## Why Does This Matter?

### The 80/20 Rule of ML
In the real world, data scientists often spend 80% of their time on data preparation and feature engineering, and only 20% on actual modeling. Why? Because:

1. **Garbage In, Garbage Out**: Even the most sophisticated algorithm can't make good predictions from poor features
2. **Domain Knowledge Wins**: A simple model with excellent features often outperforms a complex model with raw features
3. **Real Impact**: Google's machine learning experts claim that feature engineering can improve model performance by 10-50%, sometimes even more!

### Real-World Impact Example
Netflix's recommendation system doesn't just use "movies you watched." They engineer features like:
- Time of day you watch certain genres
- How long you hover over a title before clicking
- Whether you finish episodes in one sitting
- Similarity scores between shows based on multiple dimensions

This feature engineering is what makes their recommendations feel almost psychic!

## Core Concepts in Feature Engineering

### 1. Feature Creation (Feature Generation)

**What It Is:** Creating new features by combining, transforming, or extracting information from existing features.

**Why It's Powerful:** Raw data rarely tells the complete story. By creating new features, we help the model see patterns it couldn't detect before.

#### Types of Feature Creation:

**A. Domain-Based Features**
Using business or domain knowledge to create meaningful features.

*Example:* E-commerce dataset
- Raw features: `purchase_date`, `customer_birthdate`
- Created features: 
    - `customer_age_at_purchase`
    - `days_since_last_purchase`
    - `is_weekend_purchase`
    - `season_of_purchase`

**B. Mathematical Transformations**
- **Ratios**: `debt_to_income_ratio = total_debt / annual_income`
- **Differences**: `price_change = current_price - previous_price`
- **Aggregations**: `average_purchase_amount`, `total_transactions`
- **Polynomial Features**: For features x and y, create x², y², xy

**C. Temporal Features**
Extracting time-based patterns:
```python
# Example: Extracting temporal features from datetime
import pandas as pd

def extract_temporal_features(df, date_column):
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        return df
```

**D. Text-Based Features**
From text data, create:
- Word count
- Character count
- Presence of specific keywords
- Sentiment scores
- TF-IDF values

#### Pros and Cons of Feature Creation

**Pros:**
- ✅ Can dramatically improve model performance
- ✅ Incorporates domain expertise into the model
- ✅ Makes patterns more explicit and learnable
- ✅ Can reduce model complexity needs

**Cons:**
- ❌ Risk of creating too many features (curse of dimensionality)
- ❌ Can introduce noise if not done carefully
- ❌ Time-consuming and requires expertise
- ❌ Risk of data leakage if using future information

### 2. Feature Selection

**What It Is:** The process of choosing the most relevant features from your dataset while discarding irrelevant or redundant ones.

**Analogy:** It's like packing for a trip – you want to bring everything useful but avoid overpacking with items you won't need.

#### Methods of Feature Selection:

**A. Filter Methods**
Statistical tests to rank features independently of the model.

- **Correlation Analysis**: Remove features highly correlated with each other
- **Chi-Square Test**: For categorical features vs categorical target
- **ANOVA F-test**: For numerical features vs categorical target
- **Mutual Information**: Measures dependency between features and target

```python
# Example: Correlation-based feature selection
import pandas as pd
import numpy as np

def remove_correlated_features(df, threshold=0.95):
        """Remove features with correlation higher than threshold"""
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_tri.columns 
                             if any(upper_tri[column] > threshold)]
        return df.drop(columns=to_drop)
```

**B. Wrapper Methods**
Use the actual model to evaluate feature subsets.

- **Forward Selection**: Start with no features, add one at a time
- **Backward Elimination**: Start with all features, remove one at a time
- **Recursive Feature Elimination (RFE)**: Recursively remove least important features

**C. Embedded Methods**
Feature selection happens during model training.

- **LASSO Regression**: L1 regularization naturally zeros out unimportant features
- **Tree-based Feature Importance**: Random Forests, XGBoost provide feature importance scores
- **Regularization techniques**: Penalize model complexity

#### Important Considerations:

1. **Feature Redundancy**: Two features providing the same information
2. **Feature Relevance**: How much a feature contributes to prediction
3. **Feature Interaction**: Sometimes individually weak features are strong together

#### Pros and Cons of Feature Selection

**Pros:**
- ✅ Reduces overfitting risk
- ✅ Improves model interpretability
- ✅ Decreases training time
- ✅ Reduces storage requirements

**Cons:**
- ❌ May lose some information
- ❌ Selection process can be computationally expensive
- ❌ Different methods may give different results
- ❌ Optimal subset varies by algorithm

### 3. Feature Transformation

**What It Is:** Modifying features to make them more suitable for machine learning algorithms.

**Why Transform?** Many algorithms make assumptions about data (like normal distribution or similar scales). Transformations help meet these assumptions.

#### Common Transformation Techniques:

**A. Scaling and Normalization**

1. **Min-Max Scaling (Normalization)**
     - Scales features to a fixed range, typically [0, 1]
     - Formula: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
     - Use when: You know the approximate bounds of your data

2. **Z-Score Normalization (Standardization)**
     - Centers data around mean 0 with standard deviation 1
     - Formula: $x_{normalized} = \frac{x - \mu}{\sigma}$
     - Use when: Features follow normal distribution

3. **Robust Scaling**
     - Uses median and IQR, resistant to outliers
     - Formula: $x_{robust} = \frac{x - median}{IQR}$
     - Use when: Data contains outliers

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Example usage
def scale_features(X, method='standard'):
        """Scale features using specified method"""
        if method == 'minmax':
                scaler = MinMaxScaler()
        elif method == 'standard':
                scaler = StandardScaler()
        elif method == 'robust':
                scaler = RobustScaler()
        else:
                raise ValueError(f"Unknown scaling method: {method}")
        
        return scaler.fit_transform(X)
```

**B. Mathematical Transformations**

1. **Log Transformation**
     - Reduces skewness in right-skewed data
     - Makes multiplicative relationships additive
     - Example: Income, prices, population data

2. **Square Root Transformation**
     - Milder than log transformation
     - Good for count data

3. **Box-Cox Transformation**
     - Finds optimal power transformation
     - Makes data more normal-like

**C. Encoding Categorical Variables**

1. **One-Hot Encoding**
     - Creates binary columns for each category
     - Good for: Nominal categories (no order)
     - Watch out for: High cardinality creates many columns

2. **Label Encoding**
     - Assigns integer to each category
     - Good for: Ordinal categories (with order)
     - Caution: Can introduce false ordering

3. **Target Encoding**
     - Replaces category with target mean
     - Good for: High cardinality features
     - Risk: Data leakage if not done carefully

4. **Binary Encoding**
     - Converts to binary representation
     - Good for: High cardinality with less columns than one-hot

#### Pros and Cons of Feature Transformation

**Pros:**
- ✅ Makes algorithms converge faster
- ✅ Prevents features with large ranges from dominating
- ✅ Can reveal hidden patterns
- ✅ Meets algorithm assumptions

**Cons:**
- ❌ Can lose interpretability
- ❌ Wrong transformation can hurt performance
- ❌ Requires careful validation
- ❌ May need different transformations for different features

### 4. Handling Missing Values

**What It Is:** Strategies for dealing with incomplete data in your dataset.

**The Reality:** Real-world data is messy. Studies show 60-80% of data science time goes to cleaning, with missing values being a major challenge.

#### Types of Missing Data:

1. **Missing Completely at Random (MCAR)**
     - No pattern to missingness
     - Example: Sensor randomly fails

2. **Missing at Random (MAR)**
     - Missingness depends on other observed variables
     - Example: Income missing more often for younger people

3. **Missing Not at Random (MNAR)**
     - Missingness depends on the value itself
     - Example: High earners not disclosing income

#### Strategies for Handling Missing Values:

**A. Deletion Methods**

1. **Complete Case Analysis (Listwise Deletion)**
     - Remove entire rows with any missing values
     - Pros: Simple, maintains data relationships
     - Cons: Loses information, can introduce bias

2. **Pairwise Deletion**
     - Use available data for each analysis
     - Pros: Uses more data
     - Cons: Different analyses use different subsets

**B. Imputation Methods**

1. **Simple Imputation**
     ```python
     # Mean/Median/Mode imputation
     df['age'].fillna(df['age'].mean(), inplace=True)  # Mean
     df['income'].fillna(df['income'].median(), inplace=True)  # Median
     df['category'].fillna(df['category'].mode()[0], inplace=True)  # Mode
     ```

2. **Forward/Backward Fill** (Time Series)
     ```python
     df['temperature'].fillna(method='ffill')  # Forward fill
     df['stock_price'].fillna(method='bfill')  # Backward fill
     ```

3. **Advanced Imputation**
     - K-Nearest Neighbors (KNN) Imputation
     - Multivariate Imputation (MICE)
     - Deep Learning Imputation

**C. Indicator Method**
Create a binary feature indicating missingness:
```python
df['age_was_missing'] = df['age'].isna().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

#### Pros and Cons of Missing Value Handling

**Pros:**
- ✅ Prevents algorithm errors
- ✅ Retains more data for training
- ✅ Can capture missingness patterns
- ✅ Improves model robustness

**Cons:**
- ❌ Imputation can introduce bias
- ❌ May create false patterns
- ❌ Complex methods are computationally expensive
- ❌ No perfect solution exists

### 5. Feature Interaction and Polynomial Features

**What It Is:** Creating new features that capture relationships between existing features.

**Intuition:** Sometimes the magic isn't in individual ingredients but in how they combine. Salt alone and caramel alone are good, but salted caramel is extraordinary!

#### Types of Interactions:

**A. Multiplicative Interactions**
```python
# Example: Area calculation
df['room_area'] = df['room_length'] * df['room_width']

# Example: Economic indicator
df['purchasing_power'] = df['income'] / df['cost_of_living_index']
```

**B. Polynomial Features**
For features x₁ and x₂, polynomial degree 2 creates:
- Original: {x₁, x₂}
- Degree 2: {x₁, x₂, x₁², x₂², x₁x₂}
- Degree 3: {x₁, x₂, x₁², x₂², x₁x₂, x₁³, x₂³, x₁²x₂, x₁x₂²}

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

**C. Domain-Specific Interactions**
- **Finance**: Debt-to-Income Ratio
- **Healthcare**: BMI (weight/height²)
- **Marketing**: Customer Lifetime Value (avg_purchase × purchase_frequency × customer_lifespan)

#### Important Considerations:

1. **Combinatorial Explosion**: With n features and degree d, polynomial features grow as O(n^d)
2. **Interpretability**: Interactions can be hard to explain
3. **Overfitting Risk**: More features increase model complexity

## Advanced Feature Engineering Techniques

### 1. Binning (Discretization)
Converting continuous features into discrete bins:
- **Equal-width binning**: Same range for each bin
- **Equal-frequency binning**: Same number of samples per bin
- **Custom binning**: Based on domain knowledge

### 2. Feature Crossing
Combining categorical features:
```python
# Example: Location × Time crossing
df['location_time'] = df['city'] + '_' + df['time_of_day']
# Creates: "NYC_morning", "NYC_evening", etc.
```

### 3. Embedding-Based Features
- Word embeddings for text (Word2Vec, GloVe)
- Entity embeddings for categorical variables
- Image embeddings from pre-trained CNNs

### 4. Frequency Encoding
Replace categories with their frequency:
```python
frequency_map = df['category'].value_counts().to_dict()
df['category_frequency'] = df['category'].map(frequency_map)
```

## Best Practices and Guidelines

### The Feature Engineering Workflow

1. **Understand Your Data**
     - Explore distributions, correlations, patterns
     - Identify data types and quality issues
     - Talk to domain experts

2. **Start Simple**
     - Begin with basic transformations
     - Add complexity gradually
     - Measure impact at each step

3. **Validate Rigorously**
     - Use cross-validation to avoid overfitting
     - Test on holdout data
     - Monitor for data leakage

4. **Document Everything**
     - Keep track of transformations
     - Note assumptions and decisions
     - Create reproducible pipelines

### Common Pitfalls to Avoid

1. **Data Leakage**
     - Never use future information to predict past
     - Be careful with target encoding
     - Split data before feature engineering

2. **Over-Engineering**
     - More features isn't always better
     - Consider computational cost
     - Maintain interpretability when needed

3. **Ignoring Business Context**
     - Features should make business sense
     - Consider deployment constraints
     - Think about feature availability in production

## Real-World Case Studies

### Case Study 1: Kaggle Home Prices Competition
Winners often create 100+ features including:
- Polynomial combinations of area features
- Ratios of different room types
- Neighborhood-based aggregations
- Age and quality interactions

### Case Study 2: Credit Risk Modeling
Key engineered features:
- Payment-to-income ratios
- Credit utilization trends
- Time since last default
- Seasonal spending patterns

### Case Study 3: Customer Churn Prediction
Effective features:
- Engagement trend (increasing/decreasing)
- Days since last interaction
- Usage compared to similar customers
- Support ticket sentiment

## Practical Exercises

### Exercise 1: Time-Based Features
**Dataset**: Daily sales data
**Task**: Create features for:
- Day of week patterns
- Monthly seasonality
- Holiday indicators
- Rolling averages (7-day, 30-day)

### Exercise 2: Text Feature Engineering
**Dataset**: Customer reviews
**Task**: Extract:
- Review length
- Sentiment scores
- Keyword presence
- Readability metrics

### Exercise 3: Interaction Discovery
**Dataset**: House prices
**Task**: Find meaningful interactions:
- Total area (all room areas combined)
- Price per square foot by neighborhood
- Age × condition score
- Bathroom-to-bedroom ratio

### Exercise 4: Missing Value Strategy
**Dataset**: Medical records with 30% missing values
**Task**: Compare approaches:
- Simple imputation vs. KNN imputation
- With and without missingness indicators
- Impact on prediction accuracy

## Tools and Libraries

### Python Libraries for Feature Engineering

1. **Feature-engine**: Comprehensive feature engineering
2. **Featuretools**: Automated feature engineering
3. **Category Encoders**: Various encoding methods
4. **scikit-learn**: Basic preprocessing and transformation

### Example Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define preprocessing for different column types
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['education', 'occupation', 'city']

numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
        transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
        ])
```

## Conclusion

Feature engineering is where the real magic happens in machine learning. It's the bridge between raw data and accurate predictions. While algorithms get a lot of attention, experienced practitioners know that better features often trump better algorithms.

### Key Takeaways:

1. **It's an Art and Science**: Combines creativity, domain knowledge, and technical skills
2. **Iterative Process**: Continuously refine and improve features
3. **Impact is Huge**: Can improve model performance more than algorithm selection
4. **No One-Size-Fits-All**: Different problems require different approaches
5. **Validation is Critical**: Always measure impact and watch for leakage

### Your Learning Path:

1. **Start**: Practice with simple transformations (scaling, encoding)
2. **Build**: Learn to create domain-specific features
3. **Refine**: Master selection and validation techniques
4. **Advance**: Explore automated feature engineering
5. **Master**: Develop intuition for what works when

Remember: Great feature engineering is what separates good data scientists from great ones. It's worth investing time to master these techniques!

## Mathematical Foundation

### Core Formulas and Concepts

**Min-Max Scaling:**
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
- Maps values to [0, 1] range
- Preserves relationships
- Sensitive to outliers

**Z-Score Normalization:**
$$x_{normalized} = \frac{x - \mu}{\sigma}$$
- Centers around 0 with σ = 1
- Assumes normal distribution
- Less sensitive to outliers than min-max

**Robust Scaling:**
$$x_{robust} = \frac{x - Q_{50}}{Q_{75} - Q_{25}}$$
- Uses median (Q₅₀) and IQR
- Robust to outliers
- Good for skewed distributions

**Log Transformation:**
$$x_{transformed} = \log(x + c)$$
- Reduces right skewness
- Makes multiplicative relationships additive
- Constant c prevents log(0) errors

**Box-Cox Transformation:**
$$x_{transformed} = \begin{cases}
\frac{x^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$
- Finds optimal λ to normalize data
- More flexible than log transform

**Polynomial Features:**
For features $x_1, x_2$, degree-2 polynomial features:
$$\{1, x_1, x_2, x_1^2, x_2^2, x_1x_2\}$$
- Total features: ${n+d \choose d}$ where n=original features, d=degree

**Information Gain:**
$$IG(T, A) = H(T) - H(T|A)$$
Where entropy: $H(T) = -\sum_{i} p_i \log_2(p_i)$
- Measures feature importance
- Used in decision trees
- Higher IG = more informative feature

**Pearson Correlation:**
$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$
- Measures linear relationship
- Range: [-1, 1]
- 0 = no linear correlation

**Mutual Information:**
$$MI(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$
- Captures non-linear relationships
- Always non-negative
- 0 = independent variables

**Principal Component Analysis:**
$$PC_j = \sum_{i=1}^{p} a_{ji}x_i$$
Where $a_{ji}$ are eigenvectors of covariance matrix
- Reduces dimensionality
- Preserves maximum variance
- Components are orthogonal

### Comprehensive Solved Examples

#### Example 1: Complete Feature Engineering Pipeline

**Problem**: Predict house prices with raw features:
- Size: [800, 1200, 2000, 1500, 900] sqft
- Bedrooms: [2, 3, 4, 3, 2]
- Age: [5, 10, 2, 7, 15] years
- Neighborhood: ['A', 'B', 'A', 'C', 'B']
- Price: [150, 280, 450, 320, 180] (in $1000s)

**Solution**:

Step 1: Create interaction features
```
price_per_sqft = Price / Size
= [0.1875, 0.2333, 0.2250, 0.2133, 0.2000]

rooms_per_sqft = Bedrooms / Size × 1000
= [2.50, 2.50, 2.00, 2.00, 2.22]

age_factor = Age / 50 (assuming 50-year lifespan)
= [0.10, 0.20, 0.04, 0.14, 0.30]
```

Step 2: Scale numerical features (Min-Max)
```
Size_scaled = (Size - 800) / (2000 - 800)
= [0.00, 0.33, 1.00, 0.58, 0.08]

Age_scaled = (Age - 2) / (15 - 2)
= [0.23, 0.62, 0.00, 0.38, 1.00]
```

Step 3: Encode categorical features (One-Hot)
```
Neighborhood_A = [1, 0, 1, 0, 0]
Neighborhood_B = [0, 1, 0, 0, 1]
Neighborhood_C = [0, 0, 0, 1, 0]
```

Step 4: Create polynomial features (degree 2, selected)
```
Size_Age = Size_scaled × Age_scaled
= [0.00, 0.20, 0.00, 0.22, 0.08]

Size_squared = Size_scaled²
= [0.00, 0.11, 1.00, 0.34, 0.01]
```

Final feature matrix (10 features from original 4):
- Original scaled: Size_scaled, Bedrooms, Age_scaled
- Encoded: Neighborhood_A, Neighborhood_B, Neighborhood_C
- Engineered: price_per_sqft, rooms_per_sqft, Size_Age, Size_squared

#### Example 2: Information Gain for Feature Selection

**Problem**: Select best feature for predicting customer churn
- Feature A (Usage_hours): [10, 5, 8, 2, 12, 3, 9, 1]
- Feature B (Support_calls): [0, 2, 1, 3, 0, 2, 1, 4]
- Target (Churned): [0, 1, 0, 1, 0, 1, 0, 1]

**Solution**:

Step 1: Discretize Feature A (Usage_hours)
- Low (≤5): [5, 2, 3, 1] → Churned: [1, 1, 1, 1]
- High (>5): [10, 8, 12, 9] → Churned: [0, 0, 0, 0]

Step 2: Calculate entropy for target
$$H(Churned) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1.0$$

Step 3: Calculate conditional entropy for Feature A
$$H(Churned|Usage) = P(Low) \times H(Churned|Low) + P(High) \times H(Churned|High)$$
$$= 0.5 \times 0 + 0.5 \times 0 = 0$$

Step 4: Calculate Information Gain
$$IG(Churned, Usage) = 1.0 - 0 = 1.0$$ (Perfect split!)

Step 5: Repeat for Feature B (Support_calls)
- 0 calls: [10, 12] → Churned: [0, 0]
- 1 call: [8, 9] → Churned: [0, 0]
- 2 calls: [5, 3] → Churned: [1, 1]
- 3 calls: [2] → Churned: [1]
- 4 calls: [1] → Churned: [1]

$$H(Churned|Support) = 0.25 \times 0 + 0.25 \times 0 + 0.25 \times 0 + 0.125 \times 0 + 0.125 \times 0 = 0$$
$$IG(Churned, Support) = 1.0 - 0 = 1.0$$

Both features perfectly separate the classes after discretization!

#### Example 3: Handling Missing Values with Multiple Strategies

**Problem**: Dataset with missing values
```
Age: [25, NaN, 35, 28, NaN, 42, 31, 38]
Income: [50k, 60k, NaN, 55k, 45k, 80k, NaN, 70k]
Purchased: [1, 1, 0, 1, 0, 1, 0, 1]
```

**Solution**:

Strategy 1: Mean Imputation
```
Age_mean = (25+35+28+42+31+38)/6 = 33.17
Age_imputed = [25, 33.17, 35, 28, 33.17, 42, 31, 38]

Income_mean = (50+60+55+45+80+70)/6 = 60k
Income_imputed = [50k, 60k, 60k, 55k, 45k, 80k, 60k, 70k]
```

Strategy 2: KNN Imputation (k=2)
For Age[1] (index 1):
- Find 2 nearest complete samples based on Income(60k)
- Nearest: Income=55k(Age=28), Income=50k(Age=25)
- Imputed Age = (28+25)/2 = 26.5

Strategy 3: Predictive Imputation
Use correlation with Purchase behavior:
- Purchased=1: Mean Age = 33, Mean Income = 63.75k
- Purchased=0: Mean Age = 34.67, Mean Income = 50k

Strategy 4: Missingness Indicators
```
Age_missing = [0, 1, 0, 0, 1, 0, 0, 0]
Income_missing = [0, 0, 1, 0, 0, 0, 1, 0]
```

#### Example 4: Complex Feature Interaction

**Problem**: Create polynomial and interaction features for customer behavior prediction

Original features:
- Recency (days since last purchase): [7, 30, 3, 15]
- Frequency (purchases/month): [4, 1, 8, 2]
- Monetary (avg purchase $): [100, 50, 200, 75]

**Solution**:

Step 1: Normalize features (Z-score)
```
Recency: μ=13.75, σ=11.32
R_norm = [-0.60, 1.44, -0.95, 0.11]

Frequency: μ=3.75, σ=2.87
F_norm = [0.09, -0.96, 1.48, -0.61]

Monetary: μ=106.25, σ=62.5
M_norm = [-0.10, -0.90, 1.50, -0.50]
```

Step 2: Create RFM Score (domain knowledge)
```
RFM_Score = (5-R_rank) × 0.3 + F_rank × 0.3 + M_rank × 0.4
Where ranks are 1-4 based on value order

RFM_Score = [3.7, 1.3, 4.0, 2.0]
```

Step 3: Polynomial features (degree 2, selected)
```
R² = [0.36, 2.07, 0.90, 0.01]
F² = [0.01, 0.92, 2.19, 0.37]
M² = [0.01, 0.81, 2.25, 0.25]
R×F = [-0.05, -1.38, -1.41, -0.07]
R×M = [0.06, -1.30, -1.43, -0.06]
F×M = [-0.01, 0.86, 2.22, 0.31]
```

Step 4: Business logic features
```
Customer_Value = Frequency × Monetary
= [400, 50, 1600, 150]

Engagement_Score = 1/Recency × Frequency
= [0.57, 0.03, 2.67, 0.13]

Purchase_Velocity = Monetary/Recency
= [14.29, 1.67, 66.67, 5.00]
```

Final feature set expanded from 3 to 15 meaningful features!

### Feature Engineering Decision Matrix

| Scenario | Recommended Technique | Why |
|----------|----------------------|-----|
| Skewed distribution | Log/Box-Cox transform | Normalizes distribution |
| Different scales | Standardization | Equal feature importance |
| Tree-based models | Minimal scaling needed | Trees handle scales |
| High cardinality categorical | Target/Frequency encoding | Reduces dimensionality |
| Missing < 5% | Simple imputation | Minimal bias risk |
| Missing > 30% | Create missing indicator | Missingness is informative |
| Non-linear patterns | Polynomial features | Captures curves |
| Time series | Lag features, rolling stats | Temporal patterns |
| Text data | TF-IDF, embeddings | Numerical representation |
| Sparse data | Keep sparse format | Memory efficiency |

### Performance Impact Examples

Real-world improvements from feature engineering:

1. **Kaggle Competitions**:
     - House Prices: 15% → 3% error (80% improvement)
     - Titanic: 76% → 82% accuracy (25% error reduction)

2. **Industry Applications**:
     - Netflix: 10% improvement in recommendations
     - Amazon: 35% better click-through rate
     - Credit scoring: 20% reduction in default rates

3. **Research Benchmarks**:
     - Image classification: 5-10% accuracy gain with augmentation
     - NLP: 15-20% improvement with proper preprocessing
     - Time series: 30% better forecasts with seasonal features

Remember: "Applied machine learning is basically feature engineering" - Andrew Ng