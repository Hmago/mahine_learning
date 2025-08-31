# Preprocessing Pipeline: Preparing Data for Machine Learning

## 🤔 What is Data Preprocessing?

Imagine you're a chef preparing ingredients for a gourmet meal. You need to:
- **Wash** the vegetables (clean the data)
- **Chop** them uniformly (standardize features)
- **Season** appropriately (transform features)
- **Arrange** them properly (structure the data)

**Data preprocessing is exactly the same** - it's preparing your raw data so machine learning algorithms can create something amazing!

## 🎯 Why Preprocessing is Critical

**Raw data is like uncut diamonds** - valuable but needs processing to shine. Machine learning algorithms are picky eaters:

- **Algorithms expect numbers**: Can't handle "Red", "Blue", "Green" 
- **Algorithms expect similar scales**: Can't compare age (20-80) with income (20,000-200,000)
- **Algorithms expect complete data**: Can't handle missing values
- **Algorithms expect clean data**: Can't handle outliers and noise

**Skip preprocessing = Poor model performance!**

## 🧠 Core Preprocessing Concepts

### 1. **Feature Scaling: Making Data Comparable**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Create sample data with different scales
np.random.seed(42)
employee_data = pd.DataFrame({
    'age': np.random.randint(22, 65, 1000),                    # Scale: 22-65
    'salary': np.random.normal(75000, 25000, 1000),           # Scale: ~25k-125k  
    'years_experience': np.random.randint(0, 40, 1000),       # Scale: 0-40
    'performance_score': np.random.uniform(1, 10, 1000)       # Scale: 1-10
})

print("🎯 FEATURE SCALING DEMONSTRATION")
print("=" * 40)

print("Original data statistics:")
print(employee_data.describe().round(2))

print(f"\nProblem: Features have very different scales!")
print(f"Age: {employee_data['age'].min()}-{employee_data['age'].max()}")
print(f"Salary: ${employee_data['salary'].min():,.0f}-${employee_data['salary'].max():,.0f}")
print(f"Experience: {employee_data['years_experience'].min()}-{employee_data['years_experience'].max()}")

# StandardScaler: Mean=0, Std=1
scaler_standard = StandardScaler()
data_standard = pd.DataFrame(
    scaler_standard.fit_transform(employee_data),
    columns=employee_data.columns
)

print(f"\n✅ After StandardScaler (mean≈0, std≈1):")
print(data_standard.describe().round(2))

# MinMaxScaler: Scale to 0-1 range
scaler_minmax = MinMaxScaler()
data_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(employee_data),
    columns=employee_data.columns
)

print(f"\n✅ After MinMaxScaler (range: 0-1):")
print(data_minmax.describe().round(2))

# Business example: Why this matters
print(f"\n💼 Business Impact:")
print(f"Without scaling: Salary dominates (75,000 vs age 45)")
print(f"With scaling: All features contribute equally to model")
```

### 2. **Handling Categorical Variables: From Text to Numbers**

```python
print("\n🏷️ CATEGORICAL VARIABLE ENCODING")
print("=" * 40)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create categorical data
customer_data = pd.DataFrame({
    'customer_id': range(1, 1001),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'city_type': np.random.choice(['Rural', 'Suburban', 'Urban'], 1000),
    'preferred_contact': np.random.choice(['Email', 'Phone', 'SMS', 'Mail'], 1000),
    'income': np.random.normal(60000, 20000, 1000)
})

print("Original categorical data:")
print(customer_data['education'].value_counts())

# Method 1: Label Encoding (for ordinal data)
print(f"\n1. Label Encoding (Good for ordinal data):")
education_order = ['High School', 'Bachelor', 'Master', 'PhD']  # Natural order

label_encoder = LabelEncoder()
customer_data['education_label'] = label_encoder.fit_transform(customer_data['education'])

print("Education Label Mapping:")
for original, encoded in zip(education_order, label_encoder.transform(education_order)):
    print(f"{original} → {encoded}")

# Method 2: One-Hot Encoding (for nominal data)
print(f"\n2. One-Hot Encoding (Good for nominal data):")
# One-hot encoding for city_type (no natural order)
city_encoded = pd.get_dummies(customer_data['city_type'], prefix='city')
print("City type encoding:")
print(city_encoded.head())

# Method 3: Using sklearn's ColumnTransformer (Professional approach)
print(f"\n3. Professional Approach with ColumnTransformer:")

# Define which columns need which encoding
categorical_features = ['education', 'city_type', 'preferred_contact'] 
numerical_features = ['income']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # drop='first' prevents multicollinearity
    ])

# Fit and transform
X_preprocessed = preprocessor.fit_transform(customer_data[numerical_features + categorical_features])

print(f"Original features: {len(numerical_features + categorical_features)}")
print(f"After one-hot encoding: {X_preprocessed.shape[1]} features")
print("✅ Ready for machine learning!")
```

### 3. **Handling Missing Data: Dealing with Reality**

```python
print("\n🕳️ MISSING DATA STRATEGIES")
print("=" * 32)

from sklearn.impute import SimpleImputer, KNNImputer

# Create data with missing values (realistic scenario)
survey_data = pd.DataFrame({
    'age': np.random.randint(18, 70, 500),
    'income': np.random.normal(60000, 20000, 500),
    'satisfaction': np.random.uniform(1, 10, 500),
    'years_employed': np.random.randint(0, 30, 500)
})

# Introduce missing values realistically
# Older people less likely to report income
missing_income_mask = (survey_data['age'] > 50) & (np.random.random(500) < 0.3)
survey_data.loc[missing_income_mask, 'income'] = np.nan

# Random missing satisfaction scores
missing_satisfaction_mask = np.random.random(500) < 0.15
survey_data.loc[missing_satisfaction_mask, 'satisfaction'] = np.nan

print(f"Missing data created:")
print(survey_data.isnull().sum())
print(f"Missing percentages:")
print((survey_data.isnull().sum() / len(survey_data) * 100).round(1))

# Strategy 1: Simple Imputation
print(f"\n1. Simple Imputation Strategies:")

# Mean imputation for numerical data
mean_imputer = SimpleImputer(strategy='mean')
income_mean_filled = mean_imputer.fit_transform(survey_data[['income']])

# Median imputation (robust to outliers)
median_imputer = SimpleImputer(strategy='median')  
income_median_filled = median_imputer.fit_transform(survey_data[['income']])

# Mode imputation for categorical data
mode_imputer = SimpleImputer(strategy='most_frequent')

print(f"Original income mean: ${survey_data['income'].mean():,.2f}")
print(f"After mean imputation: ${income_mean_filled.mean():,.2f}")
print(f"After median imputation: ${income_median_filled.mean():,.2f}")

# Strategy 2: Advanced KNN Imputation
print(f"\n2. KNN Imputation (Smart approach):")
knn_imputer = KNNImputer(n_neighbors=5)
data_knn_filled = knn_imputer.fit_transform(survey_data)

knn_filled_df = pd.DataFrame(data_knn_filled, columns=survey_data.columns)
print(f"KNN imputed income mean: ${knn_filled_df['income'].mean():,.2f}")
print("✅ KNN uses similar records to estimate missing values")

# Strategy 3: Business Logic Imputation
print(f"\n3. Business Logic Imputation:")
business_filled = survey_data.copy()

# Income imputation based on age and experience
def estimate_income(row):
    if pd.isna(row['income']):
        # Estimate based on age and experience
        base_income = 30000 + row['age'] * 800 + row['years_employed'] * 1200
        return base_income + np.random.normal(0, 5000)  # Add some variation
    return row['income']

business_filled['income'] = business_filled.apply(estimate_income, axis=1)
print(f"Business logic income mean: ${business_filled['income'].mean():,.2f}")
print("✅ Used domain knowledge to create realistic estimates")

# Compare imputation strategies
print(f"\n📊 Imputation Strategy Comparison:")
strategies = {
    'Original (with missing)': survey_data['income'].mean(),
    'Mean imputation': income_mean_filled.mean(),
    'Median imputation': income_median_filled.mean(), 
    'KNN imputation': knn_filled_df['income'].mean(),
    'Business logic': business_filled['income'].mean()
}

for strategy, mean_value in strategies.items():
    if not pd.isna(mean_value):
        print(f"{strategy}: ${mean_value:,.2f}")
```

### 4. **Feature Engineering: Creating Powerful Features**

```python
print("\n⚙️ FEATURE ENGINEERING MASTERY")
print("=" * 38)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

# Start with customer data
customer_features = pd.DataFrame({
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.normal(60000, 20000, 1000),
    'years_customer': np.random.randint(0, 15, 1000),
    'monthly_purchases': np.random.poisson(3, 1000),
    'satisfaction': np.random.uniform(1, 10, 1000)
})

# Create target: customer lifetime value
customer_features['clv'] = (
    customer_features['income'] * 0.01 +
    customer_features['years_customer'] * 500 +
    customer_features['monthly_purchases'] * 200 +
    customer_features['satisfaction'] * 300 +
    np.random.normal(0, 500, 1000)
)

print("1. Original Features:")
print(customer_features.columns.tolist())

print(f"\n2. Derived Features (Feature Engineering):")
# Create meaningful business features
customer_features['income_per_age'] = customer_features['income'] / customer_features['age']
customer_features['purchases_per_year'] = customer_features['monthly_purchases'] * 12 / customer_features['years_customer'].clip(lower=1)
customer_features['satisfaction_income_ratio'] = customer_features['satisfaction'] / (customer_features['income'] / 10000)
customer_features['customer_maturity'] = customer_features['years_customer'] * customer_features['satisfaction']
customer_features['spending_potential'] = customer_features['income'] * customer_features['satisfaction'] / 100

print("Engineered features created:")
new_features = ['income_per_age', 'purchases_per_year', 'satisfaction_income_ratio', 
                'customer_maturity', 'spending_potential']
for feature in new_features:
    print(f"✅ {feature}")

print(f"\n3. Polynomial Features (Interaction Terms):")
# Create interaction features automatically
original_features = ['age', 'income', 'years_customer', 'monthly_purchases', 'satisfaction']
poly_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

X_original = customer_features[original_features]
X_poly = poly_transformer.fit_transform(X_original)

print(f"Original features: {X_original.shape[1]}")
print(f"With polynomial features: {X_poly.shape[1]}")
print("Polynomial features include:")
feature_names = poly_transformer.get_feature_names_out(original_features)
print(f"✅ Original features: {original_features}")
print(f"✅ Interaction examples: {feature_names[5:10].tolist()}")

print(f"\n4. Feature Selection (Keeping the Best):")
# Select top features based on statistical significance
y = customer_features['clv']
selector = SelectKBest(score_func=f_regression, k=8)  # Keep top 8 features

X_all_features = customer_features.drop('clv', axis=1)
X_selected = selector.fit_transform(X_all_features, y)

# Get selected feature names
selected_features = X_all_features.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]

print(f"Selected {len(selected_features)} best features:")
for feature, score in zip(selected_features, feature_scores):
    print(f"✅ {feature}: {score:.2f}")
```

### 5. **Advanced Preprocessing Techniques**

```python
print("\n🎨 ADVANCED PREPROCESSING TECHNIQUES")
print("=" * 45)

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA

# Handle skewed data (common in business)
print("1. Handling Skewed Data:")

# Create skewed income data (realistic - few high earners, many average earners)
skewed_income = np.random.lognormal(11, 0.5, 1000)  # Log-normal distribution
original_skewness = pd.Series(skewed_income).skew()

print(f"Original income skewness: {original_skewness:.2f}")
print(f"Original income range: ${skewed_income.min():,.0f} - ${skewed_income.max():,.0f}")

# Power transformer (makes data more normal)
power_transformer = PowerTransformer(method='yeo-johnson')
income_transformed = power_transformer.fit_transform(skewed_income.reshape(-1, 1))
transformed_skewness = pd.Series(income_transformed.flatten()).skew()

print(f"After PowerTransformer skewness: {transformed_skewness:.2f}")
print("✅ Data is now more normally distributed")

print(f"\n2. Robust Scaling (Handles Outliers):")
# Create data with outliers
data_with_outliers = customer_features['income'].copy()
# Add some extreme outliers (CEOs, lottery winners)
outlier_indices = np.random.choice(len(data_with_outliers), 20, replace=False)
data_with_outliers.iloc[outlier_indices] *= 10  # Make them 10x higher

print(f"Data with outliers:")
print(f"Mean: ${data_with_outliers.mean():,.0f}")
print(f"Median: ${data_with_outliers.median():,.0f}")
print(f"Max: ${data_with_outliers.max():,.0f}")

# Compare scaling methods
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()  # Uses median and IQR instead of mean and std

standard_scaled = standard_scaler.fit_transform(data_with_outliers.values.reshape(-1, 1))
robust_scaled = robust_scaler.fit_transform(data_with_outliers.values.reshape(-1, 1))

print(f"\nStandard scaling range: {standard_scaled.min():.2f} to {standard_scaled.max():.2f}")
print(f"Robust scaling range: {robust_scaled.min():.2f} to {robust_scaled.max():.2f}")
print("✅ Robust scaling is less affected by outliers")

print(f"\n3. Dimensionality Reduction (Simplifying Data):")
# When you have too many features, PCA can help
high_dim_data = np.random.randn(1000, 50)  # 50 features
pca = PCA(n_components=10)  # Reduce to 10 components

reduced_data = pca.fit_transform(high_dim_data)
explained_variance_ratio = pca.explained_variance_ratio_.sum()

print(f"Original dimensions: {high_dim_data.shape[1]}")
print(f"Reduced dimensions: {reduced_data.shape[1]}")
print(f"Variance preserved: {explained_variance_ratio:.1%}")
print("✅ Kept most information with fewer features")
```

### 6. **Complete Preprocessing Pipeline Example**

```python
print("\n🏭 COMPLETE PREPROCESSING PIPELINE")
print("=" * 45)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Create realistic messy business data
np.random.seed(42)
messy_business_data = pd.DataFrame({
    'customer_age': np.random.randint(18, 80, 2000),
    'annual_income': np.random.lognormal(11, 0.5, 2000),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 2000),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract', 'Unemployed'], 2000),
    'city_size': np.random.choice(['Small', 'Medium', 'Large'], 2000),
    'years_at_company': np.random.exponential(5, 2000),
    'job_satisfaction': np.random.beta(3, 2, 2000) * 10,  # Skewed toward higher satisfaction
    'commute_distance': np.random.exponential(15, 2000)
})

# Introduce realistic missing data patterns
# Higher income people less likely to share salary info
high_income_mask = messy_business_data['annual_income'] > messy_business_data['annual_income'].quantile(0.8)
missing_income_mask = high_income_mask & (np.random.random(2000) < 0.4)
messy_business_data.loc[missing_income_mask, 'annual_income'] = np.nan

# Some people skip job satisfaction question
messy_business_data.loc[np.random.random(2000) < 0.1, 'job_satisfaction'] = np.nan

print("Messy data overview:")
print(f"Shape: {messy_business_data.shape}")
print(f"Missing values:")
print(messy_business_data.isnull().sum())

# Create target variable: employee retention (will they stay?)
retention_score = (
    0.1 * messy_business_data['job_satisfaction'].fillna(messy_business_data['job_satisfaction'].mean()) +
    -0.02 * messy_business_data['commute_distance'] +
    0.05 * messy_business_data['years_at_company'] +
    np.random.normal(0, 0.5, 2000)
)
messy_business_data['will_stay'] = (retention_score > retention_score.median()).astype(int)

# Define preprocessing steps
numeric_features = ['customer_age', 'annual_income', 'years_at_company', 'job_satisfaction', 'commute_distance']
categorical_features = ['education_level', 'employment_type', 'city_size']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  # Handle missing categories
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encode
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare data
X = messy_business_data[numeric_features + categorical_features]
y = messy_business_data['will_stay']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nPreprocessing Results:")
print(f"Original features: {X.shape[1]}")
print(f"After preprocessing: {X_train_processed.shape[1]}")
print(f"Training samples: {X_train_processed.shape[0]}")
print(f"Test samples: {X_test_processed.shape[0]}")
print("✅ Data is now ML-ready!")

# Quick model to validate preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train a model on preprocessed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Evaluate
predictions = rf_model.predict(X_test_processed)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Model Performance with Preprocessing:")
print(f"Accuracy: {accuracy:.3f}")
print("✅ Preprocessing enabled successful model training!")
```

## 🎯 Business Preprocessing Scenarios

### Scenario 1: Customer Database Integration

```python
def customer_database_preprocessing():
    """Real-world customer data preprocessing scenario"""
    
    print("\n💼 CUSTOMER DATABASE PREPROCESSING")
    print("=" * 45)
    
    # Simulate data from different business systems
    # System 1: CRM data
    crm_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'first_name': [f'Customer_{i}' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
    })
    
    # System 2: Transaction data  
    transaction_data = pd.DataFrame({
        'customer_id': np.random.choice(range(1, 1001), 5000),
        'purchase_amount': np.random.exponential(100, 5000),
        'purchase_date': pd.date_range('2020-01-01', periods=5000, freq='H'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 5000)
    })
    
    # System 3: Customer service data
    service_data = pd.DataFrame({
        'customer_id': np.random.choice(range(1, 1001), 800),
        'satisfaction_rating': np.random.randint(1, 6, 800),
        'support_calls': np.random.poisson(2, 800),
        'issue_type': np.random.choice(['Technical', 'Billing', 'General'], 800)
    })
    
    print("Step 1: Data Integration")
    # Aggregate transaction data per customer
    customer_transactions = transaction_data.groupby('customer_id').agg({
        'purchase_amount': ['sum', 'mean', 'count'],
        'purchase_date': ['min', 'max']
    })
    
    customer_transactions.columns = ['total_spent', 'avg_purchase', 'purchase_count', 'first_purchase', 'last_purchase']
    customer_transactions = customer_transactions.reset_index()
    
    # Aggregate service data per customer
    customer_service = service_data.groupby('customer_id').agg({
        'satisfaction_rating': 'mean',
        'support_calls': 'sum',
        'issue_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'None'
    }).reset_index()
    
    customer_service.columns = ['customer_id', 'avg_satisfaction', 'total_support_calls', 'primary_issue_type']
    
    # Merge all data sources
    integrated_data = (crm_data
                      .merge(customer_transactions, on='customer_id', how='left')
                      .merge(customer_service, on='customer_id', how='left'))
    
    print(f"Integrated dataset shape: {integrated_data.shape}")
    print(f"Features from CRM: {len(crm_data.columns)}")
    print(f"Features from transactions: {len(customer_transactions.columns) - 1}")
    print(f"Features from service: {len(customer_service.columns) - 1}")
    
    print("\nStep 2: Handle Missing Data")
    # Fill missing values with business logic
    integrated_data['total_spent'] = integrated_data['total_spent'].fillna(0)
    integrated_data['purchase_count'] = integrated_data['purchase_count'].fillna(0)
    integrated_data['avg_purchase'] = integrated_data['avg_purchase'].fillna(0)
    integrated_data['total_support_calls'] = integrated_data['total_support_calls'].fillna(0)
    integrated_data['avg_satisfaction'] = integrated_data['avg_satisfaction'].fillna(5)  # Neutral
    integrated_data['primary_issue_type'] = integrated_data['primary_issue_type'].fillna('None')
    
    print("✅ Missing values handled with business logic")
    
    print("\nStep 3: Feature Engineering")
    # Create business-relevant features
    integrated_data['customer_lifetime_days'] = (
        pd.Timestamp.now() - integrated_data['signup_date']
    ).dt.days
    
    integrated_data['purchase_frequency'] = (
        integrated_data['purchase_count'] / 
        np.maximum(integrated_data['customer_lifetime_days'] / 30, 1)  # Purchases per month
    )
    
    integrated_data['support_intensity'] = (
        integrated_data['total_support_calls'] / 
        np.maximum(integrated_data['customer_lifetime_days'] / 30, 1)  # Calls per month
    )
    
    # Customer value score
    integrated_data['customer_value_score'] = (
        integrated_data['total_spent'] * 0.3 +
        integrated_data['purchase_frequency'] * 1000 * 0.3 +
        integrated_data['avg_satisfaction'] * 100 * 0.4
    )
    
    print("✅ Business features engineered")
    print(f"Final dataset shape: {integrated_data.shape}")
    
    return integrated_data

integrated_customer_data = customer_database_preprocessing()
```

### Scenario 2: Text Data Preprocessing

```python
def text_preprocessing_example():
    """Preprocess text data for machine learning"""
    
    print("\n📝 TEXT DATA PREPROCESSING")
    print("=" * 35)
    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    
    # Sample customer feedback data
    feedback_data = pd.DataFrame({
        'customer_feedback': [
            "Great product, excellent service!",
            "Terrible experience, very disappointed.",
            "Good quality but expensive pricing.",
            "Amazing customer support team.",
            "Product broke after one week.",
            "Love the fast shipping and packaging.",
            "Website is confusing and slow.",
            "Perfect product for my needs.",
            "Customer service was unhelpful.",
            "Great value for the money."
        ],
        'rating': [5, 1, 3, 5, 1, 4, 2, 5, 2, 4]
    })
    
    print("Sample customer feedback:")
    print(feedback_data)
    
    # Convert text to numbers using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,  # Keep top 100 words
        stop_words='english',  # Remove common words like 'the', 'and'
        lowercase=True
    )
    
    text_features = tfidf_vectorizer.fit_transform(feedback_data['customer_feedback'])
    
    print(f"\nText preprocessing results:")
    print(f"Original: {len(feedback_data)} text reviews")
    print(f"Converted to: {text_features.shape[1]} numerical features")
    print(f"Feature names (sample): {tfidf_vectorizer.get_feature_names_out()[:10].tolist()}")
    
    # Convert to DataFrame for analysis
    text_df = pd.DataFrame(
        text_features.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )
    
    # Find most important words for positive vs negative reviews
    positive_reviews = text_df[feedback_data['rating'] >= 4]
    negative_reviews = text_df[feedback_data['rating'] <= 2]
    
    positive_words = positive_reviews.mean().sort_values(ascending=False).head()
    negative_words = negative_reviews.mean().sort_values(ascending=False).head()
    
    print(f"\nWords associated with positive reviews:")
    print(positive_words.round(3))
    
    print(f"\nWords associated with negative reviews:")
    print(negative_words.round(3))
    
    return text_df, tfidf_vectorizer

text_features, text_vectorizer = text_preprocessing_example()
```

## 🎮 Preprocessing Practice Challenges

### Challenge 1: Multi-Source Data Integration

```python
def preprocessing_challenge_1():
    """Challenge: Integrate and preprocess multi-source business data"""
    
    print("\n🎯 PREPROCESSING CHALLENGE 1")
    print("=" * 35)
    print("Mission: Create ML-ready dataset from multiple messy sources")
    
    # Data Source 1: HR Database (has missing values)
    hr_data = pd.DataFrame({
        'employee_id': range(1, 501),
        'age': np.random.randint(22, 65, 500),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], 500),
        'salary': np.random.normal(75000, 25000, 500),
        'years_experience': np.random.randint(0, 30, 500)
    })
    
    # Introduce missing salaries (sensitive information)
    hr_data.loc[np.random.random(500) < 0.2, 'salary'] = np.nan
    
    # Data Source 2: Performance Reviews (different frequency)
    performance_data = pd.DataFrame({
        'employee_id': np.random.choice(range(1, 501), 300),  # Not all employees reviewed
        'performance_rating': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 300),
        'goals_met': np.random.uniform(0, 1, 300),
        'review_date': pd.date_range('2023-01-01', periods=300, freq='D')
    })
    
    # Data Source 3: Training Records (sparse data)
    training_data = pd.DataFrame({
        'employee_id': np.random.choice(range(1, 501), 200),
        'training_hours': np.random.exponential(20, 200),
        'certifications': np.random.randint(0, 5, 200)
    })
    
    print("Your Tasks:")
    print("1. Integrate all three data sources")
    print("2. Handle missing values appropriately")
    print("3. Encode categorical variables")
    print("4. Create meaningful derived features")
    print("5. Scale numerical features")
    print("6. Prepare for predicting promotion likelihood")
    
    # TODO: Complete the preprocessing pipeline
    # Hints:
    # - Use merge with how='left' to keep all employees
    # - Fill missing performance data with department averages
    # - Create features like 'training_per_year', 'salary_vs_department_avg'
    # - Use ColumnTransformer for robust preprocessing
    
    return hr_data, performance_data, training_data

# hr, performance, training = preprocessing_challenge_1()
```

### Challenge 2: Time Series Feature Engineering

```python
def time_series_preprocessing_challenge():
    """Challenge: Engineer time-based features from transaction data"""
    
    print("\n📅 TIME SERIES PREPROCESSING CHALLENGE")
    print("=" * 50)
    
    # Transaction data over time
    np.random.seed(42)
    transactions = pd.DataFrame({
        'customer_id': np.random.choice(range(1, 101), 2000),
        'transaction_date': pd.date_range('2022-01-01', periods=2000, freq='6H'),
        'amount': np.random.exponential(50, 2000),
        'merchant_category': np.random.choice(['Grocery', 'Gas', 'Restaurant', 'Retail'], 2000)
    })
    
    print("Your Tasks:")
    print("1. Create recency features (days since last transaction)")
    print("2. Create frequency features (transactions per month)")
    print("3. Create monetary features (spending patterns)")
    print("4. Create seasonality features (day of week, hour, month)")
    print("5. Create trend features (spending increasing/decreasing?)")
    print("6. Handle different time zones and holidays")
    
    # TODO: Implement comprehensive time series feature engineering
    # Goal: Predict if customer will make a large purchase (>$200) next month
    
    return transactions

# transactions = time_series_preprocessing_challenge()
```

## 🎯 Preprocessing Best Practices

### 1. **The Preprocessing Checklist**

```python
def preprocessing_checklist(data, target_column=None):
    """Comprehensive preprocessing quality check"""
    
    print("✅ PREPROCESSING QUALITY CHECKLIST")
    print("=" * 45)
    
    # Data Quality Checks
    print("1. Data Quality Assessment:")
    print(f"   ✓ Dataset shape: {data.shape}")
    print(f"   ✓ Missing values: {data.isnull().sum().sum()}")
    print(f"   ✓ Duplicate rows: {data.duplicated().sum()}")
    
    # Feature Type Analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    datetime_cols = data.select_dtypes(include=['datetime']).columns
    
    print(f"\n2. Feature Type Distribution:")
    print(f"   ✓ Numerical features: {len(numeric_cols)}")
    print(f"   ✓ Categorical features: {len(categorical_cols)}")
    print(f"   ✓ DateTime features: {len(datetime_cols)}")
    
    # Scale Analysis
    if len(numeric_cols) > 0:
        scale_ranges = data[numeric_cols].max() - data[numeric_cols].min()
        print(f"\n3. Scale Analysis (Range of values):")
        for col in numeric_cols[:5]:  # Show first 5
            print(f"   ✓ {col}: {scale_ranges[col]:.2f}")
        
        max_range = scale_ranges.max()
        min_range = scale_ranges.min()
        if max_range / min_range > 100:
            print(f"   ⚠️ WARNING: Large scale differences detected!")
            print(f"   💡 Recommendation: Apply feature scaling")
    
    # Cardinality Analysis
    if len(categorical_cols) > 0:
        print(f"\n4. Categorical Cardinality:")
        for col in categorical_cols[:5]:  # Show first 5
            unique_count = data[col].nunique()
            print(f"   ✓ {col}: {unique_count} unique values")
            if unique_count > 50:
                print(f"     ⚠️ High cardinality - consider grouping rare categories")
    
    # Target Analysis (if provided)
    if target_column and target_column in data.columns:
        print(f"\n5. Target Variable Analysis:")
        if data[target_column].dtype in ['int64', 'float64']:
            print(f"   ✓ Type: Numerical (Regression problem)")
            print(f"   ✓ Range: {data[target_column].min():.2f} to {data[target_column].max():.2f}")
            print(f"   ✓ Distribution: Mean={data[target_column].mean():.2f}, Std={data[target_column].std():.2f}")
        else:
            print(f"   ✓ Type: Categorical (Classification problem)")
            print(f"   ✓ Classes: {data[target_column].nunique()}")
            print(f"   ✓ Distribution:")
            print(data[target_column].value_counts().head())
    
    print(f"\n✅ PREPROCESSING READINESS SCORE")
    
    # Calculate readiness score
    score = 100
    if data.isnull().sum().sum() > 0:
        score -= 20
        print("   - 20 points: Missing values detected")
    if len(categorical_cols) > 0:
        score -= 15
        print("   - 15 points: Categorical encoding needed")
    if len(numeric_cols) > 1 and max_range / min_range > 100:
        score -= 15
        print("   - 15 points: Feature scaling needed")
    
    print(f"   FINAL SCORE: {score}/100")
    
    if score >= 85:
        print("   🎉 Data is ML-ready!")
    elif score >= 70:
        print("   ⚠️ Minor preprocessing needed")
    else:
        print("   🚨 Significant preprocessing required")

# Example usage
preprocessing_checklist(integrated_customer_data, 'customer_value_score')
```

### 2. **Preprocessing Pipeline Template**

```python
def create_preprocessing_template():
    """Template for standardized preprocessing pipeline"""
    
    print("\n🏭 PREPROCESSING PIPELINE TEMPLATE")
    print("=" * 45)
    
    template_code = '''
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create a standardized preprocessing pipeline
    
    Args:
        numeric_features: List of numerical column names
        categorical_features: List of categorical column names
    
    Returns:
        Fitted preprocessing pipeline
    """
    
    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing  
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Usage example:
# numeric_cols = ['age', 'income', 'experience']
# categorical_cols = ['department', 'education', 'city']
# preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
# X_processed = preprocessor.fit_transform(X_train)
    '''
    
    print("📋 Standard Preprocessing Template:")
    print(template_code)
    
    print("✅ This template handles:")
    print("   • Missing values (median for numbers, 'Unknown' for categories)")
    print("   • Feature scaling (StandardScaler for normal distribution)")
    print("   • Categorical encoding (One-hot with drop='first')")
    print("   • Unknown categories (handle_unknown='ignore')")
    
    print("\n💡 Customization Tips:")
    print("   • Use RobustScaler for data with outliers")
    print("   • Use 'mean' imputation for normally distributed data")
    print("   • Add PolynomialFeatures for feature interactions")
    print("   • Use SelectKBest for feature selection")

create_preprocessing_template()
```

## 🎯 Key Preprocessing Principles

### 1. **Order Matters**
```
Correct Order:
1. Handle missing values
2. Remove outliers  
3. Encode categories
4. Scale features
5. Engineer new features
6. Select best features
```

### 2. **Train/Test Separation**
```python
# ✅ Correct: Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training
X_test_scaled = scaler.transform(X_test)        # Transform test

# ❌ Wrong: Fitting on all data causes data leakage
# scaler.fit(pd.concat([X_train, X_test]))  # DON'T DO THIS!
```

### 3. **Business Logic First**
```python
# Use domain knowledge for better preprocessing
def business_aware_imputation(df):
    # Income missing → Use department/education to estimate
    # Performance missing → Use peer group average
    # Missing category → Create 'Unknown' category (might be meaningful)
    pass
```

## 🎯 Preprocessing Performance Tips

### 1. **Memory Optimization**

```python
def optimize_preprocessing_memory():
    """Optimize memory usage during preprocessing"""
    
    print("⚡ MEMORY OPTIMIZATION TIPS")
    print("=" * 35)
    
    # Tip 1: Use appropriate data types
    print("1. Optimize data types:")
    print("   • int8 for small integers (0-255)")
    print("   • float32 instead of float64 when precision allows")
    print("   • category dtype for repeated strings")
    
    # Example
    data = pd.DataFrame({
        'age': np.random.randint(18, 100, 10000),
        'rating': np.random.randint(1, 6, 10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    print(f"\nBefore optimization: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Optimize
    data['age'] = data['age'].astype('int8')
    data['rating'] = data['rating'].astype('int8') 
    data['category'] = data['category'].astype('category')
    
    print(f"After optimization: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Tip 2: Process in chunks for large datasets
    print(f"\n2. Chunk processing for large datasets:")
    print("   • Use pd.read_csv(chunksize=10000)")
    print("   • Process each chunk separately")
    print("   • Combine results efficiently")

optimize_preprocessing_memory()
```

### 2. **Preprocessing Validation**

```python
def validate_preprocessing(X_original, X_processed, preprocessor=None):
    """Validate preprocessing results"""
    
    print("🔍 PREPROCESSING VALIDATION")
    print("=" * 35)
    
    print(f"Original shape: {X_original.shape}")
    if hasattr(X_processed, 'shape'):
        print(f"Processed shape: {X_processed.shape}")
    else:
        print(f"Processed shape: {len(X_processed)}")
    
    # Check for missing values
    if hasattr(X_processed, 'isnull'):
        missing_after = X_processed.isnull().sum().sum()
    else:
        missing_after = np.isnan(X_processed).sum()
    
    print(f"Missing values after preprocessing: {missing_after}")
    
    # Check feature scaling
    if hasattr(X_processed, 'mean'):
        means = X_processed.mean()
        stds = X_processed.std()
        print(f"Feature means (should be ~0 if StandardScaled): {means.mean():.3f}")
        print(f"Feature stds (should be ~1 if StandardScaled): {stds.mean():.3f}")
    
    print("✅ Preprocessing validation complete")

# Example validation
validate_preprocessing(messy_business_data, X_train_processed)
```

## 🎯 Key Preprocessing Concepts Summary

1. **Feature Scaling**: Make all features comparable (StandardScaler, MinMaxScaler, RobustScaler)
2. **Categorical Encoding**: Convert text to numbers (LabelEncoder, OneHotEncoder)
3. **Missing Value Handling**: Fill gaps intelligently (SimpleImputer, KNNImputer, business logic)
4. **Feature Engineering**: Create new features from existing ones
5. **Pipeline Creation**: Automate preprocessing for consistency
6. **Validation**: Always check preprocessing results

## 🚀 What's Next?

You've mastered data preprocessing! Next up: **Model Selection & Evaluation** - learn how to choose the right algorithm and measure its performance reliably.

**Key skills unlocked:**
- ✅ Feature scaling and normalization
- ✅ Categorical variable encoding
- ✅ Missing data strategies
- ✅ Feature engineering techniques
- ✅ Preprocessing pipeline creation
- ✅ Data validation and quality checks

Ready to build and evaluate machine learning models? Let's dive into **Model Selection & Evaluation**! 🎯
