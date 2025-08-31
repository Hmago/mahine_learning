# Data Cleaning: Turning Chaos into Gold

## 🤔 Why Is Data Cleaning So Important?

Imagine you're a chef preparing a meal, but your ingredients are:
- **Moldy vegetables** (corrupted data)
- **Mislabeled containers** (wrong data types)
- **Missing spices** (missing values)
- **Different units** (inconsistent formats)

No matter how skilled you are, you can't cook a great meal with bad ingredients. **Data cleaning is like washing, chopping, and organizing your ingredients before cooking.**

**The harsh reality**: Data scientists spend 60-80% of their time cleaning data. Master this, and you'll be incredibly valuable!

## 🎯 Common Data Problems (And How to Fix Them)

### 1. Missing Values: The Universal Problem

```python
import pandas as pd
import numpy as np

# Create realistic messy data
np.random.seed(42)
messy_customers = pd.DataFrame({
    'customer_id': range(1, 1001),
    'name': [f'Customer_{i}' if i % 10 != 0 else None for i in range(1, 1001)],  # 10% missing
    'age': [np.random.randint(18, 80) if np.random.rand() > 0.05 else None for _ in range(1000)],  # 5% missing
    'income': [np.random.normal(60000, 20000) if np.random.rand() > 0.15 else None for _ in range(1000)],  # 15% missing
    'email': [f'user{i}@email.com' if i % 7 != 0 else None for i in range(1, 1001)],  # ~14% missing
    'phone': [f'555-{np.random.randint(1000, 9999)}' if np.random.rand() > 0.3 else None for _ in range(1000)]  # 30% missing
})

# Assess the damage
print("MISSING DATA ASSESSMENT")
print("=" * 35)
missing_summary = messy_customers.isnull().sum()
missing_percentage = (missing_summary / len(messy_customers) * 100).round(2)

for col, count, pct in zip(missing_summary.index, missing_summary.values, missing_percentage.values):
    if count > 0:
        print(f"{col}: {count} missing ({pct}%)")

print(f"\nTotal data points: {messy_customers.size:,}")
print(f"Missing data points: {messy_customers.isnull().sum().sum():,}")
```

### Strategy 1: Smart Missing Value Handling

```python
def handle_missing_values(df):
    """Intelligent missing value handling strategies"""
    cleaned = df.copy()
    
    # Strategy 1: Remove rows with critical missing info
    # If customer_id is missing, the row is useless
    cleaned = cleaned.dropna(subset=['customer_id'])
    print(f"Removed {len(df) - len(cleaned)} rows with missing customer_id")
    
    # Strategy 2: Fill with reasonable defaults
    # Missing names can be filled with placeholder
    cleaned['name'] = cleaned['name'].fillna('Unknown Customer')
    
    # Strategy 3: Fill with statistical measures
    # Missing age: use median age (robust to outliers)
    median_age = cleaned['age'].median()
    cleaned['age'] = cleaned['age'].fillna(median_age)
    
    # Missing income: use age-based estimation
    age_income_median = cleaned.groupby(pd.cut(cleaned['age'], bins=5))['income'].median()
    
    def fill_income_by_age(row):
        if pd.isna(row['income']):
            age_bin = pd.cut([row['age']], bins=5)[0]
            return age_income_median[age_bin]
        return row['income']
    
    cleaned['income'] = cleaned.apply(fill_income_by_age, axis=1)
    
    # Strategy 4: Mark missing values for tracking
    cleaned['phone_available'] = ~cleaned['phone'].isnull()
    cleaned['phone'] = cleaned['phone'].fillna('Not Provided')
    
    # Strategy 5: Drop columns with too much missing data
    if cleaned['email'].isnull().sum() / len(cleaned) > 0.5:  # More than 50% missing
        print("Dropping email column (too much missing data)")
        cleaned = cleaned.drop('email', axis=1)
    else:
        cleaned['email'] = cleaned['email'].fillna('no-email@unknown.com')
    
    print(f"\nCleaning complete!")
    print(f"Remaining missing values: {cleaned.isnull().sum().sum()}")
    
    return cleaned

clean_customers = handle_missing_values(messy_customers)
```

### 2. Duplicate Data: Finding Hidden Copies

```python
def handle_duplicates(df):
    """Comprehensive duplicate detection and handling"""
    
    # Create some intentional duplicates for demonstration
    df_with_dups = df.copy()
    
    # Add exact duplicates
    duplicate_rows = df.sample(50)
    df_with_dups = pd.concat([df_with_dups, duplicate_rows], ignore_index=True)
    
    # Add near-duplicates (same customer, slightly different data)
    near_dups = df.sample(30).copy()
    near_dups['phone'] = near_dups['phone'].str.replace('-', '')  # Format change
    df_with_dups = pd.concat([df_with_dups, near_dups], ignore_index=True)
    
    print("DUPLICATE ANALYSIS")
    print("=" * 30)
    
    # Check for exact duplicates
    exact_dups = df_with_dups.duplicated().sum()
    print(f"Exact duplicates: {exact_dups}")
    
    # Check for duplicate customer IDs (business logic duplicates)
    id_dups = df_with_dups['customer_id'].duplicated().sum()
    print(f"Duplicate customer IDs: {id_dups}")
    
    # Remove exact duplicates
    cleaned = df_with_dups.drop_duplicates()
    
    # Handle duplicate customer IDs (keep most recent or most complete)
    cleaned = cleaned.sort_values(['customer_id', 'income'], ascending=[True, False])
    cleaned = cleaned.drop_duplicates(subset=['customer_id'], keep='first')
    
    print(f"Original rows: {len(df_with_dups)}")
    print(f"After deduplication: {len(cleaned)}")
    print(f"Removed: {len(df_with_dups) - len(cleaned)} rows")
    
    return cleaned

deduplicated_data = handle_duplicates(clean_customers)
```

### 3. Data Type Issues: Making Everything Consistent

```python
def fix_data_types(df):
    """Fix common data type issues"""
    fixed = df.copy()
    
    # Common issue 1: Numbers stored as strings
    if 'income' in fixed.columns and fixed['income'].dtype == 'object':
        # Remove currency symbols and convert
        fixed['income'] = (fixed['income']
                          .str.replace('$', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .astype(float))
    
    # Common issue 2: Dates in various formats
    date_columns = ['signup_date', 'last_login', 'birth_date']
    for col in date_columns:
        if col in fixed.columns:
            fixed[col] = pd.to_datetime(fixed[col], errors='coerce')
    
    # Common issue 3: Boolean values as strings
    bool_columns = ['is_premium', 'email_verified', 'phone_verified']
    for col in bool_columns:
        if col in fixed.columns:
            fixed[col] = fixed[col].map({'true': True, 'false': False, 'yes': True, 'no': False})
    
    # Common issue 4: Categorical data as strings
    categorical_columns = ['city', 'country', 'subscription_type']
    for col in categorical_columns:
        if col in fixed.columns and fixed[col].nunique() < 50:  # Reasonable number of categories
            fixed[col] = fixed[col].astype('category')
    
    print("DATA TYPE FIXES APPLIED")
    print("=" * 35)
    print("Data types after fixing:")
    print(fixed.dtypes)
    
    return fixed

# Apply data type fixes
type_fixed_data = fix_data_types(clean_customers)
```

## 🧹 Advanced Cleaning Techniques

### 1. Outlier Detection and Treatment

```python
def handle_outliers(df, column):
    """Detect and handle outliers using IQR method"""
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"OUTLIER ANALYSIS: {column}")
    print("-" * 30)
    print(f"Normal range: ${lower_bound:,.2f} to ${upper_bound:,.2f}")
    print(f"Outliers found: {len(outliers)} ({len(outliers)/len(df):.1%})")
    
    if len(outliers) > 0:
        print("Sample outliers:")
        print(outliers[['customer_id', column]].head())
    
    # Treatment options:
    # Option 1: Remove outliers
    cleaned_removed = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Option 2: Cap outliers (winsorization)
    cleaned_capped = df.copy()
    cleaned_capped[column] = np.clip(cleaned_capped[column], lower_bound, upper_bound)
    
    # Option 3: Log transformation (for right-skewed data)
    cleaned_log = df.copy()
    cleaned_log[f'{column}_log'] = np.log1p(np.maximum(cleaned_log[column], 0))
    
    return {
        'original': df,
        'outliers_removed': cleaned_removed,
        'outliers_capped': cleaned_capped,
        'log_transformed': cleaned_log,
        'outlier_info': {
            'count': len(outliers),
            'percentage': len(outliers)/len(df) * 100,
            'bounds': (lower_bound, upper_bound)
        }
    }

# Analyze income outliers
outlier_analysis = handle_outliers(clean_customers, 'income')
```

### 2. String Cleaning and Standardization

```python
def clean_text_data(df):
    """Comprehensive text data cleaning"""
    cleaned = df.copy()
    
    # Clean names
    if 'name' in cleaned.columns:
        cleaned['name_clean'] = (cleaned['name']
                                .str.strip()           # Remove spaces
                                .str.title()           # Proper case
                                .str.replace(r'\s+', ' ', regex=True))  # Multiple spaces → single space
    
    # Clean and validate emails
    if 'email' in cleaned.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        cleaned['email_valid'] = cleaned['email'].str.match(email_pattern, na=False)
        cleaned['email_clean'] = cleaned['email'].str.lower().str.strip()
    
    # Clean phone numbers
    if 'phone' in cleaned.columns:
        cleaned['phone_clean'] = (cleaned['phone']
                                 .str.replace(r'[^\d]', '', regex=True)  # Keep only digits
                                 .str.replace(r'^1', '', regex=True))    # Remove country code
        
        # Validate phone numbers (US format)
        cleaned['phone_valid'] = cleaned['phone_clean'].str.len() == 10
    
    # Standardize addresses
    if 'address' in cleaned.columns:
        # Standardize common abbreviations
        address_replacements = {
            r'\bSt\b': 'Street',
            r'\bAve\b': 'Avenue',
            r'\bDr\b': 'Drive',
            r'\bRd\b': 'Road'
        }
        
        cleaned['address_clean'] = cleaned['address']
        for pattern, replacement in address_replacements.items():
            cleaned['address_clean'] = cleaned['address_clean'].str.replace(
                pattern, replacement, regex=True, case=False
            )
    
    # Report cleaning results
    print("TEXT CLEANING RESULTS")
    print("=" * 30)
    
    if 'email_valid' in cleaned.columns:
        valid_emails = cleaned['email_valid'].sum()
        print(f"Valid emails: {valid_emails}/{len(cleaned)} ({valid_emails/len(cleaned):.1%})")
    
    if 'phone_valid' in cleaned.columns:
        valid_phones = cleaned['phone_valid'].sum()
        print(f"Valid phones: {valid_phones}/{len(cleaned)} ({valid_phones/len(cleaned):.1%})")
    
    return cleaned

# Apply text cleaning
text_cleaned_data = clean_text_data(clean_customers)
```

### 3. Data Consistency and Validation

```python
def validate_business_rules(df):
    """Apply business logic validation"""
    validated = df.copy()
    
    # Business rule validations
    validation_results = {}
    
    # Rule 1: Age should be between 18 and 100
    age_invalid = (validated['age'] < 18) | (validated['age'] > 100)
    validation_results['invalid_age'] = age_invalid.sum()
    
    # Rule 2: Income should be positive and reasonable
    income_invalid = (validated['income'] < 0) | (validated['income'] > 1000000)
    validation_results['invalid_income'] = income_invalid.sum()
    
    # Rule 3: Spending shouldn't exceed income
    spending_over_income = validated['spending'] > validated['income']
    validation_results['spending_over_income'] = spending_over_income.sum()
    
    # Rule 4: Customer ID format validation
    if 'customer_id' in validated.columns:
        # Assuming format should be numeric
        try:
            validated['customer_id'] = pd.to_numeric(validated['customer_id'])
            validation_results['invalid_customer_id'] = 0
        except:
            validation_results['invalid_customer_id'] = "Format issues detected"
    
    # Report validation results
    print("BUSINESS RULE VALIDATION")
    print("=" * 35)
    for rule, issues in validation_results.items():
        print(f"{rule}: {issues} issues")
    
    # Create flags for problematic records
    validated['data_quality_score'] = (
        (~age_invalid).astype(int) +
        (~income_invalid).astype(int) +
        (~spending_over_income).astype(int)
    )
    
    # Flag records needing review
    validated['needs_review'] = validated['data_quality_score'] < 3
    
    print(f"\nRecords needing review: {validated['needs_review'].sum()}")
    
    return validated

validated_data = validate_business_rules(text_cleaned_data)
```

## 🎯 Real-World Cleaning Pipeline

### Complete Customer Data Cleaning Pipeline

```python
def comprehensive_cleaning_pipeline(raw_df):
    """Complete data cleaning pipeline for customer data"""
    
    print("🧹 STARTING COMPREHENSIVE DATA CLEANING PIPELINE")
    print("=" * 55)
    
    # Step 1: Initial assessment
    print(f"Starting with {len(raw_df)} records")
    
    # Step 2: Remove completely empty rows
    cleaned = raw_df.dropna(how='all')
    print(f"After removing empty rows: {len(cleaned)} records")
    
    # Step 3: Handle critical missing values
    # Remove rows missing essential identifiers
    essential_columns = ['customer_id']
    for col in essential_columns:
        if col in cleaned.columns:
            before_count = len(cleaned)
            cleaned = cleaned.dropna(subset=[col])
            print(f"After removing missing {col}: {len(cleaned)} records ({before_count - len(cleaned)} removed)")
    
    # Step 4: Data type corrections
    if 'age' in cleaned.columns:
        cleaned['age'] = pd.to_numeric(cleaned['age'], errors='coerce')
    
    if 'income' in cleaned.columns:
        cleaned['income'] = pd.to_numeric(cleaned['income'], errors='coerce')
    
    # Step 5: Handle remaining missing values intelligently
    if 'age' in cleaned.columns:
        cleaned['age'] = cleaned['age'].fillna(cleaned['age'].median())
    
    if 'income' in cleaned.columns:
        # Fill income based on age correlation
        age_groups = pd.cut(cleaned['age'], bins=5, labels=False)
        for group in range(5):
            mask = (age_groups == group) & cleaned['income'].isnull()
            if mask.any():
                group_median = cleaned[age_groups == group]['income'].median()
                cleaned.loc[mask, 'income'] = group_median
    
    # Step 6: Remove obvious outliers
    if 'income' in cleaned.columns:
        # Remove negative incomes
        cleaned = cleaned[cleaned['income'] >= 0]
        
        # Cap extremely high incomes (likely data errors)
        income_99th = cleaned['income'].quantile(0.99)
        cleaned['income'] = np.clip(cleaned['income'], 0, income_99th)
    
    # Step 7: Standardize text data
    text_columns = cleaned.select_dtypes(include=['object']).columns
    for col in text_columns:
        if col not in ['customer_id']:  # Don't modify IDs
            cleaned[col] = cleaned[col].str.strip().str.title()
    
    # Step 8: Create data quality indicators
    cleaned['record_completeness'] = (
        cleaned.notna().sum(axis=1) / len(cleaned.columns)
    )
    
    # Step 9: Final quality report
    print("\n📊 FINAL QUALITY REPORT")
    print("-" * 30)
    print(f"Final dataset: {len(cleaned)} records")
    print(f"Data completeness: {cleaned['record_completeness'].mean():.1%}")
    print(f"High quality records (>90% complete): {(cleaned['record_completeness'] > 0.9).sum()}")
    
    # Missing data summary
    remaining_missing = cleaned.isnull().sum()
    if remaining_missing.sum() > 0:
        print("\nRemaining missing values:")
        for col, count in remaining_missing[remaining_missing > 0].items():
            print(f"  {col}: {count} ({count/len(cleaned):.1%})")
    else:
        print("✅ No missing values remaining!")
    
    return cleaned

# Run the complete pipeline
final_clean_data = comprehensive_cleaning_pipeline(messy_customers)
```

## 🎮 Hands-On Exercise: E-commerce Data Cleaning

```python
def ecommerce_cleaning_challenge():
    """Real-world e-commerce data cleaning challenge"""
    
    # Create realistic messy e-commerce data
    np.random.seed(42)
    n_orders = 5000
    
    messy_orders = pd.DataFrame({
        'order_id': [f'ORD{i:05d}' if np.random.rand() > 0.02 else None for i in range(1, n_orders + 1)],
        'customer_email': [f'user{i}@email.com' if np.random.rand() > 0.1 else '' for i in range(1, n_orders + 1)],
        'product_name': np.random.choice(['iPhone 13', 'Samsung Galaxy', 'iPad Pro', None], n_orders, p=[0.3, 0.3, 0.3, 0.1]),
        'price': [f'${np.random.uniform(100, 2000):.2f}' if np.random.rand() > 0.05 else 'Invalid' for _ in range(n_orders)],
        'quantity': [np.random.randint(1, 5) if np.random.rand() > 0.03 else -1 for _ in range(n_orders)],
        'order_date': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)) 
                      if np.random.rand() > 0.08 else 'Invalid Date' for _ in range(n_orders)],
        'status': np.random.choice(['Completed', 'Pending', 'Cancelled', 'Unknown'], n_orders),
        'shipping_address': [f'{np.random.randint(100, 9999)} Main St, City, State' 
                           if np.random.rand() > 0.2 else None for _ in range(n_orders)]
    })
    
    print("🛒 E-COMMERCE DATA CLEANING CHALLENGE")
    print("=" * 45)
    print(f"Starting with {len(messy_orders)} orders")
    
    # Step 1: Remove orders without IDs
    cleaned_orders = messy_orders.dropna(subset=['order_id'])
    print(f"After removing orders without ID: {len(cleaned_orders)}")
    
    # Step 2: Clean price data
    def clean_price(price_str):
        if isinstance(price_str, str) and price_str.startswith('$'):
            try:
                return float(price_str[1:])
            except:
                return None
        return None
    
    cleaned_orders['price_clean'] = cleaned_orders['price'].apply(clean_price)
    cleaned_orders = cleaned_orders.dropna(subset=['price_clean'])
    print(f"After cleaning prices: {len(cleaned_orders)}")
    
    # Step 3: Fix quantity issues
    cleaned_orders = cleaned_orders[cleaned_orders['quantity'] > 0]
    print(f"After removing invalid quantities: {len(cleaned_orders)}")
    
    # Step 4: Clean dates
    cleaned_orders['order_date_clean'] = pd.to_datetime(cleaned_orders['order_date'], errors='coerce')
    cleaned_orders = cleaned_orders.dropna(subset=['order_date_clean'])
    print(f"After cleaning dates: {len(cleaned_orders)}")
    
    # Step 5: Calculate total order value
    cleaned_orders['total_value'] = cleaned_orders['price_clean'] * cleaned_orders['quantity']
    
    # Step 6: Email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    cleaned_orders['email_valid'] = cleaned_orders['customer_email'].str.match(email_pattern, na=False)
    
    # Step 7: Final quality assessment
    quality_metrics = {
        'total_orders': len(cleaned_orders),
        'total_value': cleaned_orders['total_value'].sum(),
        'avg_order_value': cleaned_orders['total_value'].mean(),
        'valid_emails': cleaned_orders['email_valid'].sum(),
        'data_loss': (len(messy_orders) - len(cleaned_orders)) / len(messy_orders) * 100
    }
    
    print(f"\n📊 CLEANING RESULTS")
    print("-" * 25)
    for metric, value in quality_metrics.items():
        if isinstance(value, float):
            if 'value' in metric.lower():
                print(f"{metric}: ${value:,.2f}")
            else:
                print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:,}")
    
    return cleaned_orders

# Run the e-commerce cleaning challenge
clean_ecommerce_data = ecommerce_cleaning_challenge()
```

## 🎯 Automated Data Quality Monitoring

```python
def create_data_quality_dashboard(df):
    """Automated data quality monitoring"""
    
    quality_report = {}
    
    # Completeness metrics
    quality_report['completeness'] = {
        'overall': (1 - df.isnull().sum().sum() / df.size) * 100,
        'by_column': ((1 - df.isnull().sum() / len(df)) * 100).to_dict()
    }
    
    # Consistency metrics
    quality_report['consistency'] = {}
    
    # Check for duplicate customer IDs
    if 'customer_id' in df.columns:
        duplicates = df['customer_id'].duplicated().sum()
        quality_report['consistency']['duplicate_ids'] = duplicates
    
    # Validity metrics
    quality_report['validity'] = {}
    
    # Check numeric ranges
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col == 'age':
            invalid_age = ((df[col] < 0) | (df[col] > 120)).sum()
            quality_report['validity'][f'invalid_{col}'] = invalid_age
        elif col == 'income':
            invalid_income = ((df[col] < 0) | (df[col] > 10000000)).sum()
            quality_report['validity'][f'invalid_{col}'] = invalid_income
    
    # Uniqueness metrics
    quality_report['uniqueness'] = {}
    for col in df.columns:
        uniqueness = df[col].nunique() / len(df) * 100
        quality_report['uniqueness'][col] = uniqueness
    
    # Print quality dashboard
    print("📊 DATA QUALITY DASHBOARD")
    print("=" * 35)
    
    print(f"Overall completeness: {quality_report['completeness']['overall']:.1f}%")
    
    print("\nColumn completeness:")
    for col, completeness in quality_report['completeness']['by_column'].items():
        status = "✅" if completeness > 95 else "⚠️" if completeness > 80 else "❌"
        print(f"  {status} {col}: {completeness:.1f}%")
    
    print("\nData validity issues:")
    for check, issues in quality_report['validity'].items():
        status = "✅" if issues == 0 else "⚠️"
        print(f"  {status} {check}: {issues} records")
    
    return quality_report

# Generate quality report
quality_report = create_data_quality_dashboard(final_clean_data)
```

## 🎯 Key Cleaning Strategies

### 1. The 80/20 Rule
- **80% of your cleaning effort** should focus on the **20% of issues** that affect the most data
- **Prioritize**: Missing values > Outliers > Format issues > Minor inconsistencies

### 2. Domain Knowledge is King
- **Understand your business**: What values make sense?
- **Know your users**: How will they interpret the data?
- **Consider context**: Seasonal patterns, business cycles

### 3. Document Everything
```python
def document_cleaning_process(original_df, cleaned_df, steps_taken):
    """Document the cleaning process for reproducibility"""
    
    documentation = {
        'original_shape': original_df.shape,
        'cleaned_shape': cleaned_df.shape,
        'rows_removed': len(original_df) - len(cleaned_df),
        'columns_added': len(cleaned_df.columns) - len(original_df.columns),
        'steps_performed': steps_taken,
        'data_loss_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100
    }
    
    print("📝 CLEANING DOCUMENTATION")
    print("=" * 30)
    for key, value in documentation.items():
        print(f"{key}: {value}")
    
    return documentation

# Document our cleaning process
cleaning_docs = document_cleaning_process(
    messy_customers, 
    final_clean_data,
    ['missing_value_handling', 'duplicate_removal', 'outlier_treatment', 'validation']
)
```

## 🎯 Key Takeaways

1. **Cleaning is iterative**: Expect multiple passes to get data right
2. **Context matters**: Business rules guide cleaning decisions
3. **Document everything**: Track what you changed and why
4. **Quality over quantity**: Better to have less, clean data than more, dirty data
5. **Automate when possible**: Build reusable cleaning functions

## 🚀 What's Next?

With clean data in hand, you're ready for the fun part: **Data Transformation** - grouping, aggregating, merging, and reshaping data to uncover insights and patterns!
