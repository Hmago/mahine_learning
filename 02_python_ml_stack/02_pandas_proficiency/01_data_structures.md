# Data Structures: DataFrames and Series Explained

## 🤔 What Are DataFrames and Series?

Think of pandas data structures like this:

### Series = A Smart Column
```python
import pandas as pd

# A Series is like a supercharged Excel column
customer_ages = pd.Series([25, 30, 35, 40, 45], 
                         index=['Alice', 'Bob', 'Carol', 'David', 'Emma'])

print(customer_ages)
print(f"Bob's age: {customer_ages['Bob']}")
print(f"Average age: {customer_ages.mean()}")
```

### DataFrame = A Smart Spreadsheet
```python
# A DataFrame is like Excel, but with superpowers
customer_data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 65000, 75000, 80000, 95000],
    'spending': [2000, 2500, 3000, 3200, 4000],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
}, index=['Alice', 'Bob', 'Carol', 'David', 'Emma'])

print(customer_data)
```

## 🎯 Why These Structures Are Powerful

### 1. Named Access (No More Numbers!)

```python
# Instead of remembering column positions
customer_ages = customer_data.iloc[:, 0]  # Which column was age again?

# Use meaningful names
customer_ages = customer_data['age']  # Crystal clear!
customer_names = customer_data.index   # Row names
```

### 2. Automatic Alignment

```python
# Different customers in different orders
customer_a_data = pd.Series([25, 50000], index=['age', 'income'])
customer_b_data = pd.Series([75000, 30], index=['income', 'age'])  # Different order!

# Pandas automatically aligns by label
combined = customer_a_data + customer_b_data  # Works perfectly!
print(combined)
```

### 3. Built-in Data Analysis

```python
# Instant insights about your data
print("Data Overview:")
print(customer_data.info())  # Data types and missing values
print("\nStatistical Summary:")
print(customer_data.describe())  # Mean, std, quartiles, etc.
print("\nValue Counts:")
print(customer_data['city'].value_counts())  # Count occurrences
```

## 🏗️ Building DataFrames: From Chaos to Order

### Creating from Dictionary (Most Common)

```python
# Real-world example: E-commerce order data
orders = pd.DataFrame({
    'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD004'],
    'customer_id': ['CUST001', 'CUST002', 'CUST001', 'CUST003'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'price': [999.99, 25.99, 79.99, 299.99],
    'quantity': [1, 2, 1, 1],
    'order_date': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18']
})

# Convert date column to datetime
orders['order_date'] = pd.to_datetime(orders['order_date'])

# Add calculated columns
orders['total_amount'] = orders['price'] * orders['quantity']

print(orders)
```

### Creating from Lists of Lists

```python
# From raw data (like CSV import)
raw_data = [
    ['Alice', 25, 'Engineer', 75000],
    ['Bob', 30, 'Manager', 85000],
    ['Carol', 35, 'Analyst', 65000]
]

employees = pd.DataFrame(raw_data, 
                        columns=['name', 'age', 'role', 'salary'])
print(employees)
```

### Creating from NumPy Arrays

```python
# Bridge from NumPy to Pandas
np_array = np.random.rand(100, 5)
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

ml_dataset = pd.DataFrame(np_array, columns=feature_names)
ml_dataset['target'] = np.random.choice([0, 1], 100)  # Binary target

print(f"ML Dataset shape: {ml_dataset.shape}")
print(ml_dataset.head())
```

## 🎯 Indexing and Selection: Finding What You Need

### Basic Selection

```python
# Select single column (returns Series)
ages = customer_data['age']

# Select multiple columns (returns DataFrame)
demographics = customer_data[['age', 'income', 'city']]

# Select rows by condition
high_earners = customer_data[customer_data['income'] > 70000]
young_customers = customer_data[customer_data['age'] < 35]

print(f"High earners: {len(high_earners)} customers")
print(f"Young customers: {len(young_customers)} customers")
```

### Advanced Selection with .loc and .iloc

```python
# Label-based selection with .loc
alice_data = customer_data.loc['Alice']  # All data for Alice
age_income = customer_data.loc[:, ['age', 'income']]  # All rows, specific columns
subset = customer_data.loc['Bob':'David', 'age':'spending']  # Range selection

# Position-based selection with .iloc
first_customer = customer_data.iloc[0]  # First row
first_three = customer_data.iloc[:3]    # First three rows
last_column = customer_data.iloc[:, -1] # Last column

print("Alice's complete profile:")
print(alice_data)
```

### Complex Filtering

```python
# Multiple conditions
target_customers = customer_data[
    (customer_data['age'] >= 25) & 
    (customer_data['age'] <= 40) & 
    (customer_data['income'] > 60000) &
    (customer_data['city'].isin(['NYC', 'LA']))
]

print("Target customers (age 25-40, income >60k, in NYC/LA):")
print(target_customers)

# String operations
chicago_customers = customer_data[customer_data['city'].str.contains('Chicago')]
```

## 🎮 Hands-On Exercise: Customer Analysis Dashboard

```python
import pandas as pd
import numpy as np

def create_customer_dashboard():
    # Generate realistic customer dataset
    np.random.seed(42)
    n_customers = 5000
    
    # Create comprehensive customer data
    customers = pd.DataFrame({
        'customer_id': [f'CUST{i:05d}' for i in range(1, n_customers + 1)],
        'first_name': np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa'], n_customers),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson'], n_customers),
        'age': np.random.randint(18, 75, n_customers),
        'income': np.random.normal(60000, 25000, n_customers),
        'spending': np.random.normal(3000, 1200, n_customers),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'signup_date': pd.date_range('2020-01-01', periods=n_customers, freq='D'),
        'is_premium': np.random.choice([True, False], n_customers, p=[0.2, 0.8])
    })
    
    # Clean the data
    customers['income'] = np.clip(customers['income'], 20000, 200000)  # Realistic range
    customers['spending'] = np.clip(customers['spending'], 500, 10000)
    customers['full_name'] = customers['first_name'] + ' ' + customers['last_name']
    
    # Analysis 1: Customer segments
    print("🎯 CUSTOMER DASHBOARD")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total customers: {len(customers):,}")
    print(f"Premium customers: {customers['is_premium'].sum():,} ({customers['is_premium'].mean():.1%})")
    print(f"Average age: {customers['age'].mean():.1f}")
    print(f"Average income: ${customers['income'].mean():,.2f}")
    print(f"Average spending: ${customers['spending'].mean():,.2f}")
    
    # City analysis
    print("\n🏙️ GEOGRAPHIC DISTRIBUTION")
    city_stats = customers.groupby('city').agg({
        'customer_id': 'count',
        'income': 'mean',
        'spending': 'mean',
        'is_premium': 'mean'
    }).round(2)
    city_stats.columns = ['customer_count', 'avg_income', 'avg_spending', 'premium_rate']
    print(city_stats)
    
    # Age group analysis
    print("\n👥 AGE GROUP ANALYSIS")
    customers['age_group'] = pd.cut(customers['age'], 
                                   bins=[0, 30, 45, 60, 100], 
                                   labels=['18-30', '31-45', '46-60', '60+'])
    
    age_analysis = customers.groupby('age_group').agg({
        'customer_id': 'count',
        'income': 'mean',
        'spending': 'mean'
    }).round(2)
    print(age_analysis)
    
    return customers

# Run the dashboard
customer_df = create_customer_dashboard()
```

## 🎯 Advanced DataFrame Operations

### MultiIndex: Hierarchical Data

```python
# Sales data with multiple levels
sales_data = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South'] * 3,
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3', 'Q3'],
    'product': ['A', 'B', 'A', 'B'] * 3,
    'sales': np.random.randint(1000, 5000, 12)
})

# Create MultiIndex
sales_pivot = sales_data.set_index(['region', 'quarter', 'product'])
print("MultiIndex DataFrame:")
print(sales_pivot)

# Access data at different levels
print("\nNorth region data:")
print(sales_pivot.loc['North'])

print("\nQ1 sales across all regions:")
print(sales_pivot.loc[:, 'Q1', :])
```

### Categorical Data: Memory Efficiency

```python
# For repeated text values, use categories
large_dataset = pd.DataFrame({
    'customer_id': range(100000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100000)
})

# Convert to categorical (saves memory)
large_dataset['city'] = large_dataset['city'].astype('category')
large_dataset['product_category'] = large_dataset['product_category'].astype('category')

print(f"Memory usage: {large_dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print("Data types:")
print(large_dataset.dtypes)
```

## 🚀 Performance Tips for Large DataFrames

### 1. Efficient Operations

```python
# Vectorized operations (fast)
customer_data['spending_per_income'] = customer_data['spending'] / customer_data['income']

# Use .query() for complex filtering (often faster)
high_value = customer_data.query('income > 70000 and spending > 3000 and age < 50')

# Chain operations efficiently
result = (customer_data
          .query('income > 50000')
          .groupby('city')
          .agg({'spending': 'mean'})
          .sort_values('spending', ascending=False))
```

### 2. Memory Optimization

```python
def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    optimized = df.copy()
    
    # Optimize numeric columns
    for col in optimized.select_dtypes(include=['int64']).columns:
        optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
    
    for col in optimized.select_dtypes(include=['float64']).columns:
        optimized[col] = pd.to_numeric(optimized[col], downcast='float')
    
    # Convert strings to categories if they have few unique values
    for col in optimized.select_dtypes(include=['object']).columns:
        if optimized[col].nunique() / len(optimized) < 0.5:  # Less than 50% unique
            optimized[col] = optimized[col].astype('category')
    
    return optimized

# Test optimization
original_size = customer_data.memory_usage(deep=True).sum()
optimized_data = optimize_dataframe(customer_data)
optimized_size = optimized_data.memory_usage(deep=True).sum()

print(f"Original size: {original_size / 1024:.1f} KB")
print(f"Optimized size: {optimized_size / 1024:.1f} KB")
print(f"Memory saved: {(1 - optimized_size/original_size):.1%}")
```

## 🎯 Key Operations You'll Use Daily

### Data Exploration

```python
# Essential exploration commands
def explore_dataset(df):
    print("DATASET EXPLORATION")
    print("=" * 40)
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types and memory
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing data
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Unique values
    print("\nUnique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Sample data
    print("\nSample Data:")
    print(df.head())
    
    return df

# Explore our customer dataset
explore_dataset(customer_data)
```

### Data Filtering and Selection

```python
# Powerful filtering techniques
def demonstrate_filtering(df):
    # Single condition
    high_earners = df[df['income'] > 70000]
    
    # Multiple conditions
    target_segment = df[
        (df['age'] >= 25) & 
        (df['age'] <= 45) & 
        (df['income'] > 60000)
    ]
    
    # String operations
    city_filter = df[df['city'].str.contains('C')]  # Cities starting with C
    
    # Using .query() method (more readable for complex conditions)
    complex_filter = df.query('income > 60000 and spending > 2500 and age < 50')
    
    # Using .isin() for multiple values
    major_cities = df[df['city'].isin(['NYC', 'LA', 'Chicago'])]
    
    print(f"High earners: {len(high_earners)}")
    print(f"Target segment: {len(target_segment)}")
    print(f"Cities with 'C': {len(city_filter)}")
    print(f"Complex filter: {len(complex_filter)}")
    print(f"Major cities: {len(major_cities)}")

demonstrate_filtering(customer_data)
```

## 🎭 Real-World Business Scenarios

### 1. Customer Churn Analysis

```python
def customer_churn_analysis():
    # Create realistic customer dataset with churn indicators
    np.random.seed(42)
    n_customers = 10000
    
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'tenure_months': np.random.exponential(12, n_customers),
        'monthly_charges': np.random.normal(70, 20, n_customers),
        'total_charges': np.random.normal(1500, 800, n_customers),
        'contract_type': np.random.choice(['Month-to-month', '1-year', '2-year'], n_customers),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Check'], n_customers),
        'churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    })
    
    # Churn analysis
    print("CHURN ANALYSIS DASHBOARD")
    print("=" * 40)
    
    # Overall churn rate
    churn_rate = customers['churned'].mean()
    print(f"Overall churn rate: {churn_rate:.1%}")
    
    # Churn by contract type
    churn_by_contract = customers.groupby('contract_type')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_contract.columns = ['total_customers', 'churned_customers', 'churn_rate']
    print("\nChurn by Contract Type:")
    print(churn_by_contract)
    
    # High-risk customers (high charges, short tenure, month-to-month)
    high_risk = customers[
        (customers['monthly_charges'] > customers['monthly_charges'].quantile(0.75)) &
        (customers['tenure_months'] < 12) &
        (customers['contract_type'] == 'Month-to-month')
    ]
    
    print(f"\nHigh-risk customers identified: {len(high_risk)}")
    print(f"Their churn rate: {high_risk['churned'].mean():.1%}")
    
    return customers

churn_data = customer_churn_analysis()
```

### 2. Sales Performance Tracking

```python
def sales_performance_dashboard():
    # Generate sales data
    np.random.seed(42)
    
    # Sales team performance data
    sales_data = pd.DataFrame({
        'salesperson': np.random.choice(['Alice', 'Bob', 'Carol', 'David', 'Emma'], 1000),
        'sale_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 1000),
        'sale_amount': np.random.exponential(500, 1000),
        'commission_rate': np.random.choice([0.05, 0.08, 0.10], 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
    })
    
    # Add calculated fields
    sales_data['commission'] = sales_data['sale_amount'] * sales_data['commission_rate']
    sales_data['month'] = sales_data['sale_date'].dt.to_period('M')
    
    # Performance analysis
    print("SALES PERFORMANCE DASHBOARD")
    print("=" * 45)
    
    # Top performers
    performance = sales_data.groupby('salesperson').agg({
        'sale_amount': ['sum', 'mean', 'count'],
        'commission': 'sum'
    }).round(2)
    
    performance.columns = ['total_sales', 'avg_sale', 'num_sales', 'total_commission']
    performance = performance.sort_values('total_sales', ascending=False)
    
    print("Salesperson Performance:")
    print(performance)
    
    # Monthly trends
    monthly_sales = sales_data.groupby('month')['sale_amount'].sum()
    print(f"\nBest month: {monthly_sales.idxmax()} (${monthly_sales.max():,.2f})")
    print(f"Worst month: {monthly_sales.idxmin()} (${monthly_sales.min():,.2f})")
    
    return sales_data

sales_df = sales_performance_dashboard()
```

## 🎯 Key Takeaways

1. **DataFrames are powerful spreadsheets**: Named columns, mixed data types, built-in analysis
2. **Series are smart columns**: Automatic alignment, vectorized operations
3. **Indexing is flexible**: Use names (.loc) or positions (.iloc)
4. **Filtering is intuitive**: Boolean conditions work naturally
5. **Everything connects**: DataFrames integrate seamlessly with other tools

## 🚀 What's Next?

Now that you understand the data structures, let's learn how to **Import and Export Data** from every source imaginable - databases, APIs, Excel files, and more!
