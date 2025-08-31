# Data Import/Export: Connecting to the Real World

## 🤔 Why Data Import/Export Matters

In the real world, data doesn't magically appear in clean pandas DataFrames. It comes from:

- **Databases**: Customer records, transaction logs
- **APIs**: Social media feeds, stock prices, weather data
- **Files**: Excel reports, CSV exports, JSON documents
- **Web scraping**: Product prices, reviews, market data

**Pandas is your universal translator** - it speaks every data format fluently.

## 📁 Reading from Files: The Essentials

### 1. CSV Files (Most Common)

```python
import pandas as pd

# Basic CSV reading
customers = pd.read_csv('customers.csv')

# Handle common issues
customers = pd.read_csv('customers.csv',
    encoding='utf-8',           # Handle special characters
    parse_dates=['signup_date'], # Auto-convert date columns
    na_values=['N/A', 'NULL', ''], # Define missing values
    dtype={'customer_id': str}   # Force specific data types
)

# Real-world example: Reading messy sales data
def read_sales_data():
    try:
        sales = pd.read_csv('sales_data.csv',
            sep=';',                    # European CSV format
            decimal=',',                # European decimal separator
            thousands='.',              # Thousands separator
            encoding='latin-1',         # Common encoding for legacy systems
            skiprows=2,                 # Skip header rows
            usecols=['Date', 'Product', 'Amount', 'Customer'],  # Only needed columns
            parse_dates=['Date'],
            date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y')
        )
        return sales
    except FileNotFoundError:
        print("File not found - creating sample data")
        return create_sample_sales_data()

def create_sample_sales_data():
    """Create sample sales data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    return pd.DataFrame({
        'Date': np.random.choice(dates, 1000),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet'], 1000),
        'Amount': np.random.uniform(100, 2000, 1000),
        'Customer': [f'Customer_{i}' for i in np.random.randint(1, 500, 1000)]
    })

sales_data = read_sales_data()
print("Sales data loaded:")
print(sales_data.head())
```

### 2. Excel Files (Business Standard)

```python
# Reading Excel files with multiple sheets
def read_excel_data():
    try:
        # Read specific sheet
        q1_data = pd.read_excel('quarterly_reports.xlsx', sheet_name='Q1_2023')
        
        # Read multiple sheets at once
        all_quarters = pd.read_excel('quarterly_reports.xlsx', sheet_name=None)  # All sheets
        
        # Combine quarterly data
        combined_data = pd.concat(all_quarters.values(), ignore_index=True)
        
        return combined_data
    except FileNotFoundError:
        # Create sample Excel-like data
        return pd.DataFrame({
            'quarter': ['Q1', 'Q2', 'Q3', 'Q4'] * 250,
            'revenue': np.random.normal(100000, 25000, 1000),
            'costs': np.random.normal(60000, 15000, 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
        })

excel_data = read_excel_data()
print("Excel data structure:")
print(excel_data.info())
```

### 3. JSON Data (Web APIs)

```python
# Reading JSON data (common from APIs)
import json

def handle_json_data():
    # Sample API response structure
    api_response = {
        "customers": [
            {"id": 1, "name": "Alice Johnson", "purchases": [
                {"product": "Laptop", "amount": 1200, "date": "2023-01-15"},
                {"product": "Mouse", "amount": 25, "date": "2023-01-16"}
            ]},
            {"id": 2, "name": "Bob Smith", "purchases": [
                {"product": "Phone", "amount": 800, "date": "2023-01-17"}
            ]}
        ]
    }
    
    # Flatten nested JSON into DataFrame
    rows = []
    for customer in api_response['customers']:
        for purchase in customer['purchases']:
            rows.append({
                'customer_id': customer['id'],
                'customer_name': customer['name'],
                'product': purchase['product'],
                'amount': purchase['amount'],
                'purchase_date': purchase['date']
            })
    
    df = pd.DataFrame(rows)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    return df

json_data = handle_json_data()
print("Flattened JSON data:")
print(json_data)
```

## 🌐 Database Connections: The Enterprise Reality

### SQLite (Local Database)

```python
import sqlite3

def work_with_database():
    # Create sample database
    conn = sqlite3.connect(':memory:')  # In-memory database for demo
    
    # Create sample table
    sample_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'name': [f'Customer_{i}' for i in range(1, 1001)],
        'email': [f'customer{i}@email.com' for i in range(1, 1001)],
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'total_spent': np.random.exponential(1000, 1000)
    })
    
    # Write to database
    sample_data.to_sql('customers', conn, index=False, if_exists='replace')
    
    # Read from database with SQL query
    high_value_customers = pd.read_sql_query("""
        SELECT customer_id, name, total_spent 
        FROM customers 
        WHERE total_spent > 2000 
        ORDER BY total_spent DESC 
        LIMIT 10
    """, conn)
    
    print("Top 10 high-value customers from database:")
    print(high_value_customers)
    
    conn.close()
    return high_value_customers

db_data = work_with_database()
```

### PostgreSQL/MySQL (Production Databases)

```python
# For real database connections (when available)
def connect_to_production_db():
    """Template for production database connections"""
    try:
        from sqlalchemy import create_engine
        
        # Database connection string
        connection_string = "postgresql://user:password@localhost:5432/dbname"
        engine = create_engine(connection_string)
        
        # Read with chunking for large tables
        chunk_size = 10000
        chunks = []
        for chunk in pd.read_sql_query("SELECT * FROM large_table", engine, chunksize=chunk_size):
            # Process each chunk
            processed_chunk = chunk.dropna().reset_index(drop=True)
            chunks.append(processed_chunk)
        
        # Combine all chunks
        full_data = pd.concat(chunks, ignore_index=True)
        return full_data
        
    except ImportError:
        print("SQLAlchemy not available - install with: pip install sqlalchemy")
        return None
```

## 🔄 Exporting Data: Sharing Your Insights

### 1. Saving to Different Formats

```python
def export_analysis_results(df):
    # CSV for data sharing
    df.to_csv('customer_analysis.csv', index=False)
    
    # Excel with multiple sheets
    with pd.ExcelWriter('customer_report.xlsx') as writer:
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Summary statistics
        summary = df.describe()
        summary.to_excel(writer, sheet_name='Summary Stats')
        
        # Segment analysis
        segments = df.groupby('city').agg({
            'income': 'mean',
            'spending': 'mean'
        })
        segments.to_excel(writer, sheet_name='City Analysis')
    
    # JSON for web applications
    df.to_json('customer_data.json', orient='records', date_format='iso')
    
    # Parquet for efficient storage (compressed, fast)
    df.to_parquet('customer_data.parquet')
    
    print("Data exported to multiple formats:")
    print("- CSV: customer_analysis.csv")
    print("- Excel: customer_report.xlsx (multiple sheets)")
    print("- JSON: customer_data.json")
    print("- Parquet: customer_data.parquet")

# Export our customer analysis
export_analysis_results(customer_data)
```

### 2. Creating Reports for Stakeholders

```python
def create_executive_report(df):
    """Create an executive summary report"""
    report = {
        'total_customers': len(df),
        'total_revenue_potential': df['income'].sum(),
        'average_customer_value': df['spending'].mean(),
        'geographic_distribution': df['city'].value_counts().to_dict(),
        'age_demographics': {
            'young_adults': len(df[df['age'] < 35]),
            'middle_aged': len(df[(df['age'] >= 35) & (df['age'] < 55)]),
            'seniors': len(df[df['age'] >= 55])
        }
    }
    
    # Convert to DataFrame for easy export
    summary_df = pd.DataFrame([
        ['Total Customers', report['total_customers']],
        ['Total Revenue Potential', f"${report['total_revenue_potential']:,.2f}"],
        ['Average Customer Value', f"${report['average_customer_value']:,.2f}"],
        ['Primary Market', max(report['geographic_distribution'], 
                             key=report['geographic_distribution'].get)]
    ], columns=['Metric', 'Value'])
    
    print("EXECUTIVE SUMMARY")
    print("=" * 30)
    print(summary_df.to_string(index=False))
    
    return summary_df

executive_summary = create_executive_report(customer_data)
```

## 🚀 Advanced Import/Export Techniques

### 1. Handling Large Files

```python
def process_large_file(filename, chunk_size=10000):
    """Process large CSV files in chunks"""
    chunk_list = []
    
    # Process in chunks to manage memory
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = chunk.dropna()  # Remove missing values
        processed_chunk = processed_chunk[processed_chunk['amount'] > 0]  # Valid amounts
        
        chunk_list.append(processed_chunk)
    
    # Combine all chunks
    full_dataset = pd.concat(chunk_list, ignore_index=True)
    return full_dataset

# Example with generated large dataset
def create_and_process_large_dataset():
    # Create large CSV for demonstration
    large_data = pd.DataFrame({
        'transaction_id': range(1, 100001),
        'customer_id': np.random.randint(1, 10000, 100000),
        'amount': np.random.exponential(100, 100000),
        'date': pd.date_range('2020-01-01', periods=100000, freq='H')
    })
    
    # Save to CSV
    large_data.to_csv('large_transactions.csv', index=False)
    print(f"Created large dataset: {len(large_data):,} rows")
    
    # Process in chunks
    processed_data = process_large_file('large_transactions.csv')
    print(f"Processed dataset: {len(processed_data):,} rows")
    
    return processed_data

# large_processed = create_and_process_large_dataset()
```

### 2. API Data Integration

```python
import requests
import json

def fetch_api_data():
    """Template for fetching data from APIs"""
    # Example: Fetching cryptocurrency prices (mock API)
    def mock_crypto_api():
        """Simulate API response"""
        return {
            "data": [
                {"symbol": "BTC", "price": 45000, "change_24h": 2.5},
                {"symbol": "ETH", "price": 3200, "change_24h": -1.2},
                {"symbol": "ADA", "price": 1.5, "change_24h": 5.8}
            ]
        }
    
    # In real scenario:
    # response = requests.get('https://api.crypto.com/v2/public/get-ticker')
    # data = response.json()
    
    # For demo, use mock data
    api_data = mock_crypto_api()
    
    # Convert to DataFrame
    crypto_df = pd.DataFrame(api_data['data'])
    crypto_df['last_updated'] = pd.Timestamp.now()
    
    print("Cryptocurrency data from API:")
    print(crypto_df)
    
    return crypto_df

api_data = fetch_api_data()
```

## 🎯 Data Quality Assurance

### Validation During Import

```python
def robust_data_import(filename):
    """Import data with comprehensive validation"""
    try:
        # Read with error handling
        df = pd.read_csv(filename)
        
        print(f"✅ Successfully loaded {len(df)} rows")
        
        # Data quality checks
        print("\n🔍 DATA QUALITY REPORT")
        print("-" * 30)
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print("⚠️  Missing values found:")
            print(missing_data[missing_data > 0])
        else:
            print("✅ No missing values")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  {duplicates} duplicate rows found")
        else:
            print("✅ No duplicate rows")
        
        # Check data types
        print("\n📊 Data Types:")
        print(df.dtypes)
        
        # Basic statistics
        print("\n📈 Quick Statistics:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Example usage
sample_df = customer_data  # Use our existing data
print("Data quality check results:")
validation_result = robust_data_import if hasattr(robust_data_import, '__call__') else None
```

### Automated Data Type Detection

```python
def smart_data_types(df):
    """Automatically optimize data types"""
    optimized = df.copy()
    
    # Convert string columns to categories if appropriate
    for col in optimized.select_dtypes(include=['object']):
        if optimized[col].nunique() / len(optimized) < 0.5:  # Less than 50% unique
            optimized[col] = optimized[col].astype('category')
            print(f"✅ Converted '{col}' to category")
    
    # Downcast numeric types
    for col in optimized.select_dtypes(include=['int64']):
        optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
    
    for col in optimized.select_dtypes(include=['float64']):
        optimized[col] = pd.to_numeric(optimized[col], downcast='float')
    
    # Memory usage comparison
    original_memory = df.memory_usage(deep=True).sum()
    optimized_memory = optimized.memory_usage(deep=True).sum()
    savings = (1 - optimized_memory / original_memory) * 100
    
    print(f"\n💾 Memory optimization:")
    print(f"Original: {original_memory / 1024:.1f} KB")
    print(f"Optimized: {optimized_memory / 1024:.1f} KB")
    print(f"Savings: {savings:.1f}%")
    
    return optimized

optimized_customers = smart_data_types(customer_data)
```

## 🔄 Real-World Integration Examples

### 1. Multi-Source Data Integration

```python
def integrate_multiple_sources():
    """Combine data from multiple sources"""
    
    # Source 1: Customer demographics (CSV)
    demographics = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'age': [25, 30, 35, 40],
        'income': [50000, 65000, 75000, 80000],
        'city': ['NYC', 'LA', 'Chicago', 'Houston']
    })
    
    # Source 2: Purchase history (Database simulation)
    purchases = pd.DataFrame({
        'customer_id': ['C001', 'C001', 'C002', 'C003', 'C003', 'C004'],
        'product': ['Laptop', 'Mouse', 'Phone', 'Tablet', 'Case', 'Monitor'],
        'amount': [1200, 25, 800, 600, 50, 300],
        'purchase_date': pd.date_range('2023-01-01', periods=6, freq='W')
    })
    
    # Source 3: Customer support interactions (Excel simulation)
    support = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C004'],
        'support_tickets': [2, 1, 3],
        'satisfaction_score': [4.5, 5.0, 3.5]
    })
    
    # Integrate all sources
    # Start with demographics as base
    integrated = demographics.copy()
    
    # Add purchase summary
    purchase_summary = purchases.groupby('customer_id').agg({
        'amount': ['sum', 'count', 'mean'],
        'purchase_date': 'max'
    }).round(2)
    
    # Flatten column names
    purchase_summary.columns = ['total_spent', 'num_purchases', 'avg_purchase', 'last_purchase']
    purchase_summary = purchase_summary.reset_index()
    
    # Merge data sources
    integrated = integrated.merge(purchase_summary, on='customer_id', how='left')
    integrated = integrated.merge(support, on='customer_id', how='left')
    
    # Fill missing values
    integrated['support_tickets'] = integrated['support_tickets'].fillna(0)
    integrated['satisfaction_score'] = integrated['satisfaction_score'].fillna(integrated['satisfaction_score'].mean())
    
    print("INTEGRATED CUSTOMER VIEW")
    print("=" * 40)
    print(integrated)
    
    return integrated

integrated_data = integrate_multiple_sources()
```

### 2. Real-Time Data Processing

```python
def process_streaming_data():
    """Simulate processing streaming data"""
    
    def generate_real_time_data():
        """Simulate real-time transaction data"""
        return pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'customer_id': [f'C{np.random.randint(1, 1000):03d}'],
            'transaction_amount': [np.random.exponential(100)],
            'product_category': [np.random.choice(['Electronics', 'Clothing', 'Books'])],
            'payment_method': [np.random.choice(['Credit', 'Debit', 'Cash'])]
        })
    
    # Process incoming data streams
    processed_batches = []
    
    for batch in range(5):  # Simulate 5 data batches
        # Get new data batch
        new_data = generate_real_time_data()
        
        # Process batch
        new_data['batch_id'] = batch
        new_data['processed_time'] = pd.Timestamp.now()
        
        processed_batches.append(new_data)
        print(f"Processed batch {batch + 1}")
    
    # Combine all processed batches
    streaming_data = pd.concat(processed_batches, ignore_index=True)
    
    print("\nStreaming data processing complete:")
    print(streaming_data)
    
    return streaming_data

streaming_result = process_streaming_data()
```

## 🎯 Export Strategies for Different Audiences

### 1. For Data Scientists (Technical)

```python
def export_for_data_scientists(df):
    """Export in formats preferred by data scientists"""
    
    # Parquet: Fast, compressed, preserves data types
    df.to_parquet('data_for_analysis.parquet')
    
    # HDF5: Great for large datasets with metadata
    df.to_hdf('data_store.h5', key='customers', mode='w')
    
    # Pickle: Preserves everything (Python-specific)
    df.to_pickle('customer_data.pkl')
    
    print("Technical exports complete:")
    print("- Parquet: Fast loading for analysis")
    print("- HDF5: Efficient for large datasets")
    print("- Pickle: Complete Python object preservation")
```

### 2. For Business Users (Accessible)

```python
def export_for_business_users(df):
    """Export in business-friendly formats"""
    
    # Excel with formatting
    with pd.ExcelWriter('business_report.xlsx', engine='openpyxl') as writer:
        # Summary sheet
        summary = pd.DataFrame({
            'Metric': ['Total Customers', 'Average Income', 'Average Spending', 'Top City'],
            'Value': [
                len(df),
                f"${df['income'].mean():,.2f}",
                f"${df['spending'].mean():,.2f}",
                df['city'].mode()[0]
            ]
        })
        summary.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Detailed data
        df.to_excel(writer, sheet_name='Customer Data', index=False)
        
        # Charts data
        city_summary = df.groupby('city').agg({
            'customer_id': 'count',
            'income': 'mean',
            'spending': 'mean'
        }).reset_index()
        city_summary.to_excel(writer, sheet_name='City Analysis', index=False)
    
    # CSV for easy sharing
    df.to_csv('customer_data_clean.csv', index=False)
    
    print("Business exports complete:")
    print("- Excel report with multiple sheets")
    print("- Clean CSV for further analysis")

export_for_business_users(customer_data)
```

## 🎯 Key Takeaways

1. **Pandas reads everything**: CSV, Excel, JSON, databases, APIs
2. **Validation is crucial**: Always check data quality after import
3. **Optimization matters**: Choose appropriate data types and formats
4. **Export strategically**: Different formats for different audiences
5. **Handle errors gracefully**: Real-world data is messy

## 🚀 What's Next?

Now that you can get data in and out of pandas, it's time to tackle the biggest challenge in data science: **Data Cleaning** - turning messy, real-world data into analysis-ready datasets!
