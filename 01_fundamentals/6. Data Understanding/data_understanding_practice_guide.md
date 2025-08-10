# Data Understanding: Practical Guide 🛠️

*Hands-on implementation of data understanding concepts*

## Quick Reference Implementation

### Essential Data Understanding Checklist

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def comprehensive_data_overview(df):
    """
    Comprehensive data understanding function
    """
    print("=" * 50)
    print("COMPREHENSIVE DATA OVERVIEW")
    print("=" * 50)
    
    # Basic Info
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data Types
    print(f"\n📋 Data Types Overview:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Missing Values
    print(f"\n❌ Missing Values Summary:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0].head(10))
    
    # Numerical Columns Summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\n🔢 Numerical Columns ({len(numerical_cols)}):")
        print(df[numerical_cols].describe())
    
    # Categorical Columns Summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
        for col in categorical_cols[:5]:  # Show first 5
            unique_count = df[col].nunique()
            print(f"   {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"      Values: {df[col].value_counts().head().to_dict()}")
    
    return missing_df, numerical_cols, categorical_cols
```

## Practical Implementation Examples

### 1. Data Loading and Initial Assessment

```python
# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your data

# Quick initial assessment
missing_info, num_cols, cat_cols = comprehensive_data_overview(df)
```

### 2. Missing Value Analysis

```python
def analyze_missing_patterns(df):
    """
    Analyze missing value patterns
    """
    # Missing value heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Missing value bar plot
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        plt.figure(figsize=(10, 6))
        missing_data.plot(kind='bar')
        plt.title('Missing Values Count by Column')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage
analyze_missing_patterns(df)
```

### 3. Numerical Data Analysis

```python
def analyze_numerical_data(df, columns=None):
    """
    Comprehensive numerical data analysis
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df[col].dtype in [np.number]:
            print(f"\n" + "="*50)
            print(f"Analysis for: {col}")
            print("="*50)
            
            # Basic statistics
            print(f"Mean: {df[col].mean():.2f}")
            print(f"Median: {df[col].median():.2f}")
            print(f"Std: {df[col].std():.2f}")
            print(f"Skewness: {df[col].skew():.2f}")
            print(f"Kurtosis: {df[col].kurtosis():.2f}")
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histogram
            axes[0].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[0].set_title(f'{col} - Distribution')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            
            # Box plot
            axes[1].boxplot(df[col].dropna())
            axes[1].set_title(f'{col} - Box Plot')
            axes[1].set_ylabel(col)
            
            # Q-Q plot
            stats.probplot(df[col].dropna(), dist="norm", plot=axes[2])
            axes[2].set_title(f'{col} - Q-Q Plot')
            
            plt.tight_layout()
            plt.show()

# Usage
analyze_numerical_data(df)
```

### 4. Categorical Data Analysis

```python
def analyze_categorical_data(df, columns=None, max_categories=20):
    """
    Comprehensive categorical data analysis
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in columns:
        print(f"\n" + "="*50)
        print(f"Analysis for: {col}")
        print("="*50)
        
        # Basic statistics
        unique_count = df[col].nunique()
        print(f"Unique values: {unique_count}")
        print(f"Missing values: {df[col].isnull().sum()}")
        
        # Value counts
        value_counts = df[col].value_counts()
        print(f"\nTop 10 most frequent values:")
        print(value_counts.head(10))
        
        # Visualization
        if unique_count <= max_categories:
            plt.figure(figsize=(12, 6))
            
            # Bar plot
            plt.subplot(1, 2, 1)
            value_counts.head(15).plot(kind='bar')
            plt.title(f'{col} - Frequency Distribution')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Pie chart for top categories
            plt.subplot(1, 2, 2)
            top_categories = value_counts.head(8)
            others = value_counts.iloc[8:].sum()
            if others > 0:
                top_categories['Others'] = others
            
            plt.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
            plt.title(f'{col} - Proportion')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Too many unique values ({unique_count}) for visualization")

# Usage
analyze_categorical_data(df)
```

### 5. Correlation Analysis

```python
def correlation_analysis(df):
    """
    Comprehensive correlation analysis
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Visualization
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold for high correlation
                high_corr_pairs.append((corr_matrix.columns[i], 
                                      corr_matrix.columns[j], 
                                      corr_val))
    
    if high_corr_pairs:
        print("\nHigh Correlation Pairs (|r| > 0.7):")
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} - {var2}: {corr:.3f}")
    
    return corr_matrix

# Usage
corr_matrix = correlation_analysis(df)
```

### 6. Bivariate Analysis

```python
def bivariate_analysis(df, target_column=None):
    """
    Bivariate analysis with target variable
    """
    if target_column is None or target_column not in df.columns:
        print("Please specify a valid target column")
        return
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from lists if it's there
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    # Numerical vs Target
    if df[target_column].dtype in [np.number]:
        print("Numerical Target Analysis")
        print("=" * 30)
        
        # Numerical features vs numerical target
        for col in numerical_cols[:6]:  # Limit to first 6
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(df[col], df[target_column], alpha=0.6)
            plt.xlabel(col)
            plt.ylabel(target_column)
            plt.title(f'{col} vs {target_column}')
            
            # Calculate correlation
            corr = df[col].corr(df[target_column])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.subplot(1, 2, 2)
            df.groupby(pd.cut(df[col], bins=10))[target_column].mean().plot(kind='bar')
            plt.title(f'Average {target_column} by {col} bins')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    else:
        print("Categorical Target Analysis")
        print("=" * 30)
        
        # Numerical features vs categorical target
        for col in numerical_cols[:6]:  # Limit to first 6
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            for category in df[target_column].unique():
                subset = df[df[target_column] == category]
                plt.hist(subset[col].dropna(), alpha=0.6, label=str(category), bins=20)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(f'{col} distribution by {target_column}')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            df.boxplot(column=col, by=target_column, ax=plt.gca())
            plt.title(f'{col} by {target_column}')
            
            plt.tight_layout()
            plt.show()

# Usage - replace 'target_column_name' with your actual target column
bivariate_analysis(df, target_column='your_target_column')
```

### 7. Data Quality Report

```python
def generate_data_quality_report(df):
    """
    Generate comprehensive data quality report
    """
    report = {}
    
    # Basic info
    report['basic_info'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Data types
    report['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    report['missing_values'] = {
        'columns_with_missing': (missing > 0).sum(),
        'total_missing_values': missing.sum(),
        'missing_percentage': (missing.sum() / (df.shape[0] * df.shape[1])) * 100
    }
    
    # Duplicates
    duplicates = df.duplicated().sum()
    report['duplicates'] = {
        'duplicate_rows': duplicates,
        'duplicate_percentage': (duplicates / df.shape[0]) * 100
    }
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        report['numerical_analysis'] = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            report['numerical_analysis'][col] = {
                'outliers': outliers,
                'outlier_percentage': (outliers / df.shape[0]) * 100,
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        report['categorical_analysis'] = {}
        for col in categorical_cols:
            report['categorical_analysis'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            }
    
    # Print report
    print("DATA QUALITY REPORT")
    print("=" * 50)
    
    print(f"\n📊 Basic Information:")
    print(f"   Rows: {report['basic_info']['rows']:,}")
    print(f"   Columns: {report['basic_info']['columns']}")
    print(f"   Memory Usage: {report['basic_info']['memory_usage_mb']:.2f} MB")
    
    print(f"\n📋 Data Types:")
    for dtype, count in report['data_types'].items():
        print(f"   {dtype}: {count}")
    
    print(f"\n❌ Missing Values:")
    print(f"   Columns with missing: {report['missing_values']['columns_with_missing']}")
    print(f"   Total missing: {report['missing_values']['total_missing_values']:,}")
    print(f"   Missing percentage: {report['missing_values']['missing_percentage']:.2f}%")
    
    print(f"\n🔄 Duplicates:")
    print(f"   Duplicate rows: {report['duplicates']['duplicate_rows']:,}")
    print(f"   Duplicate percentage: {report['duplicates']['duplicate_percentage']:.2f}%")
    
    if 'numerical_analysis' in report:
        print(f"\n🔢 Numerical Columns Issues:")
        for col, analysis in report['numerical_analysis'].items():
            if analysis['outliers'] > 0:
                print(f"   {col}: {analysis['outliers']} outliers ({analysis['outlier_percentage']:.1f}%)")
    
    return report

# Usage
quality_report = generate_data_quality_report(df)
```

## Implementation Workflow

### Complete Data Understanding Pipeline

```python
def complete_data_understanding_pipeline(df, target_column=None):
    """
    Complete data understanding pipeline
    """
    print("🚀 Starting Comprehensive Data Understanding Pipeline")
    print("=" * 60)
    
    # Step 1: Basic overview
    print("\n📊 STEP 1: Basic Data Overview")
    missing_info, num_cols, cat_cols = comprehensive_data_overview(df)
    
    # Step 2: Data quality report
    print("\n📋 STEP 2: Data Quality Assessment")
    quality_report = generate_data_quality_report(df)
    
    # Step 3: Missing value analysis
    print("\n❌ STEP 3: Missing Value Analysis")
    analyze_missing_patterns(df)
    
    # Step 4: Numerical analysis
    if len(num_cols) > 0:
        print("\n🔢 STEP 4: Numerical Data Analysis")
        analyze_numerical_data(df, num_cols[:3])  # Limit to first 3
    
    # Step 5: Categorical analysis
    if len(cat_cols) > 0:
        print("\n📝 STEP 5: Categorical Data Analysis")
        analyze_categorical_data(df, cat_cols[:3])  # Limit to first 3
    
    # Step 6: Correlation analysis
    if len(num_cols) > 1:
        print("\n🔗 STEP 6: Correlation Analysis")
        corr_matrix = correlation_analysis(df)
    
    # Step 7: Bivariate analysis
    if target_column:
        print("\n🎯 STEP 7: Bivariate Analysis with Target")
        bivariate_analysis(df, target_column)
    
    print("\n✅ Data Understanding Pipeline Complete!")
    
    return {
        'missing_info': missing_info,
        'numerical_columns': num_cols,
        'categorical_columns': cat_cols,
        'quality_report': quality_report
    }

# Usage example
# results = complete_data_understanding_pipeline(df, target_column='your_target')
```

## Quick Start Example

```python
# Quick start for any dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your data
df = pd.read_csv('your_dataset.csv')  # Replace with your data source

# 2. Run complete pipeline
results = complete_data_understanding_pipeline(df, target_column='target')  # Replace 'target'

# 3. Additional custom analysis based on findings
# Add your specific analysis here based on what you discovered
```

## Practice Exercises

### Exercise 1: Real Estate Data
- Load a real estate dataset
- Identify data quality issues
- Analyze price distributions
- Find correlations with property features

### Exercise 2: Customer Data
- Analyze customer demographics
- Identify missing value patterns
- Segment customers by behavior
- Create customer profiles

### Exercise 3: Time Series Data
- Load time series data (sales, stock prices, etc.)
- Analyze temporal patterns
- Identify seasonality
- Detect anomalies

### Exercise 4: Text Data
- Load dataset with text reviews
- Analyze text length distributions
- Find common words/phrases
- Sentiment distribution analysis

---

*Remember: Good data understanding is the foundation of successful machine learning projects. Take time to really understand your data before building models!*
