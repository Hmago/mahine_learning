# Universal Functions: NumPy's Speed Demons

## 🤔 What Are Universal Functions (ufuncs)?

Think of ufuncs as **super-powered mathematical operations** that work on entire arrays at once. They're like having a team of mathematicians who can perform calculations on millions of numbers simultaneously.

**Regular Python**: Calculate square root one number at a time
**NumPy ufuncs**: Calculate square root of a million numbers in one go

## 🚀 Why Ufuncs Are Game Changers

### Speed Comparison: The Shocking Truth

```python
import numpy as np
import time
import math

# Create a large dataset
data = list(range(1000000))
np_data = np.array(data)

# Python way (slow)
start = time.time()
python_result = [math.sqrt(x) for x in data]
python_time = time.time() - start

# NumPy way (fast)
start = time.time()
numpy_result = np.sqrt(np_data)
numpy_time = time.time() - start

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")
```

**Typical result**: NumPy is 50-100x faster!

## 📚 Categories of Universal Functions

### 1. Mathematical Operations

```python
import numpy as np

# Sample business data
sales_revenue = np.array([10000, 15000, 8000, 22000, 18000])
costs = np.array([7000, 9000, 6000, 14000, 11000])

# Basic arithmetic (element-wise)
profit = sales_revenue - costs
profit_margin = profit / sales_revenue
growth_rate = np.power(sales_revenue / 10000, 2)  # Quadratic growth model

print(f"Profits: {profit}")
print(f"Profit margins: {profit_margin:.2%}")
```

### 2. Trigonometric Functions

```python
# Analyzing seasonal sales patterns
months = np.arange(1, 13)  # January to December
seasonal_factor = np.sin(2 * np.pi * months / 12)  # Seasonal wave

# Base sales with seasonal adjustment
base_sales = 50000
seasonal_sales = base_sales * (1 + 0.3 * seasonal_factor)

print("Monthly seasonal sales:")
for month, sales in zip(months, seasonal_sales):
    print(f"Month {month}: ${sales:.0f}")
```

### 3. Exponential and Logarithmic Functions

```python
# Financial modeling: compound interest and logarithmic growth
principal = 10000
rate = 0.05  # 5% annual interest
years = np.arange(1, 21)  # 1 to 20 years

# Compound interest calculation
compound_value = principal * np.power(1 + rate, years)

# Time to double investment (using logarithms)
doubling_time = np.log(2) / np.log(1 + rate)

print(f"Initial investment: ${principal}")
print(f"Value after 10 years: ${compound_value[9]:.2f}")
print(f"Time to double: {doubling_time:.1f} years")
```

### 4. Comparison Operations

```python
# Customer analysis: finding high-value customers
customer_spending = np.array([1200, 800, 2500, 450, 1800, 3200, 950])
premium_threshold = 1500

# Boolean array showing premium customers
is_premium = customer_spending >= premium_threshold
premium_customers = customer_spending[is_premium]

# Statistical analysis
above_average = customer_spending > np.mean(customer_spending)
top_percentile = customer_spending >= np.percentile(customer_spending, 90)

print(f"Premium customers (>${premium_threshold}): {premium_customers}")
print(f"Customers above average: {above_average.sum()}")
print(f"Top 10% customers: {customer_spending[top_percentile]}")
```

## 🎯 Real-World Business Applications

### 1. Risk Assessment in Finance

```python
# Portfolio risk analysis
stock_prices = np.array([100, 105, 98, 110, 95, 108, 102])
returns = (stock_prices[1:] - stock_prices[:-1]) / stock_prices[:-1]

# Risk metrics using ufuncs
volatility = np.std(returns)  # Standard deviation
max_loss = np.min(returns)    # Maximum daily loss
sharpe_ratio = np.mean(returns) / volatility  # Risk-adjusted return

# Value at Risk (95% confidence)
var_95 = np.percentile(returns, 5)

print(f"Daily volatility: {volatility:.3f}")
print(f"Maximum loss: {max_loss:.3f}")
print(f"Sharpe ratio: {sharpe_ratio:.3f}")
print(f"VaR (95%): {var_95:.3f}")
```

### 2. Customer Lifetime Value Calculation

```python
# CLV calculation for e-commerce
monthly_revenue = np.array([150, 200, 180, 220, 190])  # Per customer
churn_rate = 0.05  # 5% monthly churn
discount_rate = 0.01  # 1% monthly discount rate

# Calculate CLV using geometric series
months = np.arange(1, 37)  # 3-year projection
retention_rate = np.power(1 - churn_rate, months - 1)
discount_factor = np.power(1 / (1 + discount_rate), months - 1)

# Average monthly revenue projected
avg_monthly = np.mean(monthly_revenue)
clv = np.sum(avg_monthly * retention_rate * discount_factor)

print(f"Customer Lifetime Value: ${clv:.2f}")
```

### 3. A/B Test Analysis

```python
# A/B test results analysis
control_group = np.array([120, 135, 118, 142, 128, 139, 125, 133, 144, 127])
test_group = np.array([145, 152, 138, 159, 148, 155, 142, 150, 163, 147])

# Statistical analysis using ufuncs
control_mean = np.mean(control_group)
test_mean = np.mean(test_group)
control_std = np.std(control_group)
test_std = np.std(test_group)

# Effect size calculation
pooled_std = np.sqrt((control_std**2 + test_std**2) / 2)
cohens_d = (test_mean - control_mean) / pooled_std

improvement = (test_mean - control_mean) / control_mean

print(f"Control group average: {control_mean:.2f}")
print(f"Test group average: {test_mean:.2f}")
print(f"Improvement: {improvement:.2%}")
print(f"Effect size (Cohen's d): {cohens_d:.3f}")
```

## 🛠️ Creating Custom Universal Functions

### Simple Custom Function

```python
# Create a custom business metric function
def customer_score(recency, frequency, monetary):
    """Calculate customer score based on RFM analysis"""
    return 0.3 * recency + 0.3 * frequency + 0.4 * monetary

# Convert to ufunc for vectorized operation
customer_score_ufunc = np.frompyfunc(customer_score, 3, 1)

# Apply to customer data
recency_scores = np.array([5, 3, 8, 2, 9, 4, 7])
frequency_scores = np.array([8, 9, 4, 10, 3, 7, 6])
monetary_scores = np.array([7, 8, 5, 9, 4, 8, 6])

customer_scores = customer_score_ufunc(recency_scores, frequency_scores, monetary_scores)
print(f"Customer scores: {customer_scores.astype(float)}")
```

### Performance-Optimized Custom Function

```python
# High-performance custom function using numba (if available)
try:
    from numba import vectorize
    
    @vectorize(['float64(float64, float64)'])
    def optimized_roi(revenue, cost):
        return (revenue - cost) / cost
    
    # Use like any other ufunc
    revenues = np.random.uniform(1000, 5000, 10000)
    costs = np.random.uniform(500, 2000, 10000)
    
    roi_values = optimized_roi(revenues, costs)
    print(f"Average ROI: {np.mean(roi_values):.2%}")
    
except ImportError:
    print("Numba not available, using regular NumPy")
    roi_values = (revenues - costs) / costs
```

## 🎮 Advanced Ufunc Techniques

### 1. Reduction Operations

```python
# Sales data analysis across multiple dimensions
sales_data = np.random.rand(12, 5, 10) * 1000  # 12 months, 5 regions, 10 products

# Different types of reductions
total_sales = np.sum(sales_data)  # Grand total
monthly_totals = np.sum(sales_data, axis=(1, 2))  # Total per month
regional_totals = np.sum(sales_data, axis=(0, 2))  # Total per region
product_totals = np.sum(sales_data, axis=(0, 1))  # Total per product

# Statistical reductions
monthly_averages = np.mean(sales_data, axis=(1, 2))
regional_std = np.std(sales_data, axis=(0, 2))
product_max = np.max(sales_data, axis=(0, 1))

print("Sales Analysis:")
print(f"Total annual sales: ${total_sales:.2f}")
print(f"Best month: {np.argmax(monthly_totals) + 1} (${monthly_totals.max():.2f})")
print(f"Best region: {np.argmax(regional_totals)} (${regional_totals.max():.2f})")
```

### 2. Accumulation Operations

```python
# Running totals and cumulative analysis
daily_sales = np.random.poisson(100, 365)  # Daily sales for a year

# Cumulative operations
cumulative_sales = np.cumsum(daily_sales)
running_average = np.cumsum(daily_sales) / np.arange(1, 366)

# Find breakeven point (assuming fixed costs)
fixed_costs = 20000
daily_profit = daily_sales * 25 - 1000  # $25 profit per sale, $1000 daily costs
cumulative_profit = np.cumsum(daily_profit)
breakeven_day = np.argmax(cumulative_profit > fixed_costs)

print(f"Annual sales: {daily_sales.sum():,} units")
print(f"Breakeven day: {breakeven_day}")
print(f"Sales on day 100: {daily_sales[99]}")
print(f"Cumulative sales by day 100: {cumulative_sales[99]:,}")
print(f"Running average by day 100: {running_average[99]:.1f}")
```

### 3. Where Function: Conditional Operations

```python
# Customer segmentation with conditional logic
customer_data = np.random.rand(1000, 3)  # 1000 customers, 3 features
spending = customer_data[:, 0] * 10000  # Annual spending
frequency = customer_data[:, 1] * 100   # Purchase frequency
recency = customer_data[:, 2] * 365     # Days since last purchase

# Complex segmentation logic
premium = (spending > 5000) & (frequency > 50)
at_risk = (recency > 180) & (spending > 2000)
new_customers = (recency < 30) & (frequency < 10)

# Use where for conditional values
customer_segments = np.where(
    premium, 'Premium',
    np.where(at_risk, 'At Risk', 
             np.where(new_customers, 'New', 'Standard'))
)

# Segment analysis
unique_segments, counts = np.unique(customer_segments, return_counts=True)
for segment, count in zip(unique_segments, counts):
    avg_spending = np.mean(spending[customer_segments == segment])
    print(f"{segment}: {count} customers (avg spending: ${avg_spending:.2f})")
```

## 🚀 Performance Optimization Tips

### 1. Vectorization Over Loops

```python
# Bad: Using Python loops
def slow_calculation(data):
    result = []
    for value in data:
        result.append(value ** 2 + 2 * value + 1)
    return result

# Good: Using ufuncs
def fast_calculation(data):
    return data ** 2 + 2 * data + 1

# Test performance
large_data = np.random.rand(1000000)

# The ufunc version will be 50-100x faster!
fast_result = fast_calculation(large_data)
```

### 2. In-Place Operations

```python
# Memory-efficient operations
large_array = np.random.rand(10000, 1000)

# Creates new array (memory expensive)
result = large_array * 2 + 1

# In-place operations (memory efficient)
large_array *= 2
large_array += 1
# Now large_array is modified in place
```

### 3. Choosing the Right Data Type

```python
# Use appropriate data types for memory and speed
# For integers 0-255 (like image pixels)
image_data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)

# For financial calculations requiring precision
financial_data = np.random.rand(1000) * 1000000
financial_data = financial_data.astype(np.float64)

# For boolean masks
is_profitable = np.random.choice([True, False], 1000)  # Already boolean
```

## 🎯 Key Takeaways

1. **Ufuncs eliminate loops**: They operate on entire arrays at once
2. **Massive performance gains**: 10-100x faster than pure Python
3. **Broadcasting compatible**: Work seamlessly with arrays of different shapes
4. **Memory efficient**: Optimized C implementations under the hood
5. **Comprehensive library**: Functions for almost every mathematical operation

## 🚀 What's Next?

Now that you've mastered the speed of ufuncs, it's time to explore **Linear Algebra** - the mathematical foundation that powers machine learning algorithms!
