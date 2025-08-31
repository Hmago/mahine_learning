# Time Series Mastery: Working with Time-Based Data

## 🤔 What is Time Series Data?

Imagine you're tracking your heartbeat, stock prices, or website visitors over time. **Time series data is any data that has a timestamp** - it shows how things change over time.

Time series data is **everywhere in business**:
- Sales revenue by month
- Website traffic by hour
- Customer signups by day
- Stock prices by minute
- Temperature readings by second

## 🎯 Why Time Series Matters in Business

**Real-world time series applications:**

- **Sales Forecasting**: "How much will we sell next quarter?"
- **Demand Planning**: "How much inventory do we need?"
- **Website Optimization**: "When do users visit our site most?"
- **Financial Analysis**: "What are our seasonal revenue patterns?"
- **Operational Efficiency**: "When do we need the most staff?"

Think of time series analysis as **understanding the rhythm of your business** - the patterns, cycles, and trends that repeat over time.

## 📅 Time Series Fundamentals

### 1. **Working with Dates and Times**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

print("📅 TIME SERIES FUNDAMENTALS")
print("=" * 32)

# Creating time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
print("Date range created:")
print(f"Start: {dates[0]}")
print(f"End: {dates[-1]}")
print(f"Frequency: Daily")

# Different frequency options
print("\n⏰ Common Time Frequencies:")
frequencies = {
    'Daily': pd.date_range('2023-01-01', periods=7, freq='D'),
    'Weekly': pd.date_range('2023-01-01', periods=4, freq='W'),
    'Monthly': pd.date_range('2023-01-01', periods=12, freq='M'),
    'Quarterly': pd.date_range('2023-01-01', periods=4, freq='Q'),
    'Hourly': pd.date_range('2023-01-01', periods=24, freq='H')
}

for freq_name, freq_dates in frequencies.items():
    print(f"{freq_name}: {freq_dates[0]} to {freq_dates[-1]}")
```

### 2. **Creating Realistic Business Time Series**

```python
# Generate realistic business data with seasonal patterns
def create_business_time_series():
    """Create realistic e-commerce sales data with trends and seasonality"""
    
    # Create date range for 2 years of daily data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # Base trend (business growing over time)
    trend = np.linspace(1000, 1500, len(dates))
    
    # Seasonal patterns
    seasonal_yearly = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Annual cycle
    seasonal_weekly = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)      # Weekly cycle
    
    # Random noise
    noise = np.random.normal(0, 100, len(dates))
    
    # Special events (Black Friday, Christmas, etc.)
    sales = trend + seasonal_yearly + seasonal_weekly + noise
    
    # Add Black Friday spikes (last Friday of November)
    for year in [2022, 2023]:
        black_friday = pd.Timestamp(f'{year}-11-01') + pd.DateOffset(days=24-pd.Timestamp(f'{year}-11-01').weekday())
        if black_friday in dates:
            bf_index = dates.get_loc(black_friday)
            sales[bf_index] += 1000  # Big spike on Black Friday
    
    # Create DataFrame
    sales_data = pd.DataFrame({
        'date': dates,
        'daily_sales': np.maximum(sales, 0),  # Ensure no negative sales
        'day_of_week': dates.day_name(),
        'month': dates.month,
        'quarter': dates.quarter,
        'year': dates.year
    })
    
    return sales_data

# Generate the business data
business_data = create_business_time_series()

print("\n💼 BUSINESS TIME SERIES DATA")
print("=" * 35)
print("Generated 2 years of daily sales data with:")
print("✅ Growth trend")
print("✅ Annual seasonality") 
print("✅ Weekly patterns")
print("✅ Special events (Black Friday)")
print("✅ Random variation")

print(f"\nData shape: {business_data.shape}")
print("Sample data:")
print(business_data.head())

# Quick analysis
print(f"\nBasic Statistics:")
print(f"Average daily sales: ${business_data['daily_sales'].mean():,.2f}")
print(f"Highest sales day: ${business_data['daily_sales'].max():,.2f}")
print(f"Lowest sales day: ${business_data['daily_sales'].min():,.2f}")
```

### 3. **Time Series Indexing and Selection**

```python
print("\n🔍 TIME SERIES INDEXING")
print("=" * 28)

# Set date as index for time series operations
ts_data = business_data.set_index('date')

print("1. Selecting specific dates:")
print("Sales on January 1, 2023:")
jan_1_sales = ts_data.loc['2023-01-01', 'daily_sales']
print(f"${jan_1_sales:,.2f}")

print("\n2. Selecting date ranges:")
print("Sales in January 2023:")
jan_2023 = ts_data.loc['2023-01':'2023-01']
print(f"Total: ${jan_2023['daily_sales'].sum():,.2f}")
print(f"Average: ${jan_2023['daily_sales'].mean():,.2f}")
print(f"Days: {len(jan_2023)}")

print("\n3. Selecting by year:")
sales_2023 = ts_data.loc['2023']
print(f"2023 total sales: ${sales_2023['daily_sales'].sum():,.2f}")

print("\n4. Advanced date filtering:")
# Get all Mondays
mondays = ts_data[ts_data['day_of_week'] == 'Monday']
print(f"Average Monday sales: ${mondays['daily_sales'].mean():,.2f}")

# Get Q4 data for both years
q4_data = ts_data[ts_data['quarter'] == 4]
print(f"Q4 average daily sales: ${q4_data['daily_sales'].mean():,.2f}")

# Get holiday season (November-December)
holiday_season = ts_data[ts_data['month'].isin([11, 12])]
print(f"Holiday season daily average: ${holiday_season['daily_sales'].mean():,.2f}")
```

### 4. **Time Series Resampling: Changing Frequency**

```python
print("\n📊 TIME SERIES RESAMPLING")
print("=" * 31)

# Resampling is like changing the zoom level on your data
# Daily -> Weekly -> Monthly -> Quarterly

print("1. Daily to Weekly Resampling:")
weekly_sales = ts_data['daily_sales'].resample('W').sum()
print("Weekly sales (last 8 weeks):")
print(weekly_sales.tail(8).round(2))

print("\n2. Daily to Monthly Resampling:")
monthly_sales = ts_data['daily_sales'].resample('M').sum()
print("Monthly sales:")
print(monthly_sales.round(2))

print("\n3. Multiple Aggregations:")
monthly_stats = ts_data['daily_sales'].resample('M').agg({
    'total_sales': 'sum',
    'avg_daily_sales': 'mean',
    'max_daily_sales': 'max',
    'min_daily_sales': 'min',
    'trading_days': 'count'
}).round(2)

print("Monthly Statistics:")
print(monthly_stats)

print("\n4. Business Calendar Resampling:")
# Resample to business quarters (more relevant than calendar quarters)
quarterly_sales = ts_data['daily_sales'].resample('Q').sum()
print("Quarterly sales:")
print(quarterly_sales.round(2))

# Year-over-Year growth
yearly_sales = ts_data['daily_sales'].resample('Y').sum()
yoy_growth = yearly_sales.pct_change() * 100
print(f"\nYear-over-Year Growth: {yoy_growth.iloc[-1]:.1f}%")
```

### 5. **Rolling Window Analysis: Moving Averages and Trends**

```python
print("\n📈 ROLLING WINDOW ANALYSIS")
print("=" * 33)

# Rolling windows help smooth out noise and identify trends
# Think of it as looking at your data through a moving window

# Calculate different moving averages
ts_data['7_day_avg'] = ts_data['daily_sales'].rolling(window=7).mean()
ts_data['30_day_avg'] = ts_data['daily_sales'].rolling(window=30).mean()
ts_data['90_day_avg'] = ts_data['daily_sales'].rolling(window=90).mean()

print("1. Moving Averages (Latest values):")
latest_data = ts_data[['daily_sales', '7_day_avg', '30_day_avg', '90_day_avg']].tail(5)
print(latest_data.round(2))

print("\n2. Rolling Statistics for Risk Analysis:")
# Calculate rolling volatility (standard deviation)
ts_data['7_day_volatility'] = ts_data['daily_sales'].rolling(window=7).std()
ts_data['30_day_volatility'] = ts_data['daily_sales'].rolling(window=30).std()

print("Sales Volatility Analysis:")
volatility_stats = ts_data[['7_day_volatility', '30_day_volatility']].describe()
print(volatility_stats.round(2))

print("\n3. Rolling Correlations (Advanced):")
# Create another time series for correlation analysis
ts_data['marketing_spend'] = (
    500 + 200 * np.sin(2 * np.pi * np.arange(len(ts_data)) / 30) + 
    np.random.normal(0, 50, len(ts_data))
)

# Rolling correlation between sales and marketing spend
ts_data['sales_marketing_corr'] = (
    ts_data['daily_sales']
    .rolling(window=30)
    .corr(ts_data['marketing_spend'].rolling(window=30))
)

latest_correlation = ts_data['sales_marketing_corr'].dropna().tail(1).iloc[0]
print(f"Latest 30-day correlation (Sales vs Marketing): {latest_correlation:.3f}")

print("\n4. Business Intelligence: Trend Detection")
# Detect if sales are trending up or down
ts_data['trend_7day'] = ts_data['daily_sales'] / ts_data['7_day_avg'] - 1
ts_data['trend_30day'] = ts_data['daily_sales'] / ts_data['30_day_avg'] - 1

# Current trend analysis
current_trend_7d = ts_data['trend_7day'].tail(7).mean()
current_trend_30d = ts_data['trend_30day'].tail(30).mean()

print(f"Current 7-day trend: {current_trend_7d:.2%}")
print(f"Current 30-day trend: {current_trend_30d:.2%}")

if current_trend_7d > 0.05:
    print("🔥 Strong upward trend detected!")
elif current_trend_7d < -0.05:
    print("📉 Downward trend detected - investigate!")
else:
    print("➡️ Sales trending sideways")
```

### 6. **Seasonal Analysis: Understanding Business Cycles**

```python
print("\n🔄 SEASONAL ANALYSIS")
print("=" * 25)

# Understanding seasonal patterns is crucial for business planning

print("1. Day-of-Week Patterns:")
dow_analysis = ts_data.groupby('day_of_week')['daily_sales'].agg(['mean', 'std']).round(2)
# Reorder by weekday
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_analysis = dow_analysis.reindex(day_order)
print(dow_analysis)

# Identify best and worst days
best_day = dow_analysis['mean'].idxmax()
worst_day = dow_analysis['mean'].idxmin()
print(f"\nBest sales day: {best_day} (${dow_analysis.loc[best_day, 'mean']:,.2f})")
print(f"Worst sales day: {worst_day} (${dow_analysis.loc[worst_day, 'mean']:,.2f})")

print("\n2. Monthly Seasonality:")
monthly_pattern = ts_data.groupby('month')['daily_sales'].mean().round(2)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_pattern.index = month_names
print("Average daily sales by month:")
print(monthly_pattern)

# Seasonal strength calculation
seasonal_strength = (monthly_pattern.max() - monthly_pattern.min()) / monthly_pattern.mean()
print(f"\nSeasonal strength: {seasonal_strength:.2f}")
print("(Higher values = more seasonal)")

print("\n3. Holiday Impact Analysis:")
# Identify top sales days (potential holidays/events)
top_sales_days = ts_data.nlargest(10, 'daily_sales')[['daily_sales', 'day_of_week', 'month']]
print("Top 10 sales days:")
print(top_sales_days)

print("\n4. Quarterly Business Performance:")
quarterly_performance = ts_data.groupby(['year', 'quarter'])['daily_sales'].agg([
    'sum', 'mean', 'count'
]).round(2)
quarterly_performance.columns = ['total_sales', 'avg_daily_sales', 'trading_days']

# Calculate quarter-over-quarter growth
quarterly_performance['qoq_growth'] = (
    quarterly_performance['total_sales'].pct_change() * 100
).round(2)

print("Quarterly Performance Analysis:")
print(quarterly_performance)
```

### 7. **Advanced Time Series Operations**

```python
print("\n⚙️ ADVANCED TIME SERIES OPERATIONS")
print("=" * 42)

# Time zone handling (important for global businesses)
print("1. Time Zone Operations:")
# Convert to different time zones
ts_data_utc = ts_data.copy()
ts_data_utc.index = ts_data_utc.index.tz_localize('UTC')
ts_data_est = ts_data_utc.tz_convert('US/Eastern')
ts_data_pst = ts_data_utc.tz_convert('US/Pacific')

print(f"UTC time: {ts_data_utc.index[0]}")
print(f"EST time: {ts_data_est.index[0]}")
print(f"PST time: {ts_data_pst.index[0]}")

print("\n2. Business Day Operations:")
# Working with business calendars
business_days = pd.bdate_range('2023-01-01', '2023-12-31')
print(f"Business days in 2023: {len(business_days)}")

# Filter to business days only
business_only_data = ts_data.loc[ts_data.index.dayofweek < 5]  # Monday=0, Sunday=6
weekend_data = ts_data.loc[ts_data.index.dayofweek >= 5]

print(f"Weekday average sales: ${business_only_data['daily_sales'].mean():,.2f}")
print(f"Weekend average sales: ${weekend_data['daily_sales'].mean():,.2f}")

print("\n3. Gap Analysis and Missing Data:")
# Find gaps in time series
expected_dates = pd.date_range(ts_data.index.min(), ts_data.index.max(), freq='D')
missing_dates = expected_dates.difference(ts_data.index)

if len(missing_dates) > 0:
    print(f"Missing dates found: {len(missing_dates)}")
    print("First few missing dates:", missing_dates[:5])
else:
    print("✅ No gaps in time series data")

print("\n4. Forward Fill and Interpolation:")
# Create some missing data for demonstration
demo_data = ts_data['daily_sales'].copy()
demo_data.iloc[100:103] = np.nan  # Create gaps

print("Handling missing values:")
print(f"Original missing values: {demo_data.isnull().sum()}")

# Different filling strategies
forward_filled = demo_data.fillna(method='ffill')  # Forward fill
backward_filled = demo_data.fillna(method='bfill')  # Backward fill
interpolated = demo_data.interpolate()  # Linear interpolation

print(f"After forward fill: {forward_filled.isnull().sum()}")
print(f"After interpolation: {interpolated.isnull().sum()}")
```

### 8. **Business Forecasting Foundations**

```python
print("\n🔮 FORECASTING FOUNDATIONS")
print("=" * 32)

# Simple forecasting methods using historical patterns

print("1. Naive Forecasting:")
# Simplest forecast: tomorrow = today
last_value = ts_data['daily_sales'].iloc[-1]
print(f"Naive forecast (next day): ${last_value:,.2f}")

print("\n2. Seasonal Naive:")
# Use same day from previous week/year
last_week_same_day = ts_data['daily_sales'].iloc[-7]
last_year_same_day = ts_data['daily_sales'].iloc[-365]
print(f"Weekly seasonal forecast: ${last_week_same_day:,.2f}")
print(f"Yearly seasonal forecast: ${last_year_same_day:,.2f}")

print("\n3. Moving Average Forecast:")
# Use average of recent values
ma_7_forecast = ts_data['daily_sales'].tail(7).mean()
ma_30_forecast = ts_data['daily_sales'].tail(30).mean()
print(f"7-day MA forecast: ${ma_7_forecast:,.2f}")
print(f"30-day MA forecast: ${ma_30_forecast:,.2f}")

print("\n4. Trend-Based Forecast:")
# Simple linear trend
recent_data = ts_data['daily_sales'].tail(90)  # Last 90 days
trend_slope = (recent_data.iloc[-1] - recent_data.iloc[0]) / 90
trend_forecast = ts_data['daily_sales'].iloc[-1] + trend_slope
print(f"Linear trend forecast: ${trend_forecast:,.2f}")

print("\n5. Confidence Intervals:")
# Calculate prediction uncertainty
recent_volatility = ts_data['daily_sales'].tail(30).std()
confidence_80 = 1.28 * recent_volatility  # 80% confidence interval

forecast_value = ma_30_forecast
lower_bound = forecast_value - confidence_80
upper_bound = forecast_value + confidence_80

print(f"30-day MA forecast: ${forecast_value:,.2f}")
print(f"80% Confidence interval: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
```

### 9. **Time Series Visualization for Business Insights**

```python
print("\n📊 TIME SERIES VISUALIZATION")
print("=" * 35)

# Create comprehensive business dashboard plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Business Time Series Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Daily sales with moving averages
axes[0,0].plot(ts_data.index, ts_data['daily_sales'], alpha=0.3, label='Daily Sales', color='gray')
axes[0,0].plot(ts_data.index, ts_data['7_day_avg'], label='7-day Average', color='blue', linewidth=2)
axes[0,0].plot(ts_data.index, ts_data['30_day_avg'], label='30-day Average', color='red', linewidth=2)
axes[0,0].set_title('Daily Sales with Moving Averages')
axes[0,0].set_ylabel('Sales ($)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Monthly sales comparison
monthly_sales.plot(kind='bar', ax=axes[0,1], color='steelblue')
axes[0,1].set_title('Monthly Sales Comparison')
axes[0,1].set_ylabel('Total Sales ($)')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Day of week pattern
dow_analysis['mean'].plot(kind='bar', ax=axes[1,0], color='green', alpha=0.7)
axes[1,0].set_title('Average Sales by Day of Week')
axes[1,0].set_ylabel('Average Daily Sales ($)')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Quarterly trends
quarterly_totals = ts_data.groupby([ts_data.index.year, ts_data.index.quarter])['daily_sales'].sum()
quarterly_totals.plot(ax=axes[1,1], marker='o', linewidth=2, color='purple')
axes[1,1].set_title('Quarterly Sales Trends')
axes[1,1].set_ylabel('Quarterly Sales ($)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Dashboard created with 4 key business views:")
print("✅ Daily trends with moving averages")
print("✅ Monthly performance comparison")
print("✅ Weekly seasonality patterns")
print("✅ Quarterly growth trends")
```

## 🎯 Business Time Series Patterns

### 1. **E-commerce Patterns**

```python
def analyze_ecommerce_patterns(data):
    """Analyze typical e-commerce time series patterns"""
    
    print("\n🛒 E-COMMERCE PATTERN ANALYSIS")
    print("=" * 38)
    
    # Weekend vs Weekday patterns
    weekday_sales = data[data.index.dayofweek < 5]['daily_sales'].mean()
    weekend_sales = data[data.index.dayofweek >= 5]['daily_sales'].mean()
    
    print(f"Weekday average: ${weekday_sales:,.2f}")
    print(f"Weekend average: ${weekend_sales:,.2f}")
    print(f"Weekend lift: {(weekend_sales/weekday_sales - 1)*100:+.1f}%")
    
    # Hour-of-day patterns (if we had hourly data)
    print("\n⏰ Typical E-commerce Hourly Patterns:")
    print("Peak hours: 12-2 PM, 7-9 PM")
    print("Low hours: 2-6 AM")
    print("Mobile traffic peaks: Evening hours")
    
    # Holiday season analysis
    holiday_months = data[data.index.month.isin([11, 12])]
    regular_months = data[~data.index.month.isin([11, 12])]
    
    holiday_avg = holiday_months['daily_sales'].mean()
    regular_avg = regular_months['daily_sales'].mean()
    
    print(f"\nHoliday season (Nov-Dec) average: ${holiday_avg:,.2f}")
    print(f"Regular season average: ${regular_avg:,.2f}")
    print(f"Holiday lift: {(holiday_avg/regular_avg - 1)*100:+.1f}%")
    
    return {
        'weekday_avg': weekday_sales,
        'weekend_avg': weekend_sales,
        'holiday_avg': holiday_avg,
        'regular_avg': regular_avg
    }

ecommerce_insights = analyze_ecommerce_patterns(ts_data)
```

### 2. **Financial Time Series Patterns**

```python
def financial_time_series_analysis():
    """Demonstrate financial time series concepts"""
    
    print("\n💰 FINANCIAL TIME SERIES CONCEPTS")
    print("=" * 40)
    
    # Simulate stock-like data with volatility clustering
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(ts_data))  # Daily returns
    
    # Add volatility clustering (common in financial data)
    volatility = np.ones(len(returns))
    for i in range(1, len(returns)):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
        returns[i] = returns[i] * volatility[i]
    
    # Convert returns to price series
    price_series = pd.Series(100 * np.exp(returns.cumsum()), index=ts_data.index)
    
    financial_data = pd.DataFrame({
        'price': price_series,
        'returns': returns,
        'volatility': volatility
    })
    
    print("Key Financial Time Series Metrics:")
    print(f"Annualized Return: {returns.mean() * 252 * 100:.2f}%")
    print(f"Annualized Volatility: {returns.std() * np.sqrt(252) * 100:.2f}%")
    print(f"Sharpe Ratio: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
    
    # Rolling metrics common in finance
    financial_data['rolling_vol_20d'] = financial_data['returns'].rolling(20).std() * np.sqrt(252)
    financial_data['rolling_return_20d'] = financial_data['returns'].rolling(20).mean() * 252
    
    print(f"\nCurrent 20-day Rolling Volatility: {financial_data['rolling_vol_20d'].iloc[-1]*100:.1f}%")
    print(f"Current 20-day Rolling Return: {financial_data['rolling_return_20d'].iloc[-1]*100:.1f}%")
    
    return financial_data

financial_insights = financial_time_series_analysis()
```

## 🧪 Advanced Time Series Techniques

### 1. **Outlier Detection in Time Series**

```python
def detect_time_series_outliers(data, column='daily_sales', window=30, threshold=2.5):
    """Detect outliers in time series data using rolling statistics"""
    
    print("\n🔍 TIME SERIES OUTLIER DETECTION")
    print("=" * 40)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    
    # Calculate z-scores
    z_scores = (data[column] - rolling_mean) / rolling_std
    
    # Identify outliers
    outliers = data[abs(z_scores) > threshold].copy()
    outliers['z_score'] = z_scores[abs(z_scores) > threshold]
    
    print(f"Outliers detected (Z-score > {threshold}): {len(outliers)}")
    
    if len(outliers) > 0:
        print("\nTop 5 Outliers:")
        top_outliers = outliers.nlargest(5, 'z_score')[['daily_sales', 'z_score', 'day_of_week']]
        print(top_outliers.round(2))
        
        # Analyze outlier patterns
        outlier_dow = outliers['day_of_week'].value_counts()
        print(f"\nOutlier day-of-week distribution:")
        print(outlier_dow)
        
        # Seasonal outlier analysis
        outlier_months = outliers.groupby(outliers.index.month)['daily_sales'].count()
        if len(outlier_months) > 0:
            print(f"\nOutliers by month:")
            print(outlier_months)
    
    return outliers

outliers = detect_time_series_outliers(ts_data)
```

### 2. **Changepoint Detection**

```python
def detect_changepoints(data, column='daily_sales', min_size=30):
    """Simple changepoint detection using rolling statistics"""
    
    print("\n📊 CHANGEPOINT DETECTION")
    print("=" * 30)
    
    # Calculate rolling means with different window sizes
    short_window = data[column].rolling(window=7).mean()
    long_window = data[column].rolling(window=30).mean()
    
    # Signal when short-term average significantly differs from long-term
    signal = (short_window - long_window) / long_window
    
    # Identify potential changepoints
    threshold = 0.1  # 10% difference
    changepoints = data[abs(signal) > threshold].copy()
    changepoints['signal_strength'] = signal[abs(signal) > threshold]
    
    print(f"Potential changepoints detected: {len(changepoints)}")
    
    if len(changepoints) > 0:
        print("\nStrong signals (Top 5):")
        strong_signals = changepoints.nlargest(5, abs(changepoints['signal_strength']))
        print(strong_signals[['daily_sales', 'signal_strength']].round(3))
        
        # Business interpretation
        upward_changes = changepoints[changepoints['signal_strength'] > 0]
        downward_changes = changepoints[changepoints['signal_strength'] < 0]
        
        print(f"\nUpward trend changes: {len(upward_changes)}")
        print(f"Downward trend changes: {len(downward_changes)}")
    
    return changepoints, signal

changepoints, change_signal = detect_changepoints(ts_data)
```

## 🎮 Time Series Practice Challenges

### Challenge 1: Sales Performance Analysis

```python
def sales_performance_challenge():
    """Analyze sales performance with time series techniques"""
    
    print("\n🎯 SALES PERFORMANCE CHALLENGE")
    print("=" * 40)
    
    # Your mission: Analyze the business data and answer these questions:
    # 1. Which quarter showed the strongest growth?
    # 2. What's the optimal day of week for promotions?
    # 3. Predict next month's total sales
    # 4. Identify the most volatile sales periods
    # 5. Calculate seasonal adjustment factors
    
    # Starter analysis:
    quarterly_growth = ts_data.groupby(['year', 'quarter'])['daily_sales'].sum().pct_change()
    print("Quarterly Growth Rates:")
    print(quarterly_growth.dropna().round(3))
    
    # TODO: Complete the analysis
    print("\n📝 Your task: Complete the remaining analysis")
    
    return quarterly_growth

# Try the challenge!
# performance_results = sales_performance_challenge()
```

### Challenge 2: Customer Behavior Time Series

```python
def customer_behavior_challenge():
    """Analyze customer behavior patterns over time"""
    
    print("\n👥 CUSTOMER BEHAVIOR CHALLENGE")
    print("=" * 40)
    
    # Generate customer visit data
    np.random.seed(42)
    visit_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=365),
        'website_visits': np.random.poisson(1000, 365),
        'mobile_visits': np.random.poisson(500, 365),
        'conversions': np.random.poisson(50, 365)
    })
    
    # Your mission:
    # 1. Calculate conversion rates over time
    # 2. Identify mobile vs desktop usage patterns
    # 3. Find the best days for marketing campaigns
    # 4. Predict future website traffic
    
    print("Sample customer behavior data:")
    print(visit_data.head())
    
    # TODO: Add your analysis here
    
    return visit_data

# customer_data = customer_behavior_challenge()
```

## 🎯 Time Series Best Practices

### 1. **Data Quality Checks**

```python
def time_series_quality_checks(data):
    """Comprehensive time series data quality assessment"""
    
    print("\n✅ TIME SERIES QUALITY ASSESSMENT")
    print("=" * 42)
    
    # Check 1: Date continuity
    date_range = pd.date_range(data.index.min(), data.index.max(), freq='D')
    missing_dates = date_range.difference(data.index)
    print(f"Missing dates: {len(missing_dates)}")
    
    # Check 2: Duplicate dates
    duplicate_dates = data.index.duplicated().sum()
    print(f"Duplicate dates: {duplicate_dates}")
    
    # Check 3: Data completeness
    completeness = (1 - data.isnull().sum() / len(data)) * 100
    print(f"Data completeness:")
    for col, pct in completeness.items():
        print(f"  {col}: {pct:.1f}%")
    
    # Check 4: Outlier summary
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
    print(f"Statistical outliers:")
    for col, count in outliers.items():
        print(f"  {col}: {count}")
    
    # Check 5: Seasonality tests
    print(f"\nSeasonality indicators:")
    print(f"Day-of-week variance: {data.groupby(data.index.dayofweek).mean().std().iloc[0]:.2f}")
    print(f"Month variance: {data.groupby(data.index.month).mean().std().iloc[0]:.2f}")
    
    return {
        'missing_dates': len(missing_dates),
        'duplicates': duplicate_dates,
        'completeness': completeness,
        'outliers': outliers
    }

quality_report = time_series_quality_checks(ts_data[['daily_sales']])
```

### 2. **Performance Optimization**

```python
def optimize_time_series_operations():
    """Demonstrate efficient time series operations"""
    
    print("\n⚡ PERFORMANCE OPTIMIZATION")
    print("=" * 35)
    
    # Tip 1: Use vectorized operations
    print("✅ Use vectorized operations instead of loops")
    
    # Tip 2: Efficient resampling
    print("✅ Use built-in resample() instead of manual groupby")
    
    # Tip 3: Memory-efficient operations
    print("✅ Use appropriate data types")
    memory_usage = ts_data.memory_usage(deep=True).sum() / 1024**2
    print(f"Current memory usage: {memory_usage:.2f} MB")
    
    # Tip 4: Indexing performance
    print("✅ Keep datetime as index for faster time-based operations")
    
    # Tip 5: Chunking for large datasets
    print("✅ Process large time series in chunks to manage memory")
    
optimize_time_series_operations()
```

## 🎯 Key Time Series Concepts Summary

1. **Date/Time Indexing**: Use datetime index for efficient time-based operations
2. **Resampling**: Change frequency (daily → monthly) with appropriate aggregation
3. **Rolling Windows**: Smooth data and calculate moving statistics
4. **Seasonality**: Identify and account for recurring patterns
5. **Trend Analysis**: Detect long-term directional changes
6. **Forecasting**: Use historical patterns to predict future values
7. **Outlier Detection**: Identify unusual values that need investigation

## 🚀 What's Next?

You've mastered time series fundamentals! Next up: **Performance Optimization** - learn to make your Pandas operations blazingly fast for large datasets.

**Key skills unlocked:**
- ✅ DateTime operations and indexing
- ✅ Resampling and frequency conversion  
- ✅ Rolling window calculations
- ✅ Seasonal pattern analysis
- ✅ Basic forecasting techniques
- ✅ Business time series insights

Ready to optimize for speed and scale? Let's dive into **Performance Mastery**! ⚡
