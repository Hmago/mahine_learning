# Exploratory Data Analysis (EDA)

## What is Exploratory Data Analysis?

Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics, often using visual methods. It is a crucial step in the data analysis process, allowing data scientists to understand the data better before applying any machine learning models.

## Why Does This Matter?

Understanding your data is essential for making informed decisions in machine learning. EDA helps identify patterns, spot anomalies, test hypotheses, and check assumptions. By performing EDA, you can ensure that the data is suitable for modeling and that you are aware of any potential issues that could affect your results.

## Key Components of EDA

1. **Data Visualization**: Using graphs and plots to visualize data distributions and relationships.
   - **Histograms**: Show the distribution of a single variable.
   - **Box Plots**: Highlight the median, quartiles, and potential outliers.
   - **Scatter Plots**: Illustrate relationships between two numerical variables.

2. **Summary Statistics**: Calculating measures such as mean, median, mode, standard deviation, and correlation coefficients to understand the data's central tendency and variability.

3. **Data Cleaning**: Identifying and handling missing values, duplicates, and outliers to ensure data quality.

4. **Feature Relationships**: Exploring how different features relate to each other and to the target variable, which can inform feature selection and engineering.

## Practical Example

Imagine you are working with a dataset containing information about houses for sale. You want to predict house prices based on various features such as size, location, and number of bedrooms. Before building a model, you would perform EDA to:

- Visualize the distribution of house prices to understand their range and identify any skewness.
- Create scatter plots to see how house size correlates with price.
- Check for missing values in critical features like location or size, which could affect your model's performance.

## Thought Experiment

Consider a dataset of customer reviews for a product. What steps would you take to explore this data? Think about the types of visualizations you could create and the summary statistics you would calculate to gain insights into customer satisfaction.

## Conclusion

Exploratory Data Analysis is a foundational step in the data science workflow. By thoroughly understanding your data through EDA, you can make better decisions about how to preprocess it, which features to include, and how to approach modeling. Always remember: "A good model is built on a good understanding of the data."

## Mathematical Foundation

### Key Formulas

**Descriptive Statistics:**

**Mean:**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Standard Deviation:**
$$s = \sqrt{s^2}$$

**Coefficient of Variation:**
$$CV = \frac{s}{\bar{x}} \times 100\%$$

**Pearson Correlation Coefficient:**
$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

**Quartiles:**
- $Q_1$: 25th percentile
- $Q_2$: 50th percentile (median)  
- $Q_3$: 75th percentile

**Skewness:**
$$Skewness = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

**Kurtosis:**
$$Kurtosis = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

### Solved Examples

#### Example 1: Summary Statistics Calculation

Given: House prices (in $1000s): [200, 250, 180, 320, 280, 190, 350]

Find: Complete descriptive statistics

Solution:
Step 1: Calculate mean
$$\bar{x} = \frac{200 + 250 + 180 + 320 + 280 + 190 + 350}{7} = \frac{1770}{7} = 252.86$$

Step 2: Calculate variance
$$s^2 = \frac{1}{6}[(200-252.86)^2 + (250-252.86)^2 + ... + (350-252.86)^2]$$
$$s^2 = \frac{1}{6}[2792.28 + 8.18 + 5305.48 + 4508.58 + 738.98 + 3959.08 + 9429.38] = 4456.99$$

Step 3: Calculate standard deviation
$$s = \sqrt{4456.99} = 66.76$$

Step 4: Calculate coefficient of variation
$$CV = \frac{66.76}{252.86} \times 100\% = 26.4\%$$

Result: Mean = $252.86k, SD = $66.76k, CV = 26.4% (moderate variability)

#### Example 2: Correlation Analysis

Given: Two features from housing dataset
Size (sqft): [1000, 1500, 2000, 1200, 1800]
Price ($1000s): [180, 270, 360, 210, 340]

Find: Pearson correlation coefficient

Solution:
Step 1: Calculate means
$$\bar{x} = \frac{1000 + 1500 + 2000 + 1200 + 1800}{5} = 1500$$
$$\bar{y} = \frac{180 + 270 + 360 + 210 + 340}{5} = 272$$

Step 2: Calculate numerator
$$\sum(x_i - \bar{x})(y_i - \bar{y}) = (1000-1500)(180-272) + (1500-1500)(270-272) + ...$$
$$= (-500)(-92) + (0)(-2) + (500)(88) + (-300)(-62) + (300)(68)$$
$$= 46000 + 0 + 44000 + 18600 + 20400 = 129000$$

Step 3: Calculate denominators
$$\sum(x_i - \bar{x})^2 = 500^2 + 0^2 + 500^2 + 300^2 + 300^2 = 700000$$
$$\sum(y_i - \bar{y})^2 = 92^2 + 2^2 + 88^2 + 62^2 + 68^2 = 24076$$

Step 4: Calculate correlation
$$r = \frac{129000}{\sqrt{700000 \times 24076}} = \frac{129000}{\sqrt{16853200}} = \frac{129000}{4105.5} = 0.94$$

Result: Strong positive correlation (r = 0.94) between size and price.

#### Example 4: Outlier Detection Using IQR

Given: Income data ($1000s): [45, 50, 48, 52, 180, 49, 51, 47]

Find: Identify outliers using IQR method

Solution:
Step 1: Sort data and find quartiles
Sorted: [45, 47, 48, 49, 50, 51, 52, 180]
$$Q_1 = 47.5, \quad Q_2 = 49.5, \quad Q_3 = 51.5$$

Step 2: Calculate IQR
$$IQR = Q_3 - Q_1 = 51.5 - 47.5 = 4$$

Step 3: Determine outlier bounds
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR = 47.5 - 1.5 \times 4 = 41.5$$
$$\text{Upper Bound} = Q_3 + 1.5 \times IQR = 51.5 + 1.5 \times 4 = 57.5$$

Step 4: Identify outliers
Values outside [41.5, 57.5]: Only 180 is an outlier.

Result: 180 (income of $180k) is identified as an outlier.

**EDA Workflow:**
1. **Data Overview**: Calculate basic statistics and data types
2. **Distribution Analysis**: Examine histograms and density plots  
3. **Relationship Exploration**: Compute correlations and create scatter plots
4. **Outlier Detection**: Apply statistical methods to identify anomalies
5. **Pattern Recognition**: Look for trends, seasonality, and clusters