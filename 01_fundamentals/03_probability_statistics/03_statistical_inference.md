# Statistical Inference

Statistical inference is a fundamental concept in statistics that allows us to make conclusions about a population based on a sample of data. It provides the tools to estimate population parameters, test hypotheses, and make predictions.

## What is Statistical Inference?

Statistical inference involves using data from a sample to infer properties about a larger population. This process is crucial in machine learning and data analysis, as it helps us understand the underlying patterns and relationships in data.

### Key Concepts

1. **Population vs. Sample**:
   - **Population**: The entire group of individuals or instances about which we want to draw conclusions.
   - **Sample**: A subset of the population used to gather insights and make inferences.

2. **Point Estimation**:
   - A single value estimate of a population parameter (e.g., the sample mean as an estimate of the population mean).

3. **Confidence Intervals**:
   - A range of values, derived from the sample, that is likely to contain the population parameter. For example, a 95% confidence interval suggests that if we were to take many samples, 95% of the intervals would contain the true population parameter.

4. **Hypothesis Testing**:
   - A method for testing a claim or hypothesis about a population parameter. It involves:
     - **Null Hypothesis (H0)**: The hypothesis that there is no effect or difference.
     - **Alternative Hypothesis (H1)**: The hypothesis that there is an effect or difference.
     - **P-value**: The probability of observing the sample data, or something more extreme, if the null hypothesis is true. A low p-value (typically < 0.05) indicates strong evidence against the null hypothesis.

### Why Does This Matter?

Understanding statistical inference is crucial for making informed decisions based on data. In machine learning, it helps us evaluate model performance, understand uncertainty, and make predictions about unseen data. For instance, when developing a model to predict customer behavior, statistical inference allows us to assess how well our model generalizes to the entire customer base based on a sample.

### Practical Example

Imagine you want to know the average height of adult men in a city. Measuring every man in the city is impractical, so you take a sample of 100 men. You find that the average height in your sample is 175 cm. Using statistical inference, you can estimate the average height of all men in the city and construct a confidence interval around your sample mean to express the uncertainty of your estimate.

### Thought Experiment

Consider a scenario where you are testing a new drug. You conduct a study with a sample of patients and find that the drug lowers blood pressure. How would you use statistical inference to determine if this effect is significant and applicable to the entire population of patients with high blood pressure?

## Conclusion

Statistical inference is a powerful tool that enables us to make data-driven decisions and understand the uncertainty inherent in our conclusions. By mastering these concepts, you will be better equipped to analyze data and draw meaningful insights in your machine learning projects.

## Suggested Exercises

- Calculate the confidence interval for a given sample mean and standard deviation.
- Conduct a hypothesis test for a sample dataset and interpret the results.
- Explore how changing the sample size affects the confidence interval width.

This file serves as an introduction to statistical inference, laying the groundwork for more advanced topics in probability and statistics.

## Mathematical Foundation

### Key Formulas

**Point Estimation:**

**Sample Mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$

**Sample Variance:** $s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$

**Standard Error:** $SE = \frac{s}{\sqrt{n}}$

**Confidence Intervals:**

**For Population Mean (known $\sigma$):**
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**For Population Mean (unknown $\sigma$):**
$$\bar{x} \pm t_{\alpha/2,n-1} \cdot \frac{s}{\sqrt{n}}$$

**Hypothesis Testing:**

**Test Statistic (Z-test):**
$$Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$$

**Test Statistic (t-test):**
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$$

**P-value:** $P = P(\text{test statistic} \geq \text{observed value}|\text{H}_0 \text{ true})$

### Solved Examples

#### Example 1: Confidence Interval for Population Mean

Given: Sample of 25 students with mean test score $\bar{x} = 82$, sample standard deviation $s = 8$

Find: 95% confidence interval for population mean

Solution:
Step 1: Identify parameters
- $n = 25$, $\bar{x} = 82$, $s = 8$
- Confidence level = 95%, so $\alpha = 0.05$
- Degrees of freedom = $n - 1 = 24$

Step 2: Find critical t-value
For $\alpha/2 = 0.025$ and $df = 24$: $t_{0.025,24} = 2.064$

Step 3: Calculate standard error
$$SE = \frac{s}{\sqrt{n}} = \frac{8}{\sqrt{25}} = \frac{8}{5} = 1.6$$

Step 4: Calculate confidence interval
$$CI = 82 \pm 2.064 \times 1.6 = 82 \pm 3.30$$
$$CI = [78.70, 85.30]$$

Result: We are 95% confident that the true population mean is between 78.70 and 85.30.

#### Example 2: Hypothesis Testing (One-sample t-test)

Given: Company claims average delivery time is 2 days. Sample of 20 deliveries shows $\bar{x} = 2.3$ days, $s = 0.5$ days.

Test: Is the actual delivery time significantly different from claimed time? (Use $\alpha = 0.05$)

Solution:
Step 1: Set up hypotheses
- $H_0: \mu = 2$ (delivery time equals claim)
- $H_1: \mu \neq 2$ (delivery time differs from claim)

Step 2: Calculate test statistic
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} = \frac{2.3 - 2}{0.5/\sqrt{20}} = \frac{0.3}{0.112} = 2.68$$

Step 3: Find critical values
For two-tailed test with $\alpha = 0.05$ and $df = 19$: $t_{critical} = \pm 2.093$

Step 4: Make decision
Since $|2.68| > 2.093$, we reject $H_0$.

Step 5: Calculate p-value
$p = 2 \times P(t_{19} > 2.68) \approx 2 \times 0.008 = 0.016$

Result: Since $p = 0.016 < 0.05$, delivery time is significantly different from claimed 2 days.

#### Example 3: Comparing Two Populations (Two-sample t-test)

Given: Algorithm A: $n_1 = 15$, $\bar{x_1} = 92.5\%$, $s_1 = 3.2\%$
Algorithm B: $n_2 = 12$, $\bar{x_2} = 89.8\%$, $s_2 = 2.8\%$

Test: Is Algorithm A significantly better than Algorithm B? (Use $\alpha = 0.05$)

Solution:
Step 1: Set up hypotheses
- $H_0: \mu_1 = \mu_2$ (no difference in performance)
- $H_1: \mu_1 > \mu_2$ (Algorithm A is better)

Step 2: Calculate pooled standard deviation
$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}} = \sqrt{\frac{14(3.2)^2 + 11(2.8)^2}{25}} = \sqrt{\frac{229.76}{25}} = 3.03$$

Step 3: Calculate test statistic
$$t = \frac{\bar{x_1} - \bar{x_2}}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} = \frac{92.5 - 89.8}{3.03\sqrt{\frac{1}{15} + \frac{1}{12}}} = \frac{2.7}{1.17} = 2.31$$

Step 4: Find critical value
For one-tailed test with $\alpha = 0.05$ and $df = 25$: $t_{critical} = 1.708$

Step 5: Make decision
Since $2.31 > 1.708$, we reject $H_0$.

Result: Algorithm A is significantly better than Algorithm B ($p \approx 0.015$).