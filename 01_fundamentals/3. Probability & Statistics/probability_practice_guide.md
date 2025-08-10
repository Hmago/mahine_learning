# Probability and Statistics Practice Guide

Hands-on exercises and practical examples for mastering probability and statistics concepts in machine learning.

## 🎯 Practice Structure

### Core Practice Areas
1. **Probability Calculations** - Build intuition with real examples
2. **Distribution Analysis** - Identify and work with different distributions  
3. **Bayesian Reasoning** - Apply Bayes' theorem to ML problems
4. **Statistical Testing** - Practical hypothesis testing scenarios
5. **ML Applications** - Real-world machine learning problems

### Practice Projects
1. **A/B Testing with Statistical Significance**
2. **Monte Carlo Simulations**  
3. **Spam Detection using Bayesian Methods**

---

## 📊 Practice Section 1: Probability Calculations

### Exercise 1.1: Conditional Probability
**Scenario**: Email Classification System

Given:
- 60% of emails are legitimate
- 40% of emails are spam
- 80% of spam emails contain the word "urgent"
- 5% of legitimate emails contain the word "urgent"

**Questions**:
1. What's the probability an email contains "urgent"?
2. If an email contains "urgent", what's the probability it's spam?
3. If an email doesn't contain "urgent", what's the probability it's legitimate?

**Solution Approach**:
```
Let S = Spam, L = Legitimate, U = Contains "urgent"

Given:
P(L) = 0.6, P(S) = 0.4
P(U|S) = 0.8, P(U|L) = 0.05

1. P(U) = P(U|S)P(S) + P(U|L)P(L) = 0.8×0.4 + 0.05×0.6 = 0.35

2. P(S|U) = P(U|S)P(S) / P(U) = (0.8×0.4) / 0.35 = 0.914

3. P(L|U') = P(U'|L)P(L) / P(U') = (0.95×0.6) / 0.65 = 0.877
```

### Exercise 1.2: Multiple Conditions
**Scenario**: Medical Diagnosis

A diagnostic test has:
- Sensitivity (True Positive Rate): 95%
- Specificity (True Negative Rate): 90%
- Disease prevalence: 2%

**Challenge**: Calculate the positive predictive value (precision)

### Exercise 1.3: Chain Rule Application
**Scenario**: Sequential Decision Making

A recommendation system makes three sequential recommendations:
- P(Click₁) = 0.3
- P(Click₂|Click₁) = 0.6, P(Click₂|¬Click₁) = 0.1
- P(Click₃|Click₁,Click₂) = 0.8, P(Click₃|Click₁,¬Click₂) = 0.4

Calculate P(Click₁ ∩ Click₂ ∩ Click₃)

---

## 📈 Practice Section 2: Distribution Analysis

### Exercise 2.1: Normal Distribution Applications

**Dataset**: Student Heights
- Mean: 170 cm, Standard Deviation: 10 cm

**Tasks**:
1. What percentage of students are between 160-180 cm?
2. Find the height that 90% of students are below
3. If we sample 25 students, what's the probability the sample mean is above 172 cm?

**Code Framework**:
```python
import numpy as np
from scipy import stats

# Given parameters
mu, sigma = 170, 10
n = 25

# Task 1: P(160 < X < 180)
prob1 = stats.norm.cdf(180, mu, sigma) - stats.norm.cdf(160, mu, sigma)

# Task 2: 90th percentile
height_90 = stats.norm.ppf(0.9, mu, sigma)

# Task 3: Sample mean distribution
sigma_sample = sigma / np.sqrt(n)
prob3 = 1 - stats.norm.cdf(172, mu, sigma_sample)
```

### Exercise 2.2: Binomial Distribution in A/B Testing

**Scenario**: Website Conversion Testing
- Control group: 1000 visitors, 50 conversions
- Treatment group: 1000 visitors, 65 conversions

**Questions**:
1. What's the 95% confidence interval for each conversion rate?
2. Is the difference statistically significant?
3. What's the probability of observing ≥65 conversions in treatment if true rate equals control?

### Exercise 2.3: Poisson Distribution for Event Modeling

**Scenario**: Server Request Analysis
- Average: 12 requests per minute
- Need to plan server capacity

**Tasks**:
1. Probability of exactly 15 requests in a minute?
2. Probability of more than 20 requests in a minute?
3. What capacity should handle 99% of scenarios?
4. In a 5-minute window, what's the expected number and variance?

### Exercise 2.4: Exponential Distribution for Reliability

**Scenario**: Component Failure Analysis
- Average failure time: 1000 hours

**Questions**:
1. Probability component lasts more than 1500 hours?
2. Given it survived 500 hours, probability it lasts another 1000?
3. What's the median lifetime?

---

## 🧠 Practice Section 3: Bayesian Reasoning

### Exercise 3.1: Naive Bayes Classifier

**Dataset**: Text Classification
```
Words: ["free", "money", "urgent", "meeting", "project"]
Spam emails: 100 (contain specific word frequencies)
Ham emails: 200 (contain different word frequencies)
```

**Task**: Build a Naive Bayes classifier from scratch

**Implementation Framework**:
```python
class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.word_likelihoods = {}
    
    def fit(self, documents, labels):
        # Calculate P(Class)
        # Calculate P(Word|Class) for each word
        pass
    
    def predict(self, document):
        # Apply Bayes' theorem
        # Return most likely class
        pass
```

### Exercise 3.2: Bayesian Parameter Estimation

**Scenario**: Coin Flip Analysis
- Unknown coin bias θ
- Prior belief: θ ~ Beta(2, 2) (slightly fair)
- Observed: 7 heads in 10 flips

**Tasks**:
1. Calculate posterior distribution
2. Find posterior mean and variance
3. Predict probability of next flip being heads
4. Compare with maximum likelihood estimate

### Exercise 3.3: Bayesian Model Comparison

**Scenario**: Which model better explains data?
- Model 1: Normal distribution
- Model 2: Exponential distribution  
- Data: [1.2, 2.3, 0.8, 1.7, 1.1, 2.9, 1.5]

**Compare using**:
- Bayes factors
- Model evidence
- Cross-validation

---

## 📊 Practice Section 4: Statistical Testing

### Exercise 4.1: Hypothesis Testing Framework

**Scenario**: Drug Effectiveness Study
- Claim: New drug reduces recovery time
- Control group: μ = 12 days, σ = 3 days, n = 30
- Treatment group: μ = 10.5 days, σ = 2.8 days, n = 35

**Complete Analysis**:
1. Formulate hypotheses
2. Choose appropriate test
3. Calculate test statistic
4. Find p-value
5. Make decision (α = 0.05)
6. Interpret results

### Exercise 4.2: Multiple Testing Correction

**Scenario**: Gene Expression Analysis
- Testing 1000 genes for differential expression
- Using α = 0.05 for each test

**Questions**:
1. How many false positives do you expect?
2. Apply Bonferroni correction
3. Apply FDR correction
4. Compare results

### Exercise 4.3: Power Analysis

**Scenario**: A/B Test Planning
- Want to detect 2% improvement in conversion rate
- Current rate: 5%
- Desired power: 80%
- Significance level: 5%

**Calculate**: Required sample size

---

## 💪 Practice Projects

## Project 1: A/B Testing with Statistical Significance

### Objective
Build a complete A/B testing framework with proper statistical analysis.

### Dataset
Create simulated e-commerce data:
- Control: 10,000 visitors, 500 conversions (5% rate)
- Treatment: 10,000 visitors, 550 conversions (5.5% rate)

### Requirements
1. **Statistical Test Design**
   - Formulate hypotheses
   - Choose appropriate test (z-test for proportions)
   - Set significance level and power

2. **Implementation**
   - Calculate test statistic
   - Compute p-value
   - Construct confidence intervals
   - Make statistical decision

3. **Advanced Analysis**
   - Sequential testing (optional stopping)
   - Bayesian A/B testing
   - Multi-armed bandit approach

4. **Results Interpretation**
   - Practical vs statistical significance
   - Business impact assessment
   - Recommendation

### Code Structure
```python
class ABTestAnalyzer:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
    
    def design_test(self, baseline_rate, effect_size):
        """Calculate required sample size"""
        pass
    
    def analyze_results(self, control_data, treatment_data):
        """Perform statistical test"""
        pass
    
    def generate_report(self):
        """Create comprehensive report"""
        pass
```

### Expected Outcomes
- P-value calculation
- Confidence intervals
- Effect size estimation
- Power analysis
- Business recommendations

---

## Project 2: Monte Carlo Simulations

### Objective
Use Monte Carlo methods to solve complex probability problems and validate analytical solutions.

### Applications

#### 2.1: Portfolio Risk Assessment
**Scenario**: Investment portfolio with 3 assets
- Asset returns follow multivariate normal distribution
- Correlation between assets
- Calculate Value at Risk (VaR)

**Implementation**:
```python
def portfolio_simulation(weights, returns, cov_matrix, n_simulations=10000):
    """
    Simulate portfolio returns and calculate risk metrics
    """
    # Generate correlated random returns
    # Calculate portfolio returns
    # Compute VaR and Expected Shortfall
    pass
```

#### 2.2: Queueing System Analysis
**Scenario**: Customer service call center
- Arrival rate: Poisson process
- Service times: Exponential distribution
- Calculate wait times and system utilization

#### 2.3: Option Pricing
**Scenario**: European call option
- Stock price follows geometric Brownian motion
- Use Monte Carlo to price option
- Compare with Black-Scholes formula

### Validation Exercises
1. **Central Limit Theorem**: Demonstrate convergence to normal
2. **Law of Large Numbers**: Show convergence of sample mean
3. **Confidence Interval Coverage**: Verify 95% CI contains true parameter 95% of time

### Advanced Topics
- Variance reduction techniques
- Importance sampling
- Markov Chain Monte Carlo (MCMC)

---

## Project 3: Spam Detection using Bayesian Methods

### Objective
Build a spam detection system using Bayesian principles and compare different approaches.

### Dataset
Use a real email dataset (e.g., Enron emails) or create synthetic data:
- 5000 spam emails
- 5000 legitimate emails
- Extract features: word frequencies, email length, sender patterns

### Implementation Stages

#### Stage 1: Basic Naive Bayes
```python
class SpamDetector:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.vocab = set()
        self.class_priors = {}
        self.word_likelihoods = {}
    
    def preprocess_email(self, email_text):
        """Clean and tokenize email"""
        pass
    
    def train(self, emails, labels):
        """Train Naive Bayes classifier"""
        pass
    
    def predict_proba(self, email):
        """Return probability of spam"""
        pass
    
    def predict(self, email):
        """Binary classification"""
        pass
```

#### Stage 2: Enhanced Features
- **Numerical features**: Email length, number of links, capital letters ratio
- **Metadata**: Time of day, sender domain
- **Advanced text**: N-grams, TF-IDF scores

#### Stage 3: Bayesian Network
Model dependencies between features:
- Word co-occurrences
- Feature correlations
- Hierarchical structure

#### Stage 4: Online Learning
Update model with new emails:
- Bayesian updating
- Forgetting factor for concept drift
- Active learning for uncertain cases

### Evaluation Metrics
1. **Classification Metrics**
   - Accuracy, Precision, Recall, F1-score
   - ROC curve and AUC
   - Precision-Recall curve

2. **Probabilistic Metrics**
   - Log-likelihood
   - Brier score
   - Calibration plots

3. **Business Metrics**
   - False positive cost (legitimate emails marked as spam)
   - False negative cost (spam emails getting through)
   - User satisfaction

### Model Comparison
Compare against:
- Logistic Regression
- Support Vector Machines
- Random Forest
- Neural Networks

### Uncertainty Quantification
- Prediction confidence intervals
- Feature importance uncertainty
- Model selection uncertainty

---

## 🔧 Tools and Libraries

### Python Essentials
```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical computing
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

# Machine learning
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Bayesian computing
import pymc3 as pm  # or PyMC4/5
import arviz as az

# Specialized libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
```

### R Alternatives
```r
# Core packages
library(tidyverse)
library(ggplot2)

# Statistical testing
library(broom)
library(infer)

# Bayesian analysis
library(rstanarm)
library(brms)
library(MCMCpack)
```

---

## 📚 Self-Assessment Checklist

### Probability Basics ✅
- [ ] Can calculate conditional probabilities
- [ ] Understands independence vs dependence
- [ ] Applies Bayes' theorem correctly
- [ ] Solves multi-step probability problems

### Distributions ✅
- [ ] Identifies appropriate distribution for given scenario
- [ ] Calculates probabilities using distribution functions
- [ ] Understands parameter relationships
- [ ] Applies distributions to ML problems

### Statistical Inference ✅
- [ ] Formulates appropriate hypotheses
- [ ] Chooses correct statistical test
- [ ] Interprets p-values and confidence intervals
- [ ] Understands Type I/II errors

### Bayesian Methods ✅
- [ ] Implements Naive Bayes from scratch
- [ ] Updates beliefs with new evidence
- [ ] Compares Bayesian vs Frequentist approaches
- [ ] Applies Bayesian thinking to ML problems

### Practical Applications ✅
- [ ] Designs and analyzes A/B tests
- [ ] Implements Monte Carlo simulations
- [ ] Builds probabilistic classifiers
- [ ] Quantifies uncertainty in predictions

---

## 🎯 Next Steps

1. **Complete all exercises** in order
2. **Implement the three projects** with real data
3. **Apply concepts** to your own ML projects
4. **Study advanced topics**: Bayesian networks, MCMC, variational inference
5. **Practice regularly** with new problems and datasets

Remember: **Understanding comes through practice!** Work through problems step-by-step and don't skip the implementation details.
