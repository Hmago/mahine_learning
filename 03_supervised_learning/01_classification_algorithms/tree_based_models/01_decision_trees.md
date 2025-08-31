# Decision Trees: The If-Then Master 🌳

## What are Decision Trees? 🤔

Imagine you're a doctor diagnosing patients. You might think:
- "If fever > 38°C AND cough = Yes → Likely flu"
- "If fever < 37°C AND rash = Yes → Likely allergy"

That's exactly how decision trees work! They create a series of yes/no questions that lead to a final decision. It's like playing "20 Questions" but with data.

## Why Decision Trees are Amazing 🌟

**They think like humans!** Decision trees mirror how we naturally make decisions:

```
Should I go outside today?
├── Is it raining?
│   ├── Yes → Stay inside
│   └── No → Check temperature
│       ├── < 10°C → Too cold, stay inside  
│       ├── 10-25°C → Perfect, go outside!
│       └── > 30°C → Too hot, go to mall instead
```

## How Decision Trees Learn 📚

### The Learning Process (Intuitive)

1. **Look at all the data**: What's the best first question to ask?
2. **Split the data**: Based on that question
3. **Repeat**: For each group, find the next best question
4. **Stop**: When groups are pure enough or other criteria are met

### The Learning Process (Technical)

Decision trees use **information gain** or **Gini impurity** to decide how to split:

```python
import numpy as np
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini impurity - how 'mixed' is this group?
    0 = perfectly pure (all same class)
    0.5 = maximum impurity (50/50 split for binary)
    """
    if len(y) == 0:
        return 0
    
    # Count each class
    class_counts = Counter(y)
    total = len(y)
    
    # Calculate Gini
    gini = 1 - sum((count/total)**2 for count in class_counts.values())
    return gini

# Example
mixed_group = [0, 0, 1, 1, 0, 1]  # Mixed classes
pure_group = [0, 0, 0, 0]         # All same class

print(f"Mixed group Gini: {gini_impurity(mixed_group):.3f}")
print(f"Pure group Gini: {gini_impurity(pure_group):.3f}")
```

## Building Your First Decision Tree 🛠️

Let's create a decision tree for a classic problem: Should we play tennis today?

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Weather data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 
                'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 
                'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 
             'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical to numerical
from sklearn.preprocessing import LabelEncoder

le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder() 
le_windy = LabelEncoder()
le_play = LabelEncoder()

X = pd.DataFrame({
    'Outlook': le_outlook.fit_transform(df['Outlook']),
    'Temperature': le_temp.fit_transform(df['Temperature']),
    'Humidity': le_humidity.fit_transform(df['Humidity']),
    'Windy': le_windy.fit_transform(df['Windy'])
})

y = le_play.fit_transform(df['PlayTennis'])

# Build decision tree
tree = DecisionTreeClassifier(random_state=42, max_depth=3)
tree.fit(X, y)

# Visualize the tree
plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=X.columns, 
          class_names=['No', 'Yes'], filled=True, fontsize=10)
plt.title('Decision Tree: Should We Play Tennis?')
plt.show()

# Make predictions
new_day = [[2, 1, 0, 0]]  # Sunny, Hot, Normal humidity, No wind
prediction = tree.predict(new_day)
print(f"Should we play tennis? {'Yes' if prediction[0] == 1 else 'No'}")
```

## Understanding Tree Splits 🍂

### Information Gain

Information gain measures how much uncertainty we remove with each split:

```python
def calculate_information_gain(parent, left_child, right_child):
    """
    Calculate how much information we gain from a split
    """
    def entropy(y):
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        total = len(y)
        return -sum((count/total) * np.log2(count/total) 
                   for count in class_counts.values())
    
    # Weighted average of child entropies
    total = len(parent)
    left_weight = len(left_child) / total
    right_weight = len(right_child) / total
    
    weighted_child_entropy = (left_weight * entropy(left_child) + 
                             right_weight * entropy(right_child))
    
    return entropy(parent) - weighted_child_entropy

# Example usage
parent = [0, 0, 1, 1, 0, 1]
left_child = [0, 0, 0]      # Pure group after split
right_child = [1, 1, 1]     # Pure group after split

gain = calculate_information_gain(parent, left_child, right_child)
print(f"Information gained from this split: {gain:.3f}")
```

## Controlling Tree Growth 🌱➡️🌳

### Key Parameters

#### 1. max_depth
```python
# Shallow tree (simple, may underfit)
shallow_tree = DecisionTreeClassifier(max_depth=2)

# Deep tree (complex, may overfit)  
deep_tree = DecisionTreeClassifier(max_depth=20)

# Let's see the difference
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for i, (model, title) in enumerate([(shallow_tree, 'Shallow Tree'), 
                                   (deep_tree, 'Deep Tree')]):
    model.fit(X, y)
    plot_tree(model, ax=axes[i], feature_names=X.columns, 
              class_names=['No', 'Yes'], filled=True)
    axes[i].set_title(f'{title} (Depth: {model.tree_.max_depth})')

plt.show()
```

#### 2. min_samples_split & min_samples_leaf
```python
# Don't split unless you have enough samples
conservative_tree = DecisionTreeClassifier(
    min_samples_split=10,  # Need 10+ samples to consider splitting
    min_samples_leaf=5     # Each leaf must have 5+ samples
)
```

#### 3. max_features
```python
# Only consider a subset of features at each split
random_tree = DecisionTreeClassifier(
    max_features='sqrt'  # Consider √(total_features) at each split
)
```

## Real-World Example: Credit Approval 💳

```python
# Simplified credit approval decision tree
credit_data = {
    'Income': [25000, 45000, 35000, 60000, 30000, 80000, 40000, 55000],
    'Credit_Score': [600, 750, 650, 800, 580, 820, 700, 720],
    'Employment_Years': [1, 5, 3, 8, 0.5, 10, 4, 6],
    'Approved': [0, 1, 0, 1, 0, 1, 1, 1]  # 0=No, 1=Yes
}

credit_df = pd.DataFrame(credit_data)
X_credit = credit_df[['Income', 'Credit_Score', 'Employment_Years']]
y_credit = credit_df['Approved']

# Train decision tree
credit_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
credit_tree.fit(X_credit, y_credit)

# Visualize decision process
plt.figure(figsize=(15, 10))
plot_tree(credit_tree, feature_names=X_credit.columns,
          class_names=['Denied', 'Approved'], filled=True, fontsize=12)
plt.title('Credit Approval Decision Tree')
plt.show()

# Test new application
new_applicant = [[42000, 680, 3]]  # Income, Credit Score, Years Employed
decision = credit_tree.predict(new_applicant)
probability = credit_tree.predict_proba(new_applicant)

print(f"Decision: {'Approved' if decision[0] == 1 else 'Denied'}")
print(f"Confidence: {probability[0].max():.2f}")
```

## Feature Importance: What Matters Most? 📊

One of the best features of decision trees is they tell you which features are most important:

```python
# Get feature importance
importances = credit_tree.feature_importances_

# Create a nice visualization
feature_importance_df = pd.DataFrame({
    'Feature': X_credit.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Credit Approval')
plt.gca().invert_yaxis()

for i, v in enumerate(feature_importance_df['Importance']):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.show()

print("Feature importance ranking:")
for idx, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.3f}")
```

## Advantages & Disadvantages 📊

### ✅ Advantages

**Highly Interpretable**: You can follow the exact logic
**No Assumptions**: Works with any data distribution
**Handles Mixed Data**: Numerical and categorical features together
**Feature Selection**: Automatically identifies important features
**Fast Prediction**: Simple tree traversal
**Non-linear Relationships**: Captures complex patterns

### ❌ Disadvantages

**Overfitting Prone**: Can memorize training data
**Unstable**: Small data changes can create very different trees
**Bias**: Tends to favor features with more levels
**Poor Extrapolation**: Doesn't work well outside training data range
**Greedy**: Makes locally optimal decisions that might not be globally optimal

## Handling Overfitting 🛡️

### 1. Pre-pruning (Stop Early)
```python
# Set limits during training
controlled_tree = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth
    min_samples_split=20,  # Need 20+ samples to split
    min_samples_leaf=10,   # Each leaf needs 10+ samples
    max_features='sqrt'    # Consider only sqrt(n) features per split
)
```

### 2. Post-pruning (Cut Back Later)
```python
# Train full tree, then prune based on validation performance
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

# Find optimal ccp_alpha (cost complexity pruning)
tree = DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas

train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name='ccp_alpha', param_range=ccp_alphas, cv=5
)

# Plot to find best alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores.mean(axis=1), label='Training')
plt.plot(ccp_alphas, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Alpha (pruning strength)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Cost Complexity Pruning')
plt.show()
```

## Decision Trees for Different Problems 🎯

### Binary Classification: Medical Diagnosis
```python
# Simplified heart disease prediction
heart_data = {
    'Age': [45, 67, 29, 56, 78, 34, 61, 52],
    'Chest_Pain': [1, 3, 0, 2, 3, 1, 2, 1],  # 0-3 scale
    'Blood_Pressure': [120, 160, 110, 140, 180, 115, 150, 135],
    'Cholesterol': [200, 280, 180, 240, 320, 190, 260, 220],
    'Heart_Disease': [0, 1, 0, 1, 1, 0, 1, 0]
}

heart_df = pd.DataFrame(heart_data)
X_heart = heart_df.drop('Heart_Disease', axis=1)
y_heart = heart_df['Heart_Disease']

heart_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
heart_tree.fit(X_heart, y_heart)

# Visualize the medical decision tree
plt.figure(figsize=(20, 12))
plot_tree(heart_tree, feature_names=X_heart.columns,
          class_names=['Healthy', 'Disease'], filled=True, fontsize=10)
plt.title('Medical Decision Tree: Heart Disease Prediction')
plt.show()
```

### Multi-class Classification: Animal Recognition
```python
# Animal classification based on characteristics
animal_data = {
    'Has_Fur': [1, 0, 1, 0, 1, 0, 1, 0],
    'Flies': [0, 1, 0, 1, 0, 0, 0, 1],
    'Lives_In_Water': [0, 0, 0, 0, 0, 1, 0, 0],
    'Warm_Blooded': [1, 1, 1, 1, 1, 0, 1, 1],
    'Animal': ['Cat', 'Bird', 'Dog', 'Bat', 'Bear', 'Fish', 'Wolf', 'Eagle']
}

animal_df = pd.DataFrame(animal_data)

# Encode animals as numbers
le_animal = LabelEncoder()
X_animal = animal_df.drop('Animal', axis=1)
y_animal = le_animal.fit_transform(animal_df['Animal'])

animal_tree = DecisionTreeClassifier(random_state=42)
animal_tree.fit(X_animal, y_animal)

# Make prediction for new animal
new_animal = [[1, 0, 0, 1]]  # Has fur, doesn't fly, doesn't live in water, warm blooded
predicted_animal = le_animal.inverse_transform(animal_tree.predict(new_animal))
print(f"This animal is probably a: {predicted_animal[0]}")
```

## Understanding Tree Visualization 👁️

When you see a decision tree diagram:

```
                    [Root Node]
                 Credit_Score <= 650.5
                   gini = 0.5
                  samples = 100
                   value = [50, 50]
                      /        \
               [Left Child]    [Right Child]
              Income <= 35000   Income > 45000
                gini = 0.3        gini = 0.2
               samples = 40      samples = 30
              value = [35, 5]   value = [5, 25]
```

**Reading the nodes:**
- **Top line**: The split condition
- **gini**: Impurity measure (lower = more pure)
- **samples**: Number of data points in this node
- **value**: [class_0_count, class_1_count]
- **Color intensity**: Darker = more pure

## Common Splitting Criteria 📏

### 1. Gini Impurity (Default)
```python
tree_gini = DecisionTreeClassifier(criterion='gini')
```
**Good for**: General purpose, slightly faster

### 2. Entropy (Information Gain)
```python
tree_entropy = DecisionTreeClassifier(criterion='entropy')
```
**Good for**: When you want to maximize information gain

### 3. Log Loss (For probability estimates)
```python
tree_log_loss = DecisionTreeClassifier(criterion='log_loss')
```
**Good for**: When you need well-calibrated probabilities

## Preventing Overfitting: Best Practices 🛡️

### 1. Set Reasonable Limits
```python
# Balanced tree settings
balanced_tree = DecisionTreeClassifier(
    max_depth=6,              # Not too deep
    min_samples_split=20,     # Need enough samples to split
    min_samples_leaf=10,      # Leaves can't be too small
    min_impurity_decrease=0.01, # Split must improve purity significantly
    random_state=42
)
```

### 2. Use Cross-Validation for Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42), 
    param_grid, cv=5, scoring='accuracy'
)

grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
```

### 3. Ensemble Methods (Preview)
```python
from sklearn.ensemble import RandomForestClassifier

# Multiple trees voting together (we'll learn this later!)
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X, y)
print(f"Forest accuracy: {forest.score(X, y):.3f}")
```

## Decision Trees vs Other Algorithms 🥊

| Feature | Decision Trees | Logistic Regression | SVM |
|---------|---------------|-------------------|-----|
| **Interpretability** | 🌟🌟🌟 | 🌟🌟 | 🌟 |
| **Handle Non-linear** | 🌟🌟🌟 | 🌟 | 🌟🌟🌟 |
| **Training Speed** | 🌟🌟🌟 | 🌟🌟🌟 | 🌟🌟 |
| **Prediction Speed** | 🌟🌟🌟 | 🌟🌟🌟 | 🌟🌟 |
| **Overfitting Risk** | 🌟 | 🌟🌟 | 🌟🌟 |
| **Feature Scaling** | Not needed | Required | Required |

## Advanced Decision Tree Concepts 🚀

### 1. Handling Categorical Features
```python
# Decision trees naturally handle categorical data
# No need for one-hot encoding!

mixed_data = pd.DataFrame({
    'Age': [25, 45, 35, 55],
    'City': ['NYC', 'LA', 'Chicago', 'NYC'],  # Categorical
    'Income': [50000, 80000, 60000, 90000],
    'Approved': [0, 1, 0, 1]
})

# Convert only categorical columns
X_mixed = pd.get_dummies(mixed_data[['Age', 'City', 'Income']])
y_mixed = mixed_data['Approved']

tree = DecisionTreeClassifier()
tree.fit(X_mixed, y_mixed)
```

### 2. Missing Value Handling
```python
# Decision trees can handle missing values with surrogate splits
# (though sklearn's implementation requires preprocessing)

from sklearn.impute import SimpleImputer

# Fill missing values before training
imputer = SimpleImputer(strategy='median')
X_filled = imputer.fit_transform(X_with_missing)
```

### 3. Regression Trees
```python
from sklearn.tree import DecisionTreeRegressor

# Predicting continuous values (house prices)
house_tree = DecisionTreeRegressor(max_depth=5)
# Uses MSE (Mean Squared Error) instead of Gini/Entropy
```

## Implementation from Scratch (Simplified) 🔨

Here's a basic decision tree implementation to understand the core algorithm:

```python
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        
    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = 0
        
        current_impurity = self.gini_impurity(y)
        
        for feature in range(X.shape[1]):
            # Try different thresholds
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                # Calculate information gain
                left_impurity = self.gini_impurity(y[left_mask])
                right_impurity = self.gini_impurity(y[right_mask])
                
                # Weighted average
                n_left, n_right = len(y[left_mask]), len(y[right_mask])
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / len(y)
                
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def fit(self, X, y, depth=0):
        # Stop conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            # Create leaf node
            return {'class': np.bincount(y).argmax()}
        
        # Find best split
        feature, threshold, gain = self.find_best_split(X, y)
        
        if gain == 0:
            return {'class': np.bincount(y).argmax()}
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        tree = {
            'feature': feature,
            'threshold': threshold,
            'left': self.fit(X[left_mask], y[left_mask], depth + 1),
            'right': self.fit(X[right_mask], y[right_mask], depth + 1)
        }
        
        return tree
    
    def predict_sample(self, x, tree):
        # Traverse tree for single sample
        if 'class' in tree:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])

# Example usage
simple_tree = SimpleDecisionTree(max_depth=3)
tree_structure = simple_tree.fit(X.values, y)
print("Simple decision tree trained!")
```

## When to Use Decision Trees 🎯

### Perfect for:
- **Exploratory analysis**: Understanding data patterns
- **Rule extraction**: When you need to explain decisions
- **Mixed data types**: Numerical and categorical together
- **Feature selection**: Identifying important variables
- **Quick prototyping**: Fast to train and interpret

### Consider alternatives when:
- **High accuracy required**: Ensemble methods often perform better
- **Stable models needed**: Small data changes shouldn't affect model much
- **Extrapolation required**: Predicting outside training data range
- **Very noisy data**: More robust algorithms might work better

## Common Pitfalls & Solutions ⚠️

### 1. Growing Trees Too Deep
```python
# Problem: Overfitting
deep_tree = DecisionTreeClassifier(max_depth=None)  # Unlimited depth!

# Solution: Set reasonable limits
reasonable_tree = DecisionTreeClassifier(max_depth=8)
```

### 2. Ignoring Class Imbalance
```python
# Problem: Tree biased toward majority class
# Solution: Use class_weight parameter
balanced_tree = DecisionTreeClassifier(class_weight='balanced')
```

### 3. Not Validating Results
```python
# Always check performance on unseen data
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Advanced Tips 💡

### 1. Visualizing Decision Boundaries
```python
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.title(title)
    plt.show()
```

### 2. Feature Engineering for Trees
```python
# Trees benefit from feature engineering too!
def create_tree_features(df):
    """Create features that trees can easily split on"""
    
    # Binning continuous variables
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    # Interaction features
    df['Income_per_Year_Employed'] = df['Income'] / (df['Years_Employed'] + 1)
    
    # Boolean features
    df['High_Income'] = (df['Income'] > df['Income'].median()).astype(int)
    
    return df
```

## Key Takeaways 🎯

1. **Decision trees mirror human decision-making** with if-then rules
2. **They're highly interpretable** - you can follow the exact logic
3. **No feature scaling required** - trees are scale-invariant
4. **Prone to overfitting** - always validate and prune
5. **Great for exploratory analysis** and understanding data
6. **Foundation for powerful ensemble methods** (Random Forest, XGBoost)
7. **Handle mixed data types** naturally

## Next Steps 🚀

1. **Practice**: Try the interactive notebook `../../notebooks/04_decision_trees_lab.ipynb`
2. **Learn ensembles**: Multiple trees are much better than one! `02_random_forest.md`
3. **Experiment**: Build trees for your own data
4. **Compare**: How do trees perform vs logistic regression on the same data?

## Quick Challenge 💪

Create a decision tree that can classify whether a person should:
- **Wear a jacket** based on temperature and wind
- **Bring an umbrella** based on humidity and cloud cover
- **Go for a run** based on weather conditions

Can you make the tree interpretable enough that anyone could follow the decision process?

*Solution and more challenges in the exercises folder!*
