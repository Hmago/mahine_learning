# Traditional Machine Learning for Text Classification

## 🎯 What You'll Learn

Traditional ML algorithms are the workhorses of text classification. They're fast, interpretable, and often perform surprisingly well compared to complex neural networks. You'll master the core algorithms that power most production text classification systems.

## 🧠 Why Traditional ML Works So Well for Text

Think of text classification like sorting mail at a post office. You look for specific patterns:
- **Address patterns** → Geographic sorting
- **Company names** → Business mail
- **Personal names** → Residential mail
- **Keywords** → Specific departments

Traditional ML algorithms excel at finding these patterns in text data!

## 🔍 The Big Three: Core Algorithms for Text

### 1. Naive Bayes: The Probability Master

**What it does:** Calculates the probability that a document belongs to each category based on the words it contains.

**Why it's "naive":** It assumes all words are independent (which isn't true, but works surprisingly well!).

**Real-world analogy:** Like a spam filter that says "emails with 'lottery' + 'winner' + 'urgent' are 95% likely to be spam."

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Sample email classification data
emails = [
    "Win lottery now! Click here for millions! Urgent response needed!",
    "Meeting scheduled for tomorrow at 2pm in conference room A",
    "Free money! No questions asked! Limited time offer!",
    "Please review the quarterly financial report attached",
    "Congratulations! You've won $1,000,000! Act now!",
    "Team lunch scheduled for Friday at the Italian restaurant",
    "Amazing deal! 90% off everything! Buy now or miss out!",
    "The project timeline needs to be updated by end of week"
]

labels = ["spam", "work", "spam", "work", "spam", "work", "spam", "work"]

# Create Naive Bayes pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.3, random_state=42)

# Train and evaluate
nb_pipeline.fit(X_train, y_train)
predictions = nb_pipeline.predict(X_test)

print("Naive Bayes Results:")
print(classification_report(y_test, predictions))

# Test with new emails
new_emails = [
    "Urgent: Team meeting moved to 3pm",
    "Free iPhone! Click now to claim yours!"
]

for email in new_emails:
    prediction = nb_pipeline.predict([email])[0]
    probability = nb_pipeline.predict_proba([email])[0]
    print(f"Email: '{email[:30]}...'")
    print(f"Prediction: {prediction} (confidence: {max(probability):.2f})")
    print()
```

**When to use Naive Bayes:**
- **Text classification tasks** (spam detection, sentiment analysis)
- **Small to medium datasets**
- **When you need fast, interpretable results**
- **As a baseline model** to compare against

### 2. Support Vector Machines (SVM): The Boundary Finder

**What it does:** Finds the best boundary that separates different categories of text.

**How it works:** Imagine plotting documents in space based on their word features, then drawing the clearest line between categories.

**Real-world analogy:** Like a bouncer at a club who learns the perfect rules to distinguish VIP guests from regular customers.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

# Document classification example
documents = [
    "The latest iPhone features an amazing camera and fast processor",
    "Scientists discover new exoplanet 100 light years away",
    "Stock market reaches new highs amid economic growth",
    "New smartphone technology revolutionizes mobile photography",
    "Research team finds cure for rare genetic disease",
    "Federal Reserve announces interest rate changes",
    "Gaming laptop with RTX graphics card now available",
    "Climate change study reveals alarming temperature trends",
    "Cryptocurrency prices surge following major announcement",
    "AI breakthrough in natural language processing announced"
]

categories = [
    "technology", "science", "business", "technology", 
    "science", "business", "technology", "science", 
    "business", "technology"
]

# Create SVM pipeline with hyperparameter tuning
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('svm', SVC(probability=True))
])

# Hyperparameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
}

# Grid search with cross-validation
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(documents, categories)

print("Best SVM Parameters:")
print(grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.3f}")

# Test predictions
test_docs = [
    "New artificial intelligence algorithm improves smartphone performance",
    "Economic indicators suggest potential market volatility ahead",
    "Breakthrough in quantum physics leads to new discoveries"
]

for doc in test_docs:
    prediction = grid_search.predict([doc])[0]
    probabilities = grid_search.predict_proba([doc])[0]
    confidence = max(probabilities)
    
    print(f"Document: '{doc}'")
    print(f"Category: {prediction} (confidence: {confidence:.3f})")
    print()
```

**When to use SVM:**
- **High-dimensional text data** (lots of features)
- **Clear separation between categories**
- **When you need robust performance**
- **Medium-sized datasets** (not too large due to computational cost)

### 3. Logistic Regression: The Probability Calculator

**What it does:** Calculates the probability of each category using a linear combination of features.

**Why it's great:** Fast, interpretable, and provides probability estimates.

**Real-world analogy:** Like a loan officer who scores applications based on multiple factors and gives you a probability of approval.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment analysis example
reviews = [
    "This movie is absolutely amazing! Great acting and wonderful story.",
    "Terrible film. Boring plot and bad acting. Waste of time.",
    "Good movie, enjoyed watching it. Would recommend to friends.",
    "Awful experience. Poor quality and disappointing throughout.",
    "Excellent film! One of the best I've seen this year.",
    "Not great, but not terrible either. Average movie overall.",
    "Outstanding performance by all actors. Brilliant storytelling.",
    "Disappointing movie. Expected much better from this director.",
    "Really enjoyed this film. Great entertainment value.",
    "Poor script and weak character development. Skip this one."
]

sentiments = [
    "positive", "negative", "positive", "negative", "positive",
    "neutral", "positive", "negative", "positive", "negative"
]

# Create Logistic Regression pipeline
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=500)),
    ('lr', LogisticRegression(random_state=42))
])

# Train the model
lr_pipeline.fit(reviews, sentiments)

# Feature importance analysis
feature_names = lr_pipeline.named_steps['tfidf'].get_feature_names_out()
coefficients = lr_pipeline.named_steps['lr'].coef_

print("Most Important Features for Each Class:")
classes = lr_pipeline.named_steps['lr'].classes_
for i, class_name in enumerate(classes):
    # Get top positive and negative features for this class
    class_coef = coefficients[i]
    top_positive_indices = np.argsort(class_coef)[-5:][::-1]
    top_negative_indices = np.argsort(class_coef)[:5]
    
    print(f"\n{class_name.upper()} sentiment:")
    print("Most indicative words:")
    for idx in top_positive_indices:
        print(f"  {feature_names[idx]}: {class_coef[idx]:.3f}")
    
    print("Least indicative words:")
    for idx in top_negative_indices:
        print(f"  {feature_names[idx]}: {class_coef[idx]:.3f}")

# Test with new reviews
test_reviews = [
    "This movie exceeded all my expectations! Absolutely loved it!",
    "Mediocre film. Nothing special but watchable.",
    "Horrible movie. Complete waste of money and time."
]

print("\nPredictions for new reviews:")
for review in test_reviews:
    prediction = lr_pipeline.predict([review])[0]
    probabilities = lr_pipeline.predict_proba([review])[0]
    
    print(f"Review: '{review}'")
    print(f"Prediction: {prediction}")
    print("Probabilities:")
    for class_name, prob in zip(classes, probabilities):
        print(f"  {class_name}: {prob:.3f}")
    print()
```

**When to use Logistic Regression:**
- **When you need interpretable results**
- **Baseline model for comparison**
- **Fast training and prediction required**
- **Probabilistic outputs needed**

## 🔧 Building a Complete Classification System

Let's build a comprehensive text classifier that compares all three algorithms:

```python
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

class TextClassificationComparer:
    """Compare multiple traditional ML algorithms for text classification"""
    
    def __init__(self):
        # Initialize all classifiers
        self.classifiers = {
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('nb', MultinomialNB())
            ]),
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('svm', SVC(probability=True, C=1, kernel='linear'))
            ]),
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('lr', LogisticRegression(random_state=42))
            ])
        }
        
        # Ensemble classifier
        self.ensemble = VotingClassifier([
            ('nb', self.classifiers['Naive Bayes']),
            ('svm', self.classifiers['SVM']),
            ('lr', self.classifiers['Logistic Regression'])
        ], voting='soft')
    
    def compare_algorithms(self, texts, labels, cv_folds=5):
        """Compare all algorithms using cross-validation"""
        results = {}
        
        print("Cross-Validation Results:")
        print("-" * 50)
        
        # Individual classifiers
        for name, classifier in self.classifiers.items():
            scores = cross_val_score(classifier, texts, labels, cv=cv_folds, scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
            print(f"{name}:")
            print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Ensemble
        ensemble_scores = cross_val_score(self.ensemble, texts, labels, cv=cv_folds, scoring='accuracy')
        results['Ensemble'] = {
            'mean_accuracy': ensemble_scores.mean(),
            'std_accuracy': ensemble_scores.std(),
            'scores': ensemble_scores
        }
        print(f"\nEnsemble (Voting):")
        print(f"  Accuracy: {ensemble_scores.mean():.3f} (+/- {ensemble_scores.std() * 2:.3f})")
        
        return results
    
    def train_best_model(self, texts, labels):
        """Train all models and return the best performing one"""
        results = self.compare_algorithms(texts, labels)
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['mean_accuracy'])
        
        if best_model_name == 'Ensemble':
            best_model = self.ensemble
        else:
            best_model = self.classifiers[best_model_name]
        
        # Train the best model
        best_model.fit(texts, labels)
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Expected Accuracy: {results[best_model_name]['mean_accuracy']:.3f}")
        
        return best_model, best_model_name

# Example: News Category Classification
news_articles = [
    "Stock market soars to record highs as investors show confidence",
    "New smartphone released with revolutionary camera technology",
    "Scientists discover potential cure for Alzheimer's disease",
    "Professional sports league announces new expansion team",
    "Federal Reserve adjusts interest rates to combat inflation",
    "Latest gaming console features advanced graphics capabilities",
    "Medical research breakthrough offers hope for cancer patients",
    "Championship game draws millions of viewers worldwide",
    "Cryptocurrency values fluctuate amid regulatory concerns",
    "Artificial intelligence startup raises $100 million in funding",
    "Clinical trial shows promising results for new drug treatment",
    "Olympic athlete breaks world record in swimming competition"
]

news_categories = [
    "business", "technology", "health", "sports",
    "business", "technology", "health", "sports",
    "business", "technology", "health", "sports"
]

# Use the comparer
comparer = TextClassificationComparer()
best_model, best_name = comparer.train_best_model(news_articles, news_categories)

# Test with new articles
test_articles = [
    "Breakthrough in quantum computing brings us closer to practical applications",
    "Major bank reports record quarterly profits despite economic uncertainty",
    "New treatment shows 90% success rate in clinical trials for diabetes",
    "Team wins championship after incredible comeback in final minutes"
]

print("\nPredictions for new articles:")
for article in test_articles:
    prediction = best_model.predict([article])[0]
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba([article])[0]
        confidence = max(probabilities)
        print(f"Article: '{article[:50]}...'")
        print(f"Category: {prediction} (confidence: {confidence:.3f})")
    else:
        print(f"Article: '{article[:50]}...'")
        print(f"Category: {prediction}")
    print()
```

## 📊 Understanding Model Performance

### Evaluation Metrics That Matter

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_evaluation(model, X_test, y_test, class_names):
    """Comprehensive evaluation of a text classification model"""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Classification report
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Error analysis
    print("\nError Analysis:")
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
    
    for idx in incorrect_indices[:5]:  # Show first 5 errors
        print(f"Text: '{X_test[idx][:100]}...'")
        print(f"True: {y_test[idx]}, Predicted: {y_pred[idx]}")
        if y_proba is not None:
            confidence = max(y_proba[idx])
            print(f"Confidence: {confidence:.3f}")
        print("-" * 50)

# Usage example with the news classifier
X_train, X_test, y_train, y_test = train_test_split(
    news_articles, news_categories, test_size=0.3, random_state=42
)

best_model.fit(X_train, y_train)
comprehensive_evaluation(best_model, X_test, y_test, ['business', 'health', 'sports', 'technology'])
```

## 🎯 Choosing the Right Algorithm

### Decision Matrix

| Algorithm | Speed | Interpretability | Memory | Performance | Best For |
|-----------|-------|------------------|---------|-------------|----------|
| **Naive Bayes** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Spam detection, sentiment analysis |
| **SVM** | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Document classification, high accuracy needs |
| **Logistic Regression** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Baseline models, feature analysis |

### Quick Selection Guide

```python
def choose_algorithm(dataset_size, interpretability_needed, accuracy_priority, speed_priority):
    """Help choose the best algorithm based on requirements"""
    
    score = {
        'Naive Bayes': 0,
        'SVM': 0,
        'Logistic Regression': 0
    }
    
    # Dataset size consideration
    if dataset_size < 1000:
        score['Naive Bayes'] += 2
        score['Logistic Regression'] += 2
    elif dataset_size < 10000:
        score['SVM'] += 2
        score['Logistic Regression'] += 1
    else:
        score['Naive Bayes'] += 1
        score['Logistic Regression'] += 1
    
    # Interpretability
    if interpretability_needed:
        score['Logistic Regression'] += 3
        score['Naive Bayes'] += 2
    
    # Accuracy priority
    if accuracy_priority:
        score['SVM'] += 3
        score['Logistic Regression'] += 2
    
    # Speed priority
    if speed_priority:
        score['Naive Bayes'] += 3
        score['Logistic Regression'] += 2
    
    best_algorithm = max(score.keys(), key=lambda x: score[x])
    
    print("Algorithm Recommendation:")
    print(f"Best choice: {best_algorithm}")
    print("\nScores:")
    for algo, s in score.items():
        print(f"  {algo}: {s}")
    
    return best_algorithm

# Example usage
recommended = choose_algorithm(
    dataset_size=5000,
    interpretability_needed=True,
    accuracy_priority=False,
    speed_priority=True
)
```

## 💡 Pro Tips for Better Performance

### 1. Feature Engineering Matters
```python
# Advanced feature engineering for better performance
def create_advanced_features(texts):
    """Create additional features beyond basic TF-IDF"""
    
    features = []
    for text in texts:
        text_features = {
            'length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_count': sum(1 for c in text if c.isdigit()),
        }
        features.append(text_features)
    
    return pd.DataFrame(features)

# Combine with TF-IDF
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def create_hybrid_pipeline():
    """Combine TF-IDF with custom features"""
    
    class TextFeatureExtractor:
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return create_advanced_features(X)
    
    # This would require more complex pipeline setup
    # See full implementation in exercises
    pass
```

### 2. Hyperparameter Optimization
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def optimize_hyperparameters(X, y):
    """Find best hyperparameters using randomized search"""
    
    # Parameter distributions for random search
    param_distributions = {
        'tfidf__max_features': [500, 1000, 2000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.7, 0.8, 0.9, 1.0],
        'classifier__C': uniform(0.1, 10),  # For SVM and LogReg
        'classifier__alpha': uniform(0.01, 1)  # For Naive Bayes
    }
    
    # Random search is faster than grid search
    random_search = RandomizedSearchCV(
        estimator=Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression())
        ]),
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    print("Best parameters:")
    print(random_search.best_params_)
    print(f"Best score: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_
```

## 🏋️ Practice Exercise

**Build a News Category Classifier**

Your task: Create a classifier that can categorize news articles into Technology, Business, Sports, and Health categories.

```python
# Starter code for your exercise
def build_news_classifier():
    """
    Build and evaluate a news classification system
    
    Requirements:
    1. Compare all three algorithms (NB, SVM, LogReg)
    2. Use proper cross-validation
    3. Implement error analysis
    4. Create feature importance analysis
    5. Build an ensemble model
    
    Bonus:
    - Add custom features (length, punctuation, etc.)
    - Implement confidence thresholding
    - Create a web demo with Streamlit
    """
    
    # Your implementation here
    pass

# Test data for your classifier
test_news = [
    "Apple announces new iPhone with advanced AI capabilities and improved battery life",
    "Stock market volatility continues as investors react to Federal Reserve policy changes",
    "Olympic champion sets new world record in 100-meter sprint at international competition",
    "Scientists develop new gene therapy treatment showing promise in cancer clinical trials",
    "Cryptocurrency startup raises $50 million to expand blockchain payment platform",
    "Professional basketball league announces new collective bargaining agreement with players",
    "Medical breakthrough offers hope for patients with rare autoimmune disorders",
    "Tech giant acquires AI startup for $2 billion to enhance cloud computing services"
]

# Your classifier should predict: 
# technology, business, sports, health, technology, sports, health, technology
```

## 💡 Key Takeaways

1. **Start with traditional ML** - Often performs as well as complex models
2. **Naive Bayes is your friend** - Excellent baseline for text classification
3. **SVM for accuracy** - When you need the best possible performance
4. **Logistic Regression for insights** - When you need to understand what the model learned
5. **Ensemble for robustness** - Combine multiple models for better results
6. **Feature engineering matters** - Good features often beat fancy algorithms

## 🚀 What's Next?

You've mastered traditional ML for text! Next, you'll learn about [Neural Networks for Text Classification](./02_neural_text_classification.md) and discover when deep learning outperforms traditional methods.

**Coming up:**
- Convolutional Neural Networks (CNNs) for text
- Recurrent Neural Networks (RNNs) and LSTMs
- When to choose neural networks over traditional ML
- Building hybrid systems that combine both approaches

Ready to dive into the neural network world? Let's continue the journey!
