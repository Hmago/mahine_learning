# N-grams & Feature Selection: Advanced Text Features

## 🎯 What You'll Learn

You'll discover how to capture phrase meanings with N-grams and select the most valuable features from potentially millions of possibilities. This is where text preprocessing becomes truly powerful!

## 🔗 What are N-grams?

### The Problem with Single Words

Think about these phrases:
- "not good" (negative sentiment)
- "very good" (positive sentiment)  
- "not bad" (positive sentiment)

If we only look at individual words, we lose the crucial relationships between them!

### N-grams: Capturing Word Sequences

N-grams are simply sequences of N consecutive words:

- **Unigrams (1-gram)**: Individual words → "good", "not", "very"
- **Bigrams (2-gram)**: Word pairs → "not good", "very good", "not bad"  
- **Trigrams (3-gram)**: Word triplets → "not very good", "really not bad"

**Real-world analogy:** It's like reading a book word by word vs. reading phrases and sentences!

## 🛠 Implementing N-grams

### Manual N-gram Generation

```python
def generate_ngrams(text, n):
    """Generate n-grams from text"""
    words = text.split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# Example
text = "The quick brown fox jumps over the lazy dog"

# Generate different n-grams
unigrams = generate_ngrams(text, 1)
bigrams = generate_ngrams(text, 2)
trigrams = generate_ngrams(text, 3)

print("Original text:", text)
print("Unigrams:", unigrams)
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)
```

### N-grams with Scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "The movie was not good",
    "The movie was very good", 
    "The film was not bad",
    "This film was really good"
]

# Different n-gram configurations
vectorizer_configs = {
    "Unigrams only": (1, 1),
    "Bigrams only": (2, 2), 
    "Trigrams only": (3, 3),
    "Unigrams + Bigrams": (1, 2),
    "All (1-3 grams)": (1, 3)
}

for name, ngram_range in vectorizer_configs.items():
    print(f"\n{name} (ngram_range={ngram_range}):")
    print("-" * 50)
    
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=True)
    matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Sample features: {feature_names[:10]}")
    
    # Show how first document is represented
    first_doc_features = matrix[0].toarray()[0]
    non_zero_indices = first_doc_features.nonzero()[0]
    
    print(f"\nFirst document: '{documents[0]}'")
    print("Non-zero features:")
    for idx in non_zero_indices:
        print(f"  {feature_names[idx]}: {first_doc_features[idx]:.3f}")
```

### Character N-grams

Sometimes character-level n-grams are useful for detecting languages, handling misspellings, or analyzing short texts:

```python
def generate_char_ngrams(text, n):
    """Generate character-level n-grams"""
    char_ngrams = []
    text = text.replace(' ', '_')  # Mark word boundaries
    
    for i in range(len(text) - n + 1):
        char_ngram = text[i:i+n]
        char_ngrams.append(char_ngram)
    
    return char_ngrams

# Example
text = "hello world"
char_trigrams = generate_char_ngrams(text, 3)
print(f"Text: '{text}'")
print(f"Character trigrams: {char_trigrams}")

# Using scikit-learn for character n-grams
vectorizer = TfidfVectorizer(
    analyzer='char',  # Character-level analysis
    ngram_range=(2, 4),  # 2-4 character n-grams
    lowercase=True
)

texts = ["hello", "helo", "hallo", "hello world"]
matrix = vectorizer.fit_transform(texts)
features = vectorizer.get_feature_names_out()

print(f"\nCharacter n-gram features: {features[:20]}")
```

## 🎯 The Curse of Dimensionality

### The Problem

```python
# Let's see how vocabulary grows with n-grams
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers", 
    "Natural language processing helps computers understand text",
    "Computer vision enables machines to interpret visual information",
    "Data science combines statistics programming and domain expertise"
]

for max_n in [1, 2, 3, 4]:
    vectorizer = TfidfVectorizer(ngram_range=(1, max_n), lowercase=True)
    matrix = vectorizer.fit_transform(documents)
    vocab_size = len(vectorizer.get_feature_names_out())
    
    print(f"N-grams up to {max_n}: {vocab_size} features")

# Output shows exponential growth!
# N-grams up to 1: 25 features
# N-grams up to 2: 45 features  
# N-grams up to 3: 60 features
# N-grams up to 4: 70 features
```

With real datasets, you can easily get millions of features! We need feature selection.

## ✂️ Feature Selection: Keeping the Best

### 1. Frequency-Based Selection

**Idea:** Keep only features that appear frequently enough to be meaningful.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample data
documents = [
    "I love this movie it is amazing",
    "This movie is terrible I hate it",
    "Amazing film great acting love it", 
    "Terrible acting bad movie hate it",
    "Love great amazing fantastic wonderful",
    "Hate terrible bad awful horrible"
]

# Before feature selection
vectorizer_all = TfidfVectorizer(ngram_range=(1, 2))
matrix_all = vectorizer_all.fit_transform(documents)
print(f"Before selection: {matrix_all.shape[1]} features")

# With minimum document frequency
vectorizer_filtered = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,  # Must appear in at least 2 documents
    max_df=0.8  # Must appear in less than 80% of documents
)
matrix_filtered = vectorizer_filtered.fit_transform(documents)
print(f"After frequency filtering: {matrix_filtered.shape[1]} features")

print("\nRemoved features (too rare or too common):")
all_features = set(vectorizer_all.get_feature_names_out())
kept_features = set(vectorizer_filtered.get_feature_names_out())
removed_features = all_features - kept_features
print(list(removed_features)[:10])
```

### 2. Statistical Feature Selection

**Idea:** Use statistical tests to find features most associated with target classes.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np

# Sample data with labels
documents = [
    "I love this movie it is amazing",      # Positive
    "This movie is terrible I hate it",     # Negative  
    "Amazing film great acting love it",    # Positive
    "Terrible acting bad movie hate it",    # Negative
    "Love great amazing fantastic",         # Positive
    "Hate terrible bad awful"               # Negative
]

labels = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

# Create TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"Original features: {X.shape[1]}")

# Use Chi-square test to select best features
selector = SelectKBest(chi2, k=10)  # Select top 10 features
X_selected = selector.fit_transform(X, labels)

print(f"Selected features: {X_selected.shape[1]}")

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]
print(f"Selected features: {selected_features}")

# Get feature scores
feature_scores = selector.scores_
print("\nFeature importance scores:")
for idx in selected_indices:
    print(f"{feature_names[idx]}: {feature_scores[idx]:.3f}")
```

### 3. Variance-Based Selection

**Idea:** Remove features with very low variance (they don't vary much across documents).

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer

# Create features
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)

print(f"Before variance filtering: {X.shape[1]} features")

# Remove low-variance features
selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
X_selected = selector.fit_transform(X.toarray())

print(f"After variance filtering: {X_selected.shape[1]} features")

# See which features were removed
feature_names = vectorizer.get_feature_names_out()
selected_mask = selector.get_support()
removed_features = [feature_names[i] for i, selected in enumerate(selected_mask) if not selected]
print(f"Removed {len(removed_features)} low-variance features")
```

## 🔧 Building a Complete Feature Engineering Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import numpy as np

class AdvancedTextFeatureExtractor:
    """Complete text feature extraction with n-grams and selection"""
    
    def __init__(self, 
                 ngram_range=(1, 2),
                 max_features=10000,
                 min_df=2,
                 max_df=0.8,
                 use_tfidf=True,
                 feature_selection_method='chi2',
                 k_best=5000):
        
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_tfidf = use_tfidf
        self.feature_selection_method = feature_selection_method
        self.k_best = k_best
        
        # Initialize components
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                lowercase=True,
                stop_words='english'
            )
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                lowercase=True,
                stop_words='english'
            )
        
        if feature_selection_method == 'chi2':
            self.selector = SelectKBest(chi2, k=k_best)
        else:
            self.selector = None
    
    def fit(self, documents, labels=None):
        """Fit the feature extractor"""
        # Create basic features
        X = self.vectorizer.fit_transform(documents)
        
        print(f"After vectorization: {X.shape[1]} features")
        
        # Apply feature selection if we have labels
        if labels is not None and self.selector is not None:
            X_selected = self.selector.fit_transform(X, labels)
            print(f"After feature selection: {X_selected.shape[1]} features")
        
        return self
    
    def transform(self, documents):
        """Transform documents to features"""
        X = self.vectorizer.transform(documents)
        
        if self.selector is not None:
            X = self.selector.transform(X)
        
        return X
    
    def fit_transform(self, documents, labels=None):
        """Fit and transform in one step"""
        return self.fit(documents, labels).transform(documents)
    
    def get_feature_names(self):
        """Get names of selected features"""
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.selector is not None:
            selected_indices = self.selector.get_support(indices=True)
            return [feature_names[i] for i in selected_indices]
        
        return feature_names
    
    def analyze_features(self, documents, labels=None):
        """Analyze the extracted features"""
        X = self.fit_transform(documents, labels)
        feature_names = self.get_feature_names()
        
        print(f"\nFeature Analysis:")
        print(f"Documents: {len(documents)}")
        print(f"Features: {len(feature_names)}")
        print(f"Matrix shape: {X.shape}")
        print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")
        
        # Show most important features if we have labels
        if labels is not None and self.selector is not None:
            print(f"\nTop 10 most discriminative features:")
            feature_scores = self.selector.scores_
            selected_indices = self.selector.get_support(indices=True)
            
            # Get scores for selected features
            scores_with_names = [(feature_names[i], feature_scores[selected_indices[i]]) 
                               for i in range(len(feature_names))]
            scores_with_names.sort(key=lambda x: x[1], reverse=True)
            
            for name, score in scores_with_names[:10]:
                print(f"  {name}: {score:.3f}")

# Example usage
documents = [
    "I absolutely love this amazing movie! Great acting and wonderful story.",
    "This movie is terrible! Awful acting and boring plot. I hate it.",
    "Amazing film with fantastic performances. Love every minute of it.",
    "Terrible movie with bad acting. Complete waste of time. Hate it.",
    "Great movie! Love the story and characters. Highly recommend.",
    "Bad film with terrible plot. Don't waste your time watching this."
]

labels = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

# Create feature extractor
extractor = AdvancedTextFeatureExtractor(
    ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
    max_features=1000,
    min_df=1,
    k_best=20
)

# Extract features and analyze
X = extractor.fit_transform(documents, labels)
extractor.analyze_features(documents, labels)
```

## 🎯 Choosing the Right N-gram Strategy

### For Sentiment Analysis

```python
# Capture negations and intensifiers
sentiment_extractor = AdvancedTextFeatureExtractor(
    ngram_range=(1, 3),  # "not good", "very bad", "not very good"
    max_features=5000,
    min_df=1,  # Keep rare emotional expressions
    k_best=1000
)
```

### For Topic Classification

```python
# Focus on content words and phrases
topic_extractor = AdvancedTextFeatureExtractor(
    ngram_range=(1, 2),  # Mostly unigrams and bigrams
    max_features=10000,
    min_df=3,  # Remove very rare terms
    max_df=0.7,  # Remove very common terms
    k_best=2000
)
```

### For Language Detection

```python
# Character n-grams work better
from sklearn.feature_extraction.text import TfidfVectorizer

language_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),  # Character 2-4 grams
    max_features=3000
)
```

## 🏋️ Practice Exercise

Build feature extractors for these different scenarios:

```python
# Exercise 1: Restaurant Reviews
restaurant_reviews = [
    "Amazing food! Great service and lovely atmosphere. Highly recommend!",
    "Terrible food and slow service. Would not go back.",
    "Good food but overpriced. Service was okay.",
    "Excellent restaurant! Best Italian food in town. Must visit!",
    "Average food, nothing special. Service was friendly though."
]
review_labels = [1, 0, 0, 1, 0]  # 1=Positive, 0=Negative

# Exercise 2: News Categories
news_articles = [
    "The stock market reached new highs today with technology companies leading gains",
    "Scientists discover new species of deep-sea fish in Pacific Ocean",
    "Local football team wins championship in thrilling overtime victory", 
    "New study shows promising results for cancer treatment using AI",
    "Federal Reserve announces interest rate changes affecting economy"
]
news_labels = ["business", "science", "sports", "science", "business"]

# Exercise 3: Product Descriptions
products = [
    "Wireless bluetooth headphones with noise cancellation and 20-hour battery life",
    "Organic cotton t-shirt available in multiple colors and sizes",
    "Smart home security camera with night vision and mobile app",
    "Natural skincare moisturizer with vitamin C and hyaluronic acid",
    "Gaming laptop with high-performance graphics and RGB keyboard"
]
product_categories = ["electronics", "clothing", "electronics", "beauty", "electronics"]

# Your tasks:
# 1. Design appropriate n-gram strategies for each scenario
# 2. Choose suitable feature selection methods
# 3. Analyze the most important features for each task
```

## 💡 Key Takeaways

1. **N-grams capture context** - Essential for understanding phrases and negations
2. **Higher N increases vocabulary exponentially** - Balance between coverage and complexity
3. **Feature selection is crucial** - Prevents overfitting and improves performance
4. **Domain matters** - Different tasks need different n-gram strategies
5. **Character n-grams help with robustness** - Good for handling misspellings and unknown words
6. **Frequency filtering is simple but effective** - Often works as well as complex methods

## 🔗 Quick Reference

```python
# Quick n-gram configurations for common tasks

# Sentiment analysis (capture negations)
sentiment_config = {
    'ngram_range': (1, 3),
    'min_df': 1,
    'max_df': 0.9,
    'max_features': 5000
}

# Document classification (focus on content)
classification_config = {
    'ngram_range': (1, 2), 
    'min_df': 3,
    'max_df': 0.7,
    'max_features': 10000
}

# Short text (social media, SMS)
short_text_config = {
    'ngram_range': (1, 2),
    'min_df': 1,
    'max_df': 0.8,
    'max_features': 3000
}

# Robust text (handle misspellings)
robust_config = {
    'analyzer': 'char',
    'ngram_range': (2, 4),
    'max_features': 5000
}
```

## 🎯 What's Next?

Congratulations! You've completed the text preprocessing fundamentals. You now know how to:

- Clean and normalize messy text data
- Tokenize text appropriately for different domains
- Convert text to numerical features with BoW and TF-IDF
- Use n-grams to capture phrase meanings
- Select the most valuable features from large vocabularies

Next, you'll learn about **Text Classification & Sentiment Analysis** where you'll use these preprocessing skills to build real classification systems!

Continue to [Text Classification](../02_text_classification/README.md) to start building your first NLP models!
