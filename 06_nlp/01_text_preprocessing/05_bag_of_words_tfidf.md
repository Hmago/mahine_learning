# Bag of Words & TF-IDF: Converting Text to Numbers

## 🎯 What You'll Learn

Now comes the magic moment - converting clean, normalized text into numbers that machines can understand! You'll learn two fundamental techniques that transform words into meaningful numerical features.

## 🔢 Why Convert Text to Numbers?

Imagine you're a detective analyzing witness statements, but you can only work with numbers, not words. How would you convert "The suspect was tall and wore a red hat" into numbers that still preserve the meaning?

That's exactly what we need to do for machine learning - computers think in numbers, not words!

## 🛍 Bag of Words: The Shopping List Approach

### The Basic Idea

Think of Bag of Words (BoW) like making a shopping list for a recipe. You list all possible ingredients (words) and count how many times each appears in your recipe (document).

**Example:**

```python
# Two recipe descriptions
recipe1 = "add sugar add flour add eggs"
recipe2 = "add flour add salt add pepper"

# Our "vocabulary" (all unique words)
vocabulary = ["add", "sugar", "flour", "eggs", "salt", "pepper"]

# Convert to numbers (count of each word)
recipe1_vector = [3, 1, 1, 1, 0, 0]  # 3 "add", 1 "sugar", 1 "flour", 1 "eggs", 0 "salt", 0 "pepper"
recipe2_vector = [3, 0, 1, 0, 1, 1]  # 3 "add", 0 "sugar", 1 "flour", 0 "eggs", 1 "salt", 1 "pepper"
```

### Implementing Bag of Words from Scratch

```python
from collections import Counter
import numpy as np

class SimpleBagOfWords:
    """A simple implementation of Bag of Words"""
    
    def __init__(self):
        self.vocabulary = []
        self.word_to_index = {}
    
    def fit(self, documents):
        """Learn the vocabulary from documents"""
        # Collect all unique words
        all_words = set()
        for doc in documents:
            words = doc.lower().split()  # Simple tokenization
            all_words.update(words)
        
        # Create vocabulary
        self.vocabulary = sorted(list(all_words))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Vocabulary: {self.vocabulary}")
    
    def transform(self, documents):
        """Convert documents to numerical vectors"""
        vectors = []
        
        for doc in documents:
            # Count words in this document
            word_counts = Counter(doc.lower().split())
            
            # Create vector for this document
            vector = np.zeros(len(self.vocabulary))
            for word, count in word_counts.items():
                if word in self.word_to_index:
                    vector[self.word_to_index[word]] = count
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)

# Example usage
documents = [
    "I love this movie",
    "This movie is great", 
    "I hate this film",
    "Great movie I love it"
]

bow = SimpleBagOfWords()
vectors = bow.fit_transform(documents)

print("\nDocument vectors:")
for i, (doc, vector) in enumerate(zip(documents, vectors)):
    print(f"Doc {i+1}: '{doc}'")
    print(f"Vector: {vector}")
    print()
```

### Using Scikit-learn's CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create sample documents
documents = [
    "The cat sat on the mat",
    "The dog ran in the park", 
    "Cats and dogs are pets",
    "I love my pet cat"
]

# Create and fit the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Get feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert to a readable format
df = pd.DataFrame(X.toarray(), columns=feature_names)
print("Bag of Words Matrix:")
print(df)

print("\nWhat each document looks like:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: '{doc}'")
    print(f"Vector: {df.iloc[i].values}")
    print()
```

## 📊 Problems with Basic Bag of Words

### Problem 1: Common Words Dominate

```python
documents = [
    "The cat is sleeping",
    "The dog is running", 
    "The bird is flying"
]

# "the" and "is" appear in every document but don't tell us much!
# They might overshadow more meaningful words like "cat", "dog", "bird"
```

### Problem 2: No Context Understanding

```python
doc1 = "The movie was not bad"      # Positive sentiment
doc2 = "The movie was not good"     # Negative sentiment

# Bag of Words sees these as very similar (both have "movie", "was", "not")
# But they have opposite meanings!
```

## 🎯 TF-IDF: Smarter Word Weighting

TF-IDF (Term Frequency-Inverse Document Frequency) solves the "common words" problem by making rare words more important and common words less important.

### Understanding TF-IDF

**Real-world analogy:** Imagine you're a news editor. If every article mentions "the" and "is", those words don't help you categorize articles. But if only one article mentions "quantum physics", that's a very important signal!

### The Formula (Don't worry, we'll break it down!)

```
TF-IDF = Term Frequency × Inverse Document Frequency

Where:
- Term Frequency (TF) = How often a word appears in THIS document
- Inverse Document Frequency (IDF) = How rare the word is across ALL documents
```

### Step-by-Step TF-IDF Calculation

```python
import math
import numpy as np
from collections import Counter

def calculate_tf(doc_words):
    """Calculate term frequency for a document"""
    word_count = len(doc_words)
    tf_dict = {}
    word_counter = Counter(doc_words)
    
    for word, count in word_counter.items():
        tf_dict[word] = count / word_count
    
    return tf_dict

def calculate_idf(documents):
    """Calculate inverse document frequency for all words"""
    N = len(documents)
    idf_dict = {}
    all_words = set(word for doc in documents for word in doc)
    
    for word in all_words:
        containing_docs = sum(1 for doc in documents if word in doc)
        idf_dict[word] = math.log(N / containing_docs)
    
    return idf_dict

def calculate_tfidf(documents):
    """Calculate TF-IDF for all documents"""
    
    # Calculate IDF for all words
    idf = calculate_idf(documents)
    
    # Calculate TF-IDF for each document
    tfidf_documents = []
    
    for doc in documents:
        tf = calculate_tf(doc)
        tfidf_doc = {}
        
        for word, tf_val in tf.items():
            tfidf_doc[word] = tf_val * idf[word]
        
        tfidf_documents.append(tfidf_doc)
    
    return tfidf_documents, idf

# Example
documents = [
    ["cat", "sat", "mat"],
    ["dog", "ran", "park"], 
    ["cat", "dog", "pets"],
    ["love", "pet", "cat"]
]

tfidf_docs, idf_values = calculate_tfidf(documents)

print("IDF values (higher = rarer word):")
for word, idf_val in sorted(idf_values.items()):
    print(f"{word}: {idf_val:.3f}")

print("\nTF-IDF for first document ['cat', 'sat', 'mat']:")
for word, tfidf_val in tfidf_docs[0].items():
    print(f"{word}: {tfidf_val:.3f}")
```

### Using Scikit-learn's TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "The cat sat on the mat",
    "The dog ran in the park", 
    "Cats and dogs are pets",
    "I love my pet cat",
    "The park has many trees"
]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert to DataFrame for better visualization
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("TF-IDF Matrix:")
print(df_tfidf.round(3))

print("\nMost important words per document:")
for i, doc in enumerate(documents):
    doc_scores = df_tfidf.iloc[i]
    top_words = doc_scores.nlargest(3)
    print(f"Doc {i+1}: '{doc}'")
    print(f"Top words: {dict(top_words)}")
    print()
```

## 🔧 Practical Comparison: BoW vs TF-IDF

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# Example documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "Quick brown foxes are very fast animals", 
    "Dogs and foxes are different animals"
]

# Bag of Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Compare the results
feature_names = bow_vectorizer.get_feature_names_out()

print("Comparison of BoW vs TF-IDF for first document:")
print(f"Document: '{documents[0]}'")
print()

bow_values = bow_matrix[0].toarray()[0]
tfidf_values = tfidf_matrix[0].toarray()[0]

# Show top words by both methods
print("Word\t\tBoW\tTF-IDF")
print("-" * 30)
for word, bow_val, tfidf_val in zip(feature_names, bow_values, tfidf_values):
    if bow_val > 0:  # Only show words that appear in the document
        print(f"{word:12}\t{bow_val}\t{tfidf_val:.3f}")
```

## 🎛 Advanced TF-IDF Configuration

### Controlling Vocabulary Size

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Various ways to control vocabulary
vectorizer = TfidfVectorizer(
    max_features=1000,        # Keep only top 1000 features
    min_df=2,                 # Word must appear in at least 2 documents
    max_df=0.8,               # Word must appear in less than 80% of documents
    stop_words='english',     # Remove common English stop words
    ngram_range=(1, 2)        # Include both unigrams and bigrams
)

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand text",
    "Computer vision enables machines to interpret visual information"
]

tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Sample features: {feature_names[:10]}")
```

### Handling Different Text Types

```python
def create_vectorizer_for_task(task_type):
    """Create appropriate vectorizer for different tasks"""
    
    if task_type == "short_text":  # SMS, tweets
        return TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,  # Keep rare words for short texts
            max_df=0.9
        )
    
    elif task_type == "long_documents":  # Articles, papers
        return TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 3),
            min_df=5,  # Remove very rare words
            max_df=0.7
        )
    
    elif task_type == "sentiment":  # Reviews, opinions
        return TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Keep words like "not", "very"
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
    
    else:  # Default
        return TfidfVectorizer()

# Example usage
short_texts = ["Love this!", "Hate it", "Best ever", "Worst movie"]
vectorizer = create_vectorizer_for_task("short_text")
matrix = vectorizer.fit_transform(short_texts)
print(f"Features for short text: {vectorizer.get_feature_names_out()}")
```

## 🏋️ Hands-On Exercise

Try building TF-IDF vectors for these different scenarios:

```python
# Exercise 1: Product Reviews
reviews = [
    "This product is amazing! Great quality and fast shipping.",
    "Terrible quality. Broke after one day. Don't buy!",
    "Good value for money. Decent quality, nothing special.",
    "Outstanding! Exceeded my expectations. Highly recommend!"
]

# Exercise 2: News Headlines
headlines = [
    "Stock market reaches new record high amid economic growth",
    "Scientists discover new species in deep ocean exploration", 
    "Technology companies report strong quarterly earnings",
    "Climate change impacts global agriculture production"
]

# Exercise 3: Social Media Posts
posts = [
    "Just watched the best movie ever! #cinema #amazing",
    "Coffee and coding all night. #developer #life",
    "Beautiful sunset at the beach today #nature #photography",
    "New restaurant in town has incredible food! #foodie"
]

# Your task: 
# 1. Choose appropriate TF-IDF settings for each type
# 2. Convert to vectors and analyze the results
# 3. Identify the most important words for each document
```

## 💡 When to Use What?

### Use Bag of Words when:

- **Simple baseline needed**: Quick and interpretable
- **Small vocabulary**: Limited unique words
- **Document classification**: Categories are clearly distinct
- **Computational resources limited**: Faster than TF-IDF

### Use TF-IDF when:

- **Large vocabulary**: Many unique words across documents
- **Variable document lengths**: Some very long, some very short
- **Information retrieval**: Finding relevant documents
- **Better accuracy needed**: Usually outperforms BoW

### Consider alternatives when:

- **Semantic meaning matters**: Word embeddings (Word2Vec, BERT)
- **Order matters**: N-grams, sequential models
- **Context is crucial**: Transformer models

## 🔍 Analyzing Your Results

```python
def analyze_tfidf_results(documents, vectorizer, matrix):
    """Analyze and visualize TF-IDF results"""
    
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Matrix shape: {matrix.shape}")
    print()
    
    # Most important words overall
    mean_scores = np.mean(matrix.toarray(), axis=0)
    top_indices = np.argsort(mean_scores)[-10:][::-1]
    
    print("Top 10 most important words overall:")
    for idx in top_indices:
        print(f"{feature_names[idx]}: {mean_scores[idx]:.3f}")
    print()
    
    # Most important words per document
    print("Most important words per document:")
    for i, doc in enumerate(documents):
        doc_scores = matrix[i].toarray()[0]
        top_indices = np.argsort(doc_scores)[-5:][::-1]
        
        print(f"Doc {i+1}: '{doc[:50]}...'")
        for idx in top_indices:
            if doc_scores[idx] > 0:
                print(f"  {feature_names[idx]}: {doc_scores[idx]:.3f}")
        print()

# Example usage
documents = [
    "Machine learning algorithms can process large datasets efficiently",
    "Deep neural networks require significant computational resources",  
    "Natural language processing enables human-computer interaction",
    "Computer vision systems analyze images and video content"
]

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(documents)

analyze_tfidf_results(documents, vectorizer, matrix)
```

## 💡 Key Takeaways

1. **Bag of Words is simple but effective** - Great starting point for text analysis
2. **TF-IDF usually performs better** - Handles common words more intelligently  
3. **Configure for your domain** - Different text types need different settings
4. **Vocabulary size matters** - Balance between coverage and computational efficiency
5. **Preprocessing affects results** - Good cleaning and normalization improve feature quality
6. **N-grams capture phrases** - Sometimes "not good" is more meaningful than separate "not" and "good"

## 🔗 Quick Reference

```python
# Quick TF-IDF setup for common scenarios

# General purpose
general_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english', 
    max_features=10000,
    ngram_range=(1, 2)
)

# Short text (social media, reviews)
short_text_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1
)

# Long documents (articles, papers)
long_doc_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=15000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.7
)
```

Ready to explore more advanced feature engineering? Continue to [N-grams & Feature Selection](./06_ngrams_feature_selection.md)!
