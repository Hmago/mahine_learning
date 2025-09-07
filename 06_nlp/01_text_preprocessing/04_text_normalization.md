# Text Normalization: Making Words Consistent

## 🎯 What You'll Learn

Text normalization is like having a universal translator that makes all variations of words look the same. You'll learn to handle different word forms, spellings, and formats so your models can understand that "running," "runs," and "ran" are all related to "run."

## 🔄 What is Text Normalization?

Imagine you're organizing a library, and you find these books:
- "The Art of Cooking"
- "The art of cooking" 
- "THE ART OF COOKING"
- "The Art of Cookin'" (missing 'g')

To a computer, these look like four completely different titles! Text normalization helps us recognize they're all the same book.

## 🛠 Types of Text Normalization

### 1. Case Normalization: Handling Upper and Lower Case

**The Problem:**
```python
words = ["Apple", "apple", "APPLE", "aPpLe"]
unique_words = set(words)
print(unique_words)
# Output: {'Apple', 'apple', 'APPLE', 'aPpLe'} - 4 different "words"!
```

**The Solution:**
```python
normalized_words = [word.lower() for word in words]
unique_normalized = set(normalized_words)
print(unique_normalized)
# Output: {'apple'} - Now it's just 1 word!
```

**When to be careful:**
```python
# Some cases where you might NOT want to lowercase everything:
proper_nouns = ["New York", "Google", "iPhone"]
acronyms = ["NASA", "FBI", "CEO"]
brand_names = ["McDonald's", "iPhone", "YouTube"]

# For these, consider keeping original case or handling separately
```

### 2. Stemming: Reducing Words to Their Root Form

**What it does:** Removes suffixes to get the word "stem"

```python
from nltk.stem import PorterStemmer

# Download required data
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()

words = ["running", "runs", "ran", "runner", "easily", "fairly"]
stems = [stemmer.stem(word) for word in words]

for word, stem in zip(words, stems):
    print(f"{word} → {stem}")

# Output:
# running → run
# runs → run  
# ran → ran (doesn't change - stemming isn't perfect!)
# runner → runner (doesn't change)
# easily → easili (not a real word!)
# fairly → fairli (not a real word!)
```

**Real-world analogy:** Stemming is like a quick haircut - fast but sometimes a bit rough around the edges!

### 3. Lemmatization: Finding the True Root Word

**What it does:** Finds the actual dictionary form (lemma) of a word

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

words = ["running", "runs", "ran", "runner", "easily", "fairly", "better", "geese"]

print("Word → Stem → Lemma")
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    print(f"{word} → {stem} → {lemma}")

# Output:
# running → run → run
# runs → run → run
# ran → ran → run
# runner → runner → runner
# easily → easili → easily
# fairly → fairli → fairly
# better → better → good
# geese → gees → goose
```

**Real-world analogy:** Lemmatization is like a professional stylist - takes more time but gives much better results!

### 4. Spell Correction: Fixing Typos and Mistakes

**The Problem:**
```python
misspelled_text = "Thsi is a sentance with mny speling erors."
```

**Simple Solution with TextBlob:**
```python
from textblob import TextBlob

# Note: Install with: pip install textblob
blob = TextBlob(misspelled_text)
corrected = blob.correct()
print(f"Original: {misspelled_text}")
print(f"Corrected: {corrected}")
# Output: "This is a sentence with many spelling errors."
```

**Advanced Solution with pyspellchecker:**
```python
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spelling(text):
    """Correct spelling in text while preserving structure"""
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Remove punctuation for checking, but remember it
        clean_word = word.strip('.,!?;:"()[]')
        punctuation = word[len(clean_word):]
        
        # Check if word is misspelled
        if clean_word.lower() not in spell:
            # Get the most likely correction
            corrected = spell.correction(clean_word.lower())
            if corrected:
                corrected_words.append(corrected + punctuation)
            else:
                corrected_words.append(word)  # Keep original if no correction found
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

text = "Thsi is a sentance with mny speling erors."
corrected = correct_spelling(text)
print(f"Original: {text}")
print(f"Corrected: {corrected}")
```

## 🔧 Building a Comprehensive Normalizer

Let's create a flexible text normalizer that combines all these techniques:

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
import re

class TextNormalizer:
    """A comprehensive text normalizer with multiple options"""
    
    def __init__(self):
        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        required_downloads = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']
        for item in required_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
            except LookupError:
                nltk.download(item)
    
    def normalize_text(self, text,
                      lowercase=True,
                      correct_spelling=False,
                      remove_numbers=False,
                      stemming=False,
                      lemmatization=True,
                      remove_stopwords=False,
                      min_word_length=2):
        """
        Comprehensive text normalization
        
        Args:
            text (str): Input text to normalize
            lowercase (bool): Convert to lowercase
            correct_spelling (bool): Fix spelling mistakes
            remove_numbers (bool): Remove numeric tokens
            stemming (bool): Apply stemming (note: conflicts with lemmatization)
            lemmatization (bool): Apply lemmatization
            remove_stopwords (bool): Remove common stopwords
            min_word_length (int): Minimum word length to keep
        """
        
        print(f"Original: {text}")
        
        # Correct spelling first if requested
        if correct_spelling:
            blob = TextBlob(text)
            text = str(blob.correct())
            print(f"After spell correction: {text}")
        
        # Tokenize
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        print(f"After tokenization: {tokens}")
        
        # Basic cleaning
        if lowercase:
            tokens = [token.lower() for token in tokens]
            print(f"After lowercasing: {tokens}")
        
        # Remove non-alphabetic tokens (punctuation, numbers if requested)
        if remove_numbers:
            tokens = [token for token in tokens if token.isalpha()]
        else:
            tokens = [token for token in tokens if token.isalnum()]
        print(f"After removing non-alphanumeric: {tokens}")
        
        # Remove short words
        tokens = [token for token in tokens if len(token) >= min_word_length]
        print(f"After removing short words: {tokens}")
        
        # Remove stopwords if requested
        if remove_stopwords:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            print(f"After removing stopwords: {tokens}")
        
        # Apply stemming or lemmatization (not both)
        if stemming and not lemmatization:
            tokens = [self.stemmer.stem(token) for token in tokens]
            print(f"After stemming: {tokens}")
        elif lemmatization:
            tokens = [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token)) 
                     for token in tokens]
            print(f"After lemmatization: {tokens}")
        
        return tokens
    
    def _get_wordnet_pos(self, word):
        """Get part of speech for lemmatization"""
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                       "N": wordnet.NOUN,
                       "V": wordnet.VERB,
                       "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN

# Example usage
normalizer = TextNormalizer()

# Test with different types of text
examples = [
    "The runners were running quickly through the beautiful gardens!",
    "I'm loving these amazingly delicious cookies! They're the best.",
    "The children's geese were flying over the peaceful lake."
]

for text in examples:
    print("\n" + "="*60)
    normalized = normalizer.normalize_text(
        text,
        lowercase=True,
        lemmatization=True,
        remove_stopwords=True,
        min_word_length=3
    )
    print(f"Final result: {normalized}")
    print("="*60)
```

## 🎯 Choosing the Right Normalization Strategy

### For Search Applications

```python
def normalize_for_search(text):
    """Normalization optimized for search"""
    normalizer = TextNormalizer()
    return normalizer.normalize_text(
        text,
        lowercase=True,
        correct_spelling=False,  # Don't change user queries
        lemmatization=True,
        remove_stopwords=False,  # Keep stopwords for phrase search
        min_word_length=2
    )

# Example
search_query = "How to cook delicious Italian pasta?"
normalized_query = normalize_for_search(search_query)
print(f"Search normalization: {normalized_query}")
```

### For Sentiment Analysis

```python
def normalize_for_sentiment(text):
    """Normalization that preserves emotional content"""
    normalizer = TextNormalizer()
    return normalizer.normalize_text(
        text,
        lowercase=True,
        correct_spelling=True,  # Fix typos that might confuse sentiment
        lemmatization=True,
        remove_stopwords=False,  # Keep words like "not", "very"
        min_word_length=2
    )

# Example
review = "This movie was absolutely terrible! I'm disappointed."
normalized_review = normalize_for_sentiment(review)
print(f"Sentiment normalization: {normalized_review}")
```

### For Topic Modeling

```python
def normalize_for_topics(text):
    """Normalization focused on content words"""
    normalizer = TextNormalizer()
    return normalizer.normalize_text(
        text,
        lowercase=True,
        correct_spelling=True,
        lemmatization=True,
        remove_stopwords=True,  # Remove common words
        min_word_length=3  # Remove very short words
    )

# Example
article = "The latest scientific research shows promising results in cancer treatment."
normalized_article = normalize_for_topics(article)
print(f"Topic modeling normalization: {normalized_article}")
```

## 🚨 Common Normalization Pitfalls

### 1. Over-normalization

```python
# Bad: Lost too much information
original = "Dr. Smith's iPhone costs $500 in the U.S.A."
over_normalized = ["dr", "smith", "iphone", "cost", "usa"]  # Lost money amount, lost structure

# Better: Preserve important information
better_normalized = ["dr", "smith", "iphone", "cost", "500", "usa"]
```

### 2. Context-Dependent Words

```python
# The word "bank" has different meanings:
sentences = [
    "I went to the bank to deposit money.",      # Financial institution
    "The boat sailed along the river bank.",     # Side of river
    "I can bank on his support."                 # Rely on
]

# Same normalization might not work for all contexts
```

### 3. Domain-Specific Terms

```python
# Medical text
medical = "The patient shows symptoms of COVID-19."
# Don't normalize disease names or medical terms!

# Technical text  
technical = "The API returns a JSON response."
# Don't normalize technical acronyms!

# Legal text
legal = "As per Section 3.1.4 of the agreement."
# Don't normalize legal references!
```

## 🏋️ Practice Exercises

Try normalizing these challenging examples:

```python
# Exercise 1: Social media post
social_text = "OMG! Just watched the BEST movie ever!!! The actors were amazng and the story was incredble!"

# Exercise 2: Academic paper abstract
academic_text = "This study investigates the effectiveness of machine learning algorithms in predicting stock market trends using historical data."

# Exercise 3: Customer review
review_text = "The products quality is realy good but the shipping was very slow. I'd definately recommend it though!"

# Exercise 4: News article
news_text = "The CEO announced that the company's quarterly earnings exceeded expectations by 15%."

# Your task: Choose appropriate normalization settings for each text type
```

## 🔍 Advanced Normalization Techniques

### Handling Abbreviations and Acronyms

```python
def expand_abbreviations(text):
    """Expand common abbreviations"""
    abbreviations = {
        "can't": "cannot",
        "won't": "will not", 
        "it's": "it is",
        "they're": "they are",
        "we're": "we are",
        "I'm": "I am",
        "you're": "you are",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "there's": "there is"
    }
    
    for abbrev, full_form in abbreviations.items():
        text = text.replace(abbrev, full_form)
        text = text.replace(abbrev.upper(), full_form.upper())
        text = text.replace(abbrev.capitalize(), full_form.capitalize())
    
    return text

text = "I can't believe it's working! We're finally done."
expanded = expand_abbreviations(text)
print(f"Expanded: {expanded}")
```

### Handling Numbers and Dates

```python
import re

def normalize_numbers(text):
    """Convert number words to digits or vice versa"""
    
    # Convert digit numbers to words (useful for some applications)
    def digits_to_words(match):
        number = int(match.group())
        if number <= 20:
            words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
                    "eighteen", "nineteen", "twenty"]
            return words[number]
        return match.group()  # Keep as is for larger numbers
    
    # Replace digits with words for numbers 0-20
    text = re.sub(r'\b(\d+)\b', digits_to_words, text)
    
    return text

text = "I have 3 cats and 15 dogs."
normalized = normalize_numbers(text)
print(f"Normalized numbers: {normalized}")
```

## 💡 Key Takeaways

1. **Choose normalization based on your task** - Search, sentiment analysis, and topic modeling need different approaches
2. **Don't over-normalize** - Sometimes variations contain important information
3. **Consider your domain** - Technical, medical, and legal texts have special requirements
4. **Test with real data** - Always validate your normalization with actual data from your use case
5. **Lemmatization > Stemming** - Usually gives better results, though it's slower
6. **Preserve context when needed** - Some applications benefit from keeping original formatting

## 🔗 Quick Reference

```python
# Quick normalization templates

# Conservative (preserves more information)
def conservative_normalize(text):
    normalizer = TextNormalizer()
    return normalizer.normalize_text(text, lowercase=True, lemmatization=True, 
                                   remove_stopwords=False, min_word_length=2)

# Aggressive (removes more noise)  
def aggressive_normalize(text):
    normalizer = TextNormalizer()
    return normalizer.normalize_text(text, lowercase=True, lemmatization=True,
                                   remove_stopwords=True, remove_numbers=True, min_word_length=3)

# Balanced (good for most applications)
def balanced_normalize(text):
    normalizer = TextNormalizer()
    return normalizer.normalize_text(text, lowercase=True, lemmatization=True,
                                   remove_stopwords=True, min_word_length=2)
```

Ready to learn how to convert text into numbers? Continue to [Bag of Words & TF-IDF](./05_bag_of_words_tfidf.md)!
