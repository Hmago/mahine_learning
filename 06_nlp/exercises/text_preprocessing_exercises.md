# Text Preprocessing Exercises

Welcome to hands-on practice! These exercises will help you master text preprocessing through real-world scenarios and progressively challenging problems.

## 🎯 Exercise Overview

Each exercise includes:
- **Problem description** with business context
- **Starter code** to get you going
- **Sample data** to work with
- **Expected outcomes** to check your work
- **Extension challenges** for deeper learning

## 📝 Exercise 1: Social Media Text Cleaner

**Business Context:** You're building a sentiment analysis system for a social media monitoring company. The raw tweets and posts are full of noise that needs cleaning.

**Your Task:** Build a comprehensive social media text cleaner.

```python
# Starter Code
import re
import string

def social_media_cleaner(text):
    """
    Clean social media text for sentiment analysis
    
    Requirements:
    1. Remove URLs (http/https links)
    2. Handle @mentions (remove @ but keep username)
    3. Handle hashtags (remove # but keep word)
    4. Remove extra whitespace
    5. Convert to lowercase
    6. Handle emojis (either remove or convert to text)
    
    Args:
        text (str): Raw social media text
        
    Returns:
        str: Cleaned text ready for analysis
    """
    
    # Your implementation here
    pass

# Test Data
test_posts = [
    "OMG!!! Just watched @Marvel's new movie 🔥🔥🔥 SO GOOD!!! #Marvel #MustWatch https://t.co/abc123",
    "Can't believe how amazing this product is! 😍 Thanks @company for the great service! #CustomerService",
    "Terrible experience at the restaurant today 😞 Food was cold and service was slow... #disappointed",
    "RT @friend: This sunset is absolutely beautiful! 🌅 #photography #nature https://instagram.com/abc"
]

# Expected behavior example:
# Input: "Love this movie! 😍 #amazing @everyone should watch https://link.com"
# Output: "love this movie amazing everyone should watch"

# Test your function
for post in test_posts:
    cleaned = social_media_cleaner(post)
    print(f"Original: {post}")
    print(f"Cleaned: {cleaned}")
    print("-" * 50)
```

**Extension Challenges:**
1. Add emoji-to-text conversion instead of removal
2. Preserve important punctuation for sentiment (e.g., !!!)
3. Handle different languages in hashtags
4. Create different cleaning levels (light, medium, aggressive)

## 📊 Exercise 2: Document Classification Feature Engineering

**Business Context:** You're working for a law firm that needs to automatically categorize legal documents into different practice areas (corporate, litigation, real estate, etc.).

**Your Task:** Build a complete feature engineering pipeline for document classification.

```python
# Starter Code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def legal_document_processor(documents, labels=None, 
                           ngram_range=(1, 2), 
                           max_features=5000,
                           feature_selection=True,
                           k_best=1000):
    """
    Process legal documents for classification
    
    Requirements:
    1. Handle legal-specific text (citations, case names, etc.)
    2. Create appropriate n-gram features
    3. Apply TF-IDF weighting
    4. Select most informative features
    5. Preserve legal terminology
    
    Args:
        documents (list): List of legal document texts
        labels (list): Document categories (for feature selection)
        ngram_range (tuple): N-gram range for feature extraction
        max_features (int): Maximum number of features
        feature_selection (bool): Whether to apply feature selection
        k_best (int): Number of best features to select
        
    Returns:
        tuple: (feature_matrix, feature_names, processor_pipeline)
    """
    
    # Your implementation here
    pass

# Test Data
legal_documents = [
    """
    PURCHASE AND SALE AGREEMENT
    
    This Purchase and Sale Agreement ("Agreement") is entered into on January 15, 2024,
    between ABC Corp, a Delaware corporation ("Buyer"), and XYZ LLC, a California limited 
    liability company ("Seller"). The parties agree to the sale of real property located 
    at 123 Main Street, San Francisco, CA 94102 for the purchase price of $2,500,000.
    """,
    
    """
    COMPLAINT FOR BREACH OF CONTRACT
    
    Plaintiff ABC Company hereby alleges as follows:
    1. This Court has jurisdiction pursuant to 28 U.S.C. § 1332.
    2. On or about December 1, 2023, Plaintiff and Defendant entered into a written 
    contract for the provision of consulting services.
    3. Defendant materially breached said contract by failing to perform services
    as specified in Section 3.2 of the Agreement.
    """,
    
    """
    ARTICLES OF INCORPORATION
    
    The undersigned, in order to form a corporation under the laws of the State of 
    Delaware, do hereby certify:
    Article I: The name of the corporation is TechStart Inc.
    Article II: The corporation is authorized to issue 10,000,000 shares of common stock.
    Article III: The registered office is located at 1234 Corporate Blvd, Wilmington, DE.
    """,
    
    """
    LEASE AGREEMENT
    
    This Lease Agreement is made between Property Owner LLC ("Landlord") and 
    John Smith ("Tenant") for the premises located at 456 Oak Avenue, Apartment 2B,
    Berkeley, CA 94704. The monthly rent is $3,200, due on the first day of each month.
    The lease term begins February 1, 2024 and ends January 31, 2025.
    """
]

document_labels = ["real_estate", "litigation", "corporate", "real_estate"]

# Test your function
features, feature_names, pipeline = legal_document_processor(
    legal_documents, 
    document_labels, 
    ngram_range=(1, 3),
    max_features=1000,
    k_best=100
)

print(f"Feature matrix shape: {features.shape}")
print(f"Top 10 features: {feature_names[:10]}")
```

**Extension Challenges:**
1. Add legal citation recognition and preservation
2. Handle different document formats (contracts vs. court filings)
3. Create domain-specific stopword lists
4. Implement custom tokenization for legal terms

## 🌍 Exercise 3: Multilingual Product Review Analyzer

**Business Context:** Your e-commerce company receives product reviews in multiple languages. You need to normalize and prepare them for sentiment analysis.

**Your Task:** Build a multilingual text preprocessing system.

```python
# Starter Code
import langdetect
from textblob import TextBlob

def multilingual_review_processor(reviews, 
                                target_language='english',
                                normalize_ratings=True,
                                handle_mixed_languages=True):
    """
    Process multilingual product reviews
    
    Requirements:
    1. Detect language of each review
    2. Translate non-English reviews (optional)
    3. Extract and normalize ratings/scores
    4. Handle mixed-language reviews
    5. Standardize text formatting
    6. Preserve important product-specific terms
    
    Args:
        reviews (list): List of review dictionaries with 'text' and 'rating'
        target_language (str): Target language for processing
        normalize_ratings (bool): Whether to normalize rating scales
        handle_mixed_languages (bool): Whether to handle mixed language text
        
    Returns:
        list: Processed reviews with language metadata
    """
    
    # Your implementation here
    pass

# Test Data
multilingual_reviews = [
    {
        "text": "This product is amazing! Great quality and fast shipping. Highly recommend!",
        "rating": 5,
        "rating_scale": "1-5"
    },
    {
        "text": "Este producto es terrible. No funciona como se describe y el servicio al cliente es malo.",
        "rating": 2,
        "rating_scale": "1-5"
    },
    {
        "text": "Très bon produit, livraison rapide. Je recommande vivement!",
        "rating": 4,
        "rating_scale": "1-5"
    },
    {
        "text": "Das ist ein großartiges Produkt! Excellent quality und fast delivery. 10/10 würde wieder kaufen!",
        "rating": 10,
        "rating_scale": "1-10"
    },
    {
        "text": "製品の品質は良いですが、価格が高すぎます。Good value とは言えません。",
        "rating": 3,
        "rating_scale": "1-5"
    }
]

# Test your function
processed_reviews = multilingual_review_processor(
    multilingual_reviews,
    target_language='english',
    normalize_ratings=True
)

for review in processed_reviews:
    print(f"Original: {review.get('original_text', '')[:50]}...")
    print(f"Processed: {review.get('processed_text', '')[:50]}...")
    print(f"Language: {review.get('detected_language', 'unknown')}")
    print(f"Normalized Rating: {review.get('normalized_rating', 'N/A')}")
    print("-" * 50)
```

**Extension Challenges:**
1. Add language-specific preprocessing rules
2. Implement custom translation for product-specific terms
3. Handle code-switching (mixed languages within sentences)
4. Create language confidence scores

## 📈 Exercise 4: Financial News Preprocessing Pipeline

**Business Context:** You're building a trading algorithm that analyzes financial news for market sentiment. The text needs special handling for financial terms, numbers, and time-sensitive information.

**Your Task:** Create a specialized financial text preprocessor.

```python
# Starter Code
import re
from datetime import datetime

def financial_news_processor(articles, 
                           preserve_numbers=True,
                           preserve_tickers=True,
                           normalize_companies=True,
                           extract_metrics=True):
    """
    Process financial news articles for sentiment analysis
    
    Requirements:
    1. Preserve stock tickers (e.g., AAPL, MSFT)
    2. Handle financial numbers and percentages
    3. Normalize company name variations
    4. Extract key financial metrics
    5. Preserve temporal information
    6. Handle financial terminology properly
    
    Args:
        articles (list): List of financial news article texts
        preserve_numbers (bool): Whether to keep financial numbers
        preserve_tickers (bool): Whether to preserve stock symbols
        normalize_companies (bool): Whether to standardize company names
        extract_metrics (bool): Whether to extract financial metrics
        
    Returns:
        list: Processed articles with extracted financial information
    """
    
    # Your implementation here
    pass

# Test Data
financial_articles = [
    """
    Apple Inc. (AAPL) reported quarterly earnings that beat expectations, with revenue 
    up 15.6% year-over-year to $89.5 billion. The iPhone maker's stock jumped 5.2% 
    in after-hours trading. CEO Tim Cook highlighted strong performance in the Services 
    segment, which grew 16.9% to $19.8B. Apple's market cap now exceeds $2.8 trillion.
    """,
    
    """
    Tesla (TSLA) shares plummeted 8.3% after the electric vehicle manufacturer missed 
    Q3 delivery targets. The company delivered 435,059 vehicles, falling short of the 
    450,000 consensus estimate. Tesla's gross automotive margin compressed to 19.3% 
    from 21.5% in the previous quarter. Analysts remain divided on TSLA's $800B valuation.
    """,
    
    """
    The Federal Reserve announced a 0.75% interest rate hike, bringing the federal funds 
    rate to 3.25%. Fed Chair Jerome Powell signaled additional tightening measures to 
    combat inflation, which remains at 8.2% year-over-year. The Dow Jones Industrial 
    Average fell 1.7% following the announcement, while the S&P 500 declined 2.1%.
    """
]

# Test your function
processed_articles = financial_news_processor(
    financial_articles,
    preserve_numbers=True,
    preserve_tickers=True,
    extract_metrics=True
)

for i, article in enumerate(processed_articles):
    print(f"Article {i+1}:")
    print(f"Processed text: {article.get('processed_text', '')[:100]}...")
    print(f"Extracted tickers: {article.get('tickers', [])}")
    print(f"Financial metrics: {article.get('metrics', {})}")
    print("-" * 50)
```

**Extension Challenges:**
1. Add company name resolution (Apple Inc. → AAPL)
2. Extract and categorize financial events
3. Handle different financial data formats
4. Create financial entity linking

## 🏥 Exercise 5: Medical Text Anonymization and Preprocessing

**Business Context:** You're working with a healthcare analytics company that needs to process clinical notes while protecting patient privacy.

**Your Task:** Build a medical text preprocessor with anonymization features.

```python
# Starter Code
import re
from datetime import datetime, timedelta

def medical_text_processor(clinical_notes,
                         anonymize_phi=True,
                         preserve_medical_terms=True,
                         standardize_measurements=True,
                         extract_temporal_info=True):
    """
    Process medical text with privacy protection
    
    Requirements:
    1. Remove/anonymize PHI (Personal Health Information)
    2. Preserve medical terminology and abbreviations
    3. Standardize medical measurements and dosages
    4. Extract temporal information (dates, durations)
    5. Handle medical abbreviations and acronyms
    6. Maintain clinical context while ensuring privacy
    
    Args:
        clinical_notes (list): List of clinical note texts
        anonymize_phi (bool): Whether to anonymize personal information
        preserve_medical_terms (bool): Whether to keep medical terminology
        standardize_measurements (bool): Whether to normalize measurements
        extract_temporal_info (bool): Whether to extract time information
        
    Returns:
        list: Processed notes with anonymization and extracted info
    """
    
    # Your implementation here
    pass

# Test Data (Note: This is synthetic medical data for educational purposes)
clinical_notes = [
    """
    Patient: John Smith (DOB: 03/15/1965, MRN: 123456789)
    Date: 10/15/2023
    
    Chief Complaint: Chest pain and shortness of breath
    
    History: 58 y.o. male presents with acute onset chest pain starting 2 hours ago.
    Pain is described as crushing, 8/10 severity, radiating to left arm. Patient has 
    history of HTN, DM2, and hyperlipidemia. Takes metformin 1000mg BID and lisinopril 
    10mg daily. Vital signs: BP 160/95 mmHg, HR 102 bpm, RR 22, O2 sat 94% on RA.
    
    Assessment: Likely ACS. Order troponin, ECG, CBC, BMP.
    """,
    
    """
    Patient: Mary Johnson (DOB: 07/22/1978, MRN: 987654321)
    Date: 10/16/2023
    
    Follow-up visit for diabetes management. HbA1c improved from 9.2% to 7.8% over 
    past 3 months. Patient reports good adherence to insulin regimen (Lantus 24 units 
    qHS, Humalog 8 units TID with meals). Weight decreased 15 lbs since last visit.
    No episodes of hypoglycemia. Blood glucose logs show average of 145 mg/dL.
    
    Plan: Continue current regimen. Return in 3 months for follow-up.
    """
]

# Test your function
processed_notes = medical_text_processor(
    clinical_notes,
    anonymize_phi=True,
    preserve_medical_terms=True,
    standardize_measurements=True
)

for i, note in enumerate(processed_notes):
    print(f"Clinical Note {i+1}:")
    print(f"Anonymized text: {note.get('anonymized_text', '')[:200]}...")
    print(f"Extracted medications: {note.get('medications', [])}")
    print(f"Extracted measurements: {note.get('measurements', {})}")
    print(f"Medical terms preserved: {note.get('medical_terms', [])[:5]}")
    print("-" * 50)
```

**Extension Challenges:**
1. Add medical entity recognition and linking
2. Implement HIPAA-compliant anonymization
3. Handle different clinical note formats
4. Create medical abbreviation expansion

## 🔍 Exercise 6: Search Query Understanding and Expansion

**Business Context:** You're improving a search engine by preprocessing and understanding user queries to provide better results.

**Your Task:** Build an intelligent query processor that understands user intent.

```python
# Starter Code
from collections import defaultdict

def search_query_processor(queries,
                         expand_synonyms=True,
                         handle_typos=True,
                         extract_intent=True,
                         normalize_entities=True):
    """
    Process search queries for better understanding
    
    Requirements:
    1. Correct common typos and misspellings
    2. Expand queries with synonyms and related terms
    3. Extract search intent (informational, navigational, transactional)
    4. Normalize named entities (companies, products, locations)
    5. Handle different query types (questions, keywords, phrases)
    6. Preserve important search operators
    
    Args:
        queries (list): List of search query texts
        expand_synonyms (bool): Whether to add synonym expansion
        handle_typos (bool): Whether to correct spelling errors
        extract_intent (bool): Whether to classify search intent
        normalize_entities (bool): Whether to standardize entity names
        
    Returns:
        list: Processed queries with expanded and corrected text
    """
    
    # Your implementation here
    pass

# Test Data
search_queries = [
    "how to cook pasta recipies",  # Typo + informational intent
    "apple iphone 14 buy online",  # Transactional intent
    "weather in new york city",    # Informational + location
    "facebook login page",         # Navigational intent
    "best restaurants near me",    # Local + informational
    "what is machine lerning",     # Typo + definitional
    "amazon prime membership cost", # Transactional + specific product
    "covid 19 symptoms treatment", # Medical informational
    "how to fix broken laptop screen", # How-to informational
    "youtube music download app"   # Navigational + product
]

# Test your function
processed_queries = search_query_processor(
    search_queries,
    expand_synonyms=True,
    handle_typos=True,
    extract_intent=True
)

for i, query in enumerate(processed_queries):
    print(f"Query {i+1}:")
    print(f"Original: {search_queries[i]}")
    print(f"Corrected: {query.get('corrected_query', '')}")
    print(f"Expanded: {query.get('expanded_query', '')}")
    print(f"Intent: {query.get('intent', 'unknown')}")
    print(f"Entities: {query.get('entities', [])}")
    print("-" * 50)
```

**Extension Challenges:**
1. Add personalization based on user history
2. Implement query classification for different domains
3. Handle voice search queries (more conversational)
4. Create query similarity and clustering

## 🎯 Exercise Completion Checklist

For each exercise, make sure you can:

- [ ] **Identify the problem**: Understand what type of text processing is needed
- [ ] **Choose appropriate techniques**: Select the right combination of cleaning, normalization, and feature extraction
- [ ] **Handle edge cases**: Deal with unusual input and error conditions
- [ ] **Evaluate results**: Check that your preprocessing improves downstream tasks
- [ ] **Optimize performance**: Ensure your code can handle real-world data volumes
- [ ] **Document your approach**: Explain your design decisions and trade-offs

## 🏆 Advanced Challenge: Build Your Own Preprocessing Library

**Ultimate Challenge:** Combine all techniques learned to create a comprehensive, reusable text preprocessing library.

**Requirements:**
1. **Modular Design**: Separate classes for different preprocessing tasks
2. **Configurable Pipelines**: Easy to customize for different domains
3. **Performance Optimization**: Efficient processing of large text collections
4. **Multilingual Support**: Handle multiple languages gracefully
5. **Comprehensive Testing**: Unit tests for all major functions
6. **Documentation**: Clear usage examples and API documentation

This library should be something you could actually use in real projects or even open-source for the community!

## 📚 Additional Resources

### Datasets for Practice
- **IMDB Movie Reviews**: Sentiment analysis preprocessing
- **20 Newsgroups**: Topic classification text preparation
- **CoNLL-2003**: Named entity recognition data
- **Reuters News**: News categorization preprocessing
- **Amazon Product Reviews**: Multilingual review processing

### Performance Benchmarks
- **Processing Speed**: Aim for 1000+ documents per second
- **Memory Efficiency**: Handle datasets larger than available RAM
- **Accuracy Improvement**: Preprocessing should improve downstream task performance by 5-10%

Ready to become a text preprocessing expert? Start with Exercise 1 and work your way up. Remember, good preprocessing is the foundation of all successful NLP projects!
