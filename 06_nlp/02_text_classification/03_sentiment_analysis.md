# Sentiment Analysis and Opinion Mining

## 🎯 What You'll Learn

Sentiment analysis goes beyond simple positive/negative classification. You'll master techniques to understand emotions, opinions, and attitudes in text, from basic polarity detection to sophisticated aspect-based analysis that businesses use to understand customer feedback.

## 🧠 Understanding Sentiment: More Than Just Happy or Sad

Think of sentiment analysis as being an emotional interpreter for machines. Just like humans can detect sarcasm, frustration, or excitement in speech, you'll teach computers to:

- **Detect basic polarity** → Positive, negative, neutral
- **Understand emotions** → Joy, anger, fear, surprise, sadness
- **Identify aspects** → What specifically people like or dislike
- **Handle context** → Sarcasm, cultural nuances, domain-specific language

## 🎭 The Sentiment Analysis Hierarchy

### 1. Basic Polarity Detection: The Foundation

**Goal:** Classify text as positive, negative, or neutral.

**Real-world example:** "This movie is amazing!" → Positive

```python
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re

class BasicSentimentAnalyzer:
    """A comprehensive sentiment analysis toolkit"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags for social media
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_with_textblob(self, texts):
        """Quick sentiment analysis using TextBlob"""
        results = []
        
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert polarity to label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            results.append({
                'text': text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': label,
                'confidence': abs(polarity)
            })
        
        return pd.DataFrame(results)
    
    def train_custom_model(self, texts, labels):
        """Train a custom sentiment classifier"""
        
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
        
        print("Custom sentiment model trained successfully!")
        
        return self
    
    def predict_sentiment(self, texts):
        """Predict sentiment using custom trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Preprocess and vectorize
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(cleaned_texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            confidence = max(probs)
            results.append({
                'text': text,
                'sentiment': pred,
                'confidence': confidence,
                'probabilities': dict(zip(self.model.classes_, probs))
            })
        
        return results
    
    def analyze_sentiment_patterns(self, df, text_col='text', sentiment_col='sentiment'):
        """Analyze patterns in sentiment data"""
        
        print("Sentiment Distribution:")
        sentiment_counts = df[sentiment_col].value_counts()
        print(sentiment_counts)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment distribution
        sentiment_counts.plot(kind='bar', ax=axes[0,0], color=['red', 'gray', 'green'])
        axes[0,0].set_title('Sentiment Distribution')
        axes[0,0].set_ylabel('Count')
        
        # Text length vs sentiment
        df['text_length'] = df[text_col].str.len()
        df.boxplot(column='text_length', by=sentiment_col, ax=axes[0,1])
        axes[0,1].set_title('Text Length by Sentiment')
        
        # Polarity distribution (if available)
        if 'polarity' in df.columns:
            df['polarity'].hist(bins=20, ax=axes[1,0], alpha=0.7)
            axes[1,0].set_title('Polarity Distribution')
            axes[1,0].set_xlabel('Polarity Score')
        
        # Confidence distribution (if available)
        if 'confidence' in df.columns:
            df['confidence'].hist(bins=20, ax=axes[1,1], alpha=0.7)
            axes[1,1].set_title('Confidence Distribution')
            axes[1,1].set_xlabel('Confidence Score')
        
        plt.tight_layout()
        plt.show()
        
        return df

# Example: Movie Review Sentiment Analysis
movie_reviews = [
    "This movie is absolutely fantastic! The acting is superb and the plot is engaging.",
    "Terrible film. Complete waste of time. Poor acting and boring storyline.",
    "An okay movie. Not great, not terrible. Worth watching if you have time.",
    "I loved every minute of this film! Outstanding performances and brilliant direction.",
    "Disappointing. Expected much better from this director and cast.",
    "Average movie with some good moments but overall forgettable.",
    "Incredible cinematography and amazing soundtrack. A true masterpiece!",
    "Boring and predictable. The ending was obvious from the first scene.",
    "Good entertainment value. Fun to watch with friends on a weekend.",
    "Awful movie. Poor script, bad acting, and terrible special effects."
]

# Quick analysis with TextBlob
analyzer = BasicSentimentAnalyzer()
textblob_results = analyzer.analyze_with_textblob(movie_reviews)
print("TextBlob Sentiment Analysis Results:")
print(textblob_results[['text', 'sentiment', 'polarity', 'confidence']].head())

# Analyze patterns
analyzer.analyze_sentiment_patterns(textblob_results)

# Create training data for custom model
training_texts = movie_reviews * 5  # Repeat for more training data
training_labels = ['positive', 'negative', 'neutral', 'positive', 'negative', 
                  'neutral', 'positive', 'negative', 'positive', 'negative'] * 5

# Train custom model
analyzer.train_custom_model(training_texts, training_labels)

# Test predictions
test_reviews = [
    "This film exceeded all my expectations! Absolutely brilliant!",
    "Mediocre movie. Nothing special but watchable.",
    "Complete disaster. Avoid at all costs."
]

custom_predictions = analyzer.predict_sentiment(test_reviews)
print("\nCustom Model Predictions:")
for pred in custom_predictions:
    print(f"Text: '{pred['text']}'")
    print(f"Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
    print()
```

### 2. Emotion Detection: Beyond Positive and Negative

**Goal:** Identify specific emotions like joy, anger, fear, sadness, surprise, disgust.

**Why it matters:** Understanding specific emotions provides richer insights for customer service, mental health apps, and content moderation.

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionAnalyzer:
    """Advanced emotion detection using transformer models"""
    
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.emotion_pipeline = None
        self.load_model()
        
        # Define emotion mappings
        self.emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#DC143C',
            'fear': '#800080',
            'surprise': '#FF69B4',
            'disgust': '#228B22',
            'neutral': '#808080'
        }
    
    def load_model(self):
        """Load pre-trained emotion classification model"""
        try:
            self.emotion_pipeline = pipeline(
                "text-classification", 
                model=self.model_name,
                return_all_scores=True
            )
            print(f"Loaded emotion model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler approach
            self.emotion_pipeline = None
    
    def analyze_emotions(self, texts):
        """Analyze emotions in texts"""
        if self.emotion_pipeline is None:
            return self.analyze_emotions_rule_based(texts)
        
        results = []
        
        for text in texts:
            try:
                emotion_scores = self.emotion_pipeline(text)[0]
                
                # Get top emotion
                top_emotion = max(emotion_scores, key=lambda x: x['score'])
                
                # Create emotion distribution
                emotion_dist = {item['label']: item['score'] for item in emotion_scores}
                
                results.append({
                    'text': text,
                    'top_emotion': top_emotion['label'],
                    'confidence': top_emotion['score'],
                    'emotion_scores': emotion_dist
                })
                
            except Exception as e:
                print(f"Error analyzing text: {e}")
                results.append({
                    'text': text,
                    'top_emotion': 'neutral',
                    'confidence': 0.5,
                    'emotion_scores': {'neutral': 1.0}
                })
        
        return results
    
    def analyze_emotions_rule_based(self, texts):
        """Fallback rule-based emotion analysis"""
        
        emotion_keywords = {
            'joy': ['happy', 'excited', 'wonderful', 'amazing', 'fantastic', 'love', 'great'],
            'sadness': ['sad', 'depressed', 'cry', 'disappointed', 'terrible', 'awful', 'bad'],
            'anger': ['angry', 'furious', 'hate', 'mad', 'annoyed', 'frustrated', 'outraged'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'unexpected', 'wow', 'amazing', 'incredible'],
            'disgust': ['disgusting', 'gross', 'horrible', 'nasty', 'repulsive']
        }
        
        results = []
        
        for text in texts:
            text_lower = text.lower()
            emotion_scores = {emotion: 0 for emotion in emotion_keywords.keys()}
            emotion_scores['neutral'] = 1.0  # Default
            
            for emotion, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    emotion_scores[emotion] = score / len(keywords)
                    emotion_scores['neutral'] = max(0, emotion_scores['neutral'] - score/len(keywords))
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            top_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
            
            results.append({
                'text': text,
                'top_emotion': top_emotion,
                'confidence': emotion_scores[top_emotion],
                'emotion_scores': emotion_scores
            })
        
        return results
    
    def visualize_emotions(self, emotion_results):
        """Create visualizations for emotion analysis"""
        
        # Extract data
        emotions = [result['top_emotion'] for result in emotion_results]
        confidences = [result['confidence'] for result in emotion_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Emotion distribution
        emotion_counts = pd.Series(emotions).value_counts()
        colors = [self.emotion_colors.get(emotion, '#808080') for emotion in emotion_counts.index]
        emotion_counts.plot(kind='bar', ax=axes[0,0], color=colors)
        axes[0,0].set_title('Emotion Distribution')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        axes[0,1].hist(confidences, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Emotion Confidence Distribution')
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Frequency')
        
        # Emotion vs Confidence
        emotion_confidence_df = pd.DataFrame({
            'emotion': emotions,
            'confidence': confidences
        })
        emotion_confidence_df.boxplot(column='confidence', by='emotion', ax=axes[1,0])
        axes[1,0].set_title('Confidence by Emotion')
        
        # Average emotion scores heatmap
        all_emotions = set()
        for result in emotion_results:
            all_emotions.update(result['emotion_scores'].keys())
        
        emotion_matrix = []
        for result in emotion_results:
            row = [result['emotion_scores'].get(emotion, 0) for emotion in sorted(all_emotions)]
            emotion_matrix.append(row)
        
        emotion_df = pd.DataFrame(emotion_matrix, columns=sorted(all_emotions))
        sns.heatmap(emotion_df.corr(), ax=axes[1,1], cmap='coolwarm', center=0)
        axes[1,1].set_title('Emotion Correlation Heatmap')
        
        plt.tight_layout()
        plt.show()
        
        return emotion_confidence_df

# Example: Social Media Emotion Analysis
social_posts = [
    "Just got the promotion I've been working towards for months! So excited! 🎉",
    "Can't believe my flight got cancelled again. This airline is horrible. 😡",
    "Watching this movie and literally crying. Such a beautiful story. 😭",
    "Spider just crawled across my desk. I'm terrified of spiders! 😱",
    "My friend surprised me with tickets to my favorite band! I can't believe it! 😲",
    "This food tastes absolutely disgusting. How do people eat this? 🤢",
    "Another Monday morning. Just another regular day at work.",
    "Feeling so grateful for all the amazing people in my life today! ❤️",
    "Traffic is horrible today. Why does everything go wrong at once? 😤",
    "Found out my favorite restaurant is closing permanently. So sad. 💔"
]

# Analyze emotions
emotion_analyzer = EmotionAnalyzer()
emotion_results = emotion_analyzer.analyze_emotions(social_posts)

print("Emotion Analysis Results:")
for result in emotion_results[:3]:  # Show first 3
    print(f"Text: '{result['text']}'")
    print(f"Top Emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    print("All emotion scores:")
    for emotion, score in result['emotion_scores'].items():
        print(f"  {emotion}: {score:.3f}")
    print()

# Visualize results
emotion_df = emotion_analyzer.visualize_emotions(emotion_results)
```

### 3. Aspect-Based Sentiment Analysis: The Business Gold Mine

**Goal:** Identify what aspects people are talking about and their sentiment towards each aspect.

**Example:** "The food was amazing but the service was terrible." 
- Food: Positive
- Service: Negative

```python
import spacy
from collections import defaultdict

class AspectBasedSentimentAnalyzer:
    """Analyze sentiment towards specific aspects/features"""
    
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define aspect keywords for different domains
        self.aspect_keywords = {
            'restaurant': {
                'food': ['food', 'meal', 'dish', 'cuisine', 'taste', 'flavor', 'menu'],
                'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'employee'],
                'ambiance': ['atmosphere', 'ambiance', 'ambience', 'music', 'lighting', 'decor'],
                'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'bill'],
                'location': ['location', 'parking', 'area', 'neighborhood', 'accessible']
            },
            'hotel': {
                'room': ['room', 'bed', 'bathroom', 'shower', 'amenities', 'comfort'],
                'service': ['service', 'staff', 'reception', 'concierge', 'housekeeping'],
                'location': ['location', 'area', 'transportation', 'nearby', 'access'],
                'facilities': ['pool', 'gym', 'spa', 'wifi', 'parking', 'restaurant'],
                'price': ['price', 'rate', 'value', 'expensive', 'affordable', 'cost']
            },
            'product': {
                'quality': ['quality', 'build', 'material', 'durable', 'solid', 'cheap'],
                'design': ['design', 'appearance', 'look', 'style', 'color', 'size'],
                'functionality': ['performance', 'speed', 'efficiency', 'feature', 'function'],
                'price': ['price', 'cost', 'value', 'expensive', 'affordable', 'cheap'],
                'delivery': ['shipping', 'delivery', 'packaging', 'arrived', 'fast', 'slow']
            }
        }
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = BasicSentimentAnalyzer()
    
    def extract_aspects(self, text, domain='restaurant'):
        """Extract aspects mentioned in the text"""
        if self.nlp is None:
            return self.extract_aspects_simple(text, domain)
        
        doc = self.nlp(text.lower())
        aspects_found = []
        
        # Get aspect keywords for the domain
        domain_aspects = self.aspect_keywords.get(domain, {})
        
        for aspect, keywords in domain_aspects.items():
            for keyword in keywords:
                if keyword in text.lower():
                    # Find the sentence containing this keyword
                    for sent in doc.sents:
                        if keyword in sent.text:
                            aspects_found.append({
                                'aspect': aspect,
                                'keyword': keyword,
                                'sentence': sent.text.strip(),
                                'position': text.lower().find(keyword)
                            })
        
        return aspects_found
    
    def extract_aspects_simple(self, text, domain='restaurant'):
        """Simple keyword-based aspect extraction"""
        aspects_found = []
        domain_aspects = self.aspect_keywords.get(domain, {})
        
        # Split text into sentences
        sentences = text.split('.')
        
        for aspect, keywords in domain_aspects.items():
            for keyword in keywords:
                if keyword in text.lower():
                    # Find which sentence contains this keyword
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            aspects_found.append({
                                'aspect': aspect,
                                'keyword': keyword,
                                'sentence': sentence.strip(),
                                'position': text.lower().find(keyword)
                            })
                            break
        
        return aspects_found
    
    def analyze_aspect_sentiment(self, text, domain='restaurant'):
        """Analyze sentiment for each aspect in the text"""
        
        # Extract aspects
        aspects = self.extract_aspects(text, domain)
        
        if not aspects:
            # No specific aspects found, analyze overall sentiment
            overall_result = self.sentiment_analyzer.analyze_with_textblob([text])
            return {
                'overall_sentiment': overall_result.iloc[0]['sentiment'],
                'overall_polarity': overall_result.iloc[0]['polarity'],
                'aspects': [],
                'text': text
            }
        
        # Analyze sentiment for each aspect's sentence
        aspect_sentiments = []
        
        for aspect_info in aspects:
            sentence = aspect_info['sentence']
            if sentence:  # Make sure sentence is not empty
                sentence_sentiment = self.sentiment_analyzer.analyze_with_textblob([sentence])
                
                aspect_sentiments.append({
                    'aspect': aspect_info['aspect'],
                    'keyword': aspect_info['keyword'],
                    'sentence': sentence,
                    'sentiment': sentence_sentiment.iloc[0]['sentiment'],
                    'polarity': sentence_sentiment.iloc[0]['polarity'],
                    'confidence': sentence_sentiment.iloc[0]['confidence']
                })
        
        # Overall sentiment
        overall_result = self.sentiment_analyzer.analyze_with_textblob([text])
        
        return {
            'overall_sentiment': overall_result.iloc[0]['sentiment'],
            'overall_polarity': overall_result.iloc[0]['polarity'],
            'aspects': aspect_sentiments,
            'text': text
        }
    
    def analyze_multiple_reviews(self, reviews, domain='restaurant'):
        """Analyze multiple reviews and aggregate results"""
        
        all_results = []
        aspect_summary = defaultdict(list)
        
        for review in reviews:
            result = self.analyze_aspect_sentiment(review, domain)
            all_results.append(result)
            
            # Aggregate aspect sentiments
            for aspect_info in result['aspects']:
                aspect_summary[aspect_info['aspect']].append(aspect_info['polarity'])
        
        # Calculate average sentiment for each aspect
        aspect_averages = {}
        for aspect, polarities in aspect_summary.items():
            if polarities:
                avg_polarity = np.mean(polarities)
                aspect_averages[aspect] = {
                    'average_polarity': avg_polarity,
                    'sentiment': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral',
                    'mention_count': len(polarities),
                    'polarities': polarities
                }
        
        return {
            'individual_results': all_results,
            'aspect_summary': aspect_averages,
            'total_reviews': len(reviews)
        }
    
    def visualize_aspect_sentiment(self, analysis_results):
        """Visualize aspect-based sentiment analysis results"""
        
        aspect_summary = analysis_results['aspect_summary']
        
        if not aspect_summary:
            print("No aspects found to visualize.")
            return
        
        # Prepare data
        aspects = list(aspect_summary.keys())
        avg_polarities = [aspect_summary[aspect]['average_polarity'] for aspect in aspects]
        mention_counts = [aspect_summary[aspect]['mention_count'] for aspect in aspects]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average sentiment by aspect
        colors = ['red' if p < -0.1 else 'green' if p > 0.1 else 'gray' for p in avg_polarities]
        axes[0,0].bar(aspects, avg_polarities, color=colors)
        axes[0,0].set_title('Average Sentiment by Aspect')
        axes[0,0].set_ylabel('Average Polarity')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Mention frequency
        axes[0,1].bar(aspects, mention_counts, color='skyblue')
        axes[0,1].set_title('Aspect Mention Frequency')
        axes[0,1].set_ylabel('Number of Mentions')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Sentiment distribution for each aspect
        for i, aspect in enumerate(aspects):
            polarities = aspect_summary[aspect]['polarities']
            axes[1,0].hist(polarities, alpha=0.5, label=aspect, bins=10)
        
        axes[1,0].set_title('Sentiment Distribution by Aspect')
        axes[1,0].set_xlabel('Polarity')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Overall sentiment vs aspect sentiment
        individual_results = analysis_results['individual_results']
        overall_sentiments = [result['overall_polarity'] for result in individual_results]
        
        # Calculate correlation between overall and aspect sentiments
        if overall_sentiments and avg_polarities:
            axes[1,1].scatter(overall_sentiments, [np.mean(avg_polarities)] * len(overall_sentiments))
            axes[1,1].set_title('Overall vs Average Aspect Sentiment')
            axes[1,1].set_xlabel('Overall Sentiment')
            axes[1,1].set_ylabel('Average Aspect Sentiment')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nAspect-Based Sentiment Summary:")
        print("-" * 50)
        for aspect, data in aspect_summary.items():
            print(f"{aspect.capitalize()}:")
            print(f"  Average Sentiment: {data['sentiment']} ({data['average_polarity']:.3f})")
            print(f"  Mentions: {data['mention_count']}")
            print()

# Example: Restaurant Review Analysis
restaurant_reviews = [
    "The food was absolutely delicious and the service was outstanding! However, the prices are quite expensive.",
    "Poor service and rude staff, but the ambiance was nice and the location is convenient.",
    "Amazing food quality and great atmosphere. The staff was friendly and helpful. Worth every penny!",
    "The meal was terrible and overpriced. The waiter was slow and inattentive. Won't be coming back.",
    "Good food and reasonable prices. The restaurant is in a great location but can get quite noisy.",
    "Excellent cuisine and perfect ambiance for a date night. Service was a bit slow but overall great experience.",
    "The food was okay, nothing special. Service was quick and efficient. Good value for money.",
    "Horrible experience. Bad food, terrible service, and way too expensive for what you get."
]

# Analyze aspect-based sentiment
aspect_analyzer = AspectBasedSentimentAnalyzer()
restaurant_analysis = aspect_analyzer.analyze_multiple_reviews(restaurant_reviews, domain='restaurant')

print("Individual Review Analysis (first 2 reviews):")
for i, result in enumerate(restaurant_analysis['individual_results'][:2]):
    print(f"\nReview {i+1}: '{result['text']}'")
    print(f"Overall Sentiment: {result['overall_sentiment']} ({result['overall_polarity']:.3f})")
    print("Aspect-specific sentiments:")
    for aspect in result['aspects']:
        print(f"  {aspect['aspect']} ({aspect['keyword']}): {aspect['sentiment']} ({aspect['polarity']:.3f})")

# Visualize results
aspect_analyzer.visualize_aspect_sentiment(restaurant_analysis)
```

## 🎯 Advanced Techniques

### 1. Handling Sarcasm and Irony

```python
class SarcasmDetector:
    """Detect sarcasm in text to improve sentiment accuracy"""
    
    def __init__(self):
        # Sarcasm indicators
        self.sarcasm_patterns = [
            r"oh\s+great",
            r"just\s+what\s+i\s+needed",
            r"fantastic\s*!+.*terrible",
            r"wonderful\s*!+.*awful",
            r"love\s+how.*not",
            r"thanks\s+for\s+nothing",
            r"couldn't\s+be\s+happier.*sarcasm",
        ]
        
        self.contradiction_words = [
            ('love', 'hate'), ('great', 'terrible'), ('amazing', 'awful'),
            ('perfect', 'horrible'), ('wonderful', 'disgusting'),
            ('fantastic', 'worst'), ('excellent', 'pathetic')
        ]
    
    def detect_sarcasm(self, text):
        """Detect potential sarcasm in text"""
        text_lower = text.lower()
        sarcasm_score = 0
        indicators = []
        
        # Pattern-based detection
        for pattern in self.sarcasm_patterns:
            if re.search(pattern, text_lower):
                sarcasm_score += 1
                indicators.append(f"Pattern: {pattern}")
        
        # Contradiction detection
        for positive, negative in self.contradiction_words:
            if positive in text_lower and negative in text_lower:
                sarcasm_score += 1
                indicators.append(f"Contradiction: {positive}/{negative}")
        
        # Punctuation patterns (excessive punctuation often indicates sarcasm)
        if re.search(r'!{2,}', text) or re.search(r'\?{2,}', text):
            sarcasm_score += 0.5
            indicators.append("Excessive punctuation")
        
        # All caps with positive words but negative context
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if caps_words:
            sarcasm_score += 0.3
            indicators.append(f"Caps words: {caps_words}")
        
        is_sarcastic = sarcasm_score > 0.5
        
        return {
            'is_sarcastic': is_sarcastic,
            'sarcasm_score': sarcasm_score,
            'indicators': indicators
        }

class SarcasmAwareSentimentAnalyzer:
    """Sentiment analyzer that handles sarcasm"""
    
    def __init__(self):
        self.sentiment_analyzer = BasicSentimentAnalyzer()
        self.sarcasm_detector = SarcasmDetector()
    
    def analyze_with_sarcasm_detection(self, texts):
        """Analyze sentiment while considering sarcasm"""
        results = []
        
        for text in texts:
            # Basic sentiment analysis
            basic_sentiment = self.sentiment_analyzer.analyze_with_textblob([text]).iloc[0]
            
            # Sarcasm detection
            sarcasm_result = self.sarcasm_detector.detect_sarcasm(text)
            
            # Adjust sentiment if sarcasm detected
            if sarcasm_result['is_sarcastic']:
                # Flip the sentiment
                original_polarity = basic_sentiment['polarity']
                adjusted_polarity = -original_polarity * 0.8  # Slightly dampened flip
                
                if adjusted_polarity > 0.1:
                    adjusted_sentiment = 'positive'
                elif adjusted_polarity < -0.1:
                    adjusted_sentiment = 'negative'
                else:
                    adjusted_sentiment = 'neutral'
            else:
                adjusted_polarity = basic_sentiment['polarity']
                adjusted_sentiment = basic_sentiment['sentiment']
            
            results.append({
                'text': text,
                'original_sentiment': basic_sentiment['sentiment'],
                'original_polarity': basic_sentiment['polarity'],
                'sarcasm_detected': sarcasm_result['is_sarcastic'],
                'sarcasm_score': sarcasm_result['sarcasm_score'],
                'sarcasm_indicators': sarcasm_result['indicators'],
                'final_sentiment': adjusted_sentiment,
                'final_polarity': adjusted_polarity
            })
        
        return results

# Example: Sarcasm Detection
sarcastic_texts = [
    "Oh great, another Monday morning!",
    "I just love waiting in traffic for hours.",
    "Fantastic! My computer crashed right before the deadline.",
    "This is a genuinely good movie, I really enjoyed it.",
    "WONDERFUL!!! The service here is absolutely TERRIBLE!",
    "Thanks for nothing, customer service!",
    "What a perfect day... it's raining and I forgot my umbrella.",
    "I'm so happy to work late again tonight.",
    "This restaurant has excellent food and great service.",
    "Love how my phone battery dies exactly when I need it most."
]

# Analyze with sarcasm awareness
sarcasm_aware_analyzer = SarcasmAwareSentimentAnalyzer()
sarcasm_results = sarcasm_aware_analyzer.analyze_with_sarcasm_detection(sarcastic_texts)

print("Sarcasm-Aware Sentiment Analysis:")
print("-" * 60)
for result in sarcasm_results:
    print(f"Text: '{result['text']}'")
    print(f"Original: {result['original_sentiment']} ({result['original_polarity']:.3f})")
    if result['sarcasm_detected']:
        print(f"🔄 SARCASM DETECTED (score: {result['sarcasm_score']:.1f})")
        print(f"Indicators: {', '.join(result['sarcasm_indicators'])}")
        print(f"Adjusted: {result['final_sentiment']} ({result['final_polarity']:.3f})")
    else:
        print(f"Final: {result['final_sentiment']} ({result['final_polarity']:.3f})")
    print()
```

### 2. Domain-Specific Sentiment Analysis

```python
class DomainSpecificSentimentAnalyzer:
    """Adapt sentiment analysis for specific domains"""
    
    def __init__(self):
        # Domain-specific lexicons
        self.domain_lexicons = {
            'financial': {
                'positive': ['profit', 'growth', 'bull', 'rally', 'gain', 'surge', 'outperform'],
                'negative': ['loss', 'decline', 'bear', 'crash', 'drop', 'plunge', 'underperform']
            },
            'medical': {
                'positive': ['recovery', 'healing', 'improvement', 'effective', 'successful', 'breakthrough'],
                'negative': ['symptom', 'pain', 'deterioration', 'complication', 'adverse', 'failure']
            },
            'technology': {
                'positive': ['innovative', 'efficient', 'fast', 'user-friendly', 'cutting-edge', 'breakthrough'],
                'negative': ['bug', 'crash', 'slow', 'outdated', 'incompatible', 'glitch']
            }
        }
        
        # Domain-specific intensifiers
        self.domain_intensifiers = {
            'financial': ['significantly', 'dramatically', 'substantially'],
            'medical': ['severely', 'critically', 'acutely'],
            'technology': ['extremely', 'incredibly', 'remarkably']
        }
    
    def analyze_domain_sentiment(self, text, domain='general'):
        """Analyze sentiment with domain-specific considerations"""
        
        # Basic analysis
        basic_analyzer = BasicSentimentAnalyzer()
        basic_result = basic_analyzer.analyze_with_textblob([text]).iloc[0]
        
        if domain not in self.domain_lexicons:
            return basic_result.to_dict()
        
        # Domain-specific analysis
        text_lower = text.lower()
        domain_lexicon = self.domain_lexicons[domain]
        
        # Count domain-specific positive/negative words
        positive_count = sum(1 for word in domain_lexicon['positive'] if word in text_lower)
        negative_count = sum(1 for word in domain_lexicon['negative'] if word in text_lower)
        
        # Check for intensifiers
        intensifier_count = sum(1 for intensifier in self.domain_intensifiers.get(domain, []) 
                              if intensifier in text_lower)
        
        # Calculate domain-adjusted sentiment
        domain_polarity = 0
        if positive_count > 0 or negative_count > 0:
            domain_polarity = (positive_count - negative_count) / (positive_count + negative_count)
            
            # Apply intensifier effect
            if intensifier_count > 0:
                domain_polarity *= (1 + 0.3 * intensifier_count)
        
        # Combine basic and domain-specific scores
        if domain_polarity != 0:
            # Weight: 70% basic sentiment, 30% domain-specific
            combined_polarity = 0.7 * basic_result['polarity'] + 0.3 * domain_polarity
        else:
            combined_polarity = basic_result['polarity']
        
        # Determine final sentiment
        if combined_polarity > 0.1:
            final_sentiment = 'positive'
        elif combined_polarity < -0.1:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'text': text,
            'domain': domain,
            'basic_sentiment': basic_result['sentiment'],
            'basic_polarity': basic_result['polarity'],
            'domain_positive_words': positive_count,
            'domain_negative_words': negative_count,
            'intensifiers': intensifier_count,
            'domain_polarity': domain_polarity,
            'final_sentiment': final_sentiment,
            'final_polarity': combined_polarity
        }

# Example: Domain-Specific Analysis
financial_texts = [
    "The stock market experienced a significant rally today with major gains across all sectors.",
    "Company profits plunged dramatically after the quarterly earnings report.",
    "The new investment strategy shows promising growth potential.",
    "Market volatility caused substantial losses for many investors."
]

medical_texts = [
    "Patient shows remarkable improvement after the new treatment protocol.",
    "Severe complications arose during the surgical procedure.",
    "The breakthrough therapy demonstrated exceptional healing results.",
    "Critical symptoms persisted despite intensive medical intervention."
]

technology_texts = [
    "This innovative software solution is incredibly user-friendly and efficient.",
    "The system crashes frequently with numerous bugs and compatibility issues.",
    "Cutting-edge technology delivers remarkably fast performance improvements.",
    "Outdated interface with extremely slow response times and glitches."
]

# Analyze each domain
domain_analyzer = DomainSpecificSentimentAnalyzer()

print("Domain-Specific Sentiment Analysis Results:")
print("=" * 60)

domains_data = [
    ('financial', financial_texts),
    ('medical', medical_texts), 
    ('technology', technology_texts)
]

for domain, texts in domains_data:
    print(f"\n{domain.upper()} DOMAIN:")
    print("-" * 30)
    
    for text in texts:
        result = domain_analyzer.analyze_domain_sentiment(text, domain)
        print(f"Text: '{text}'")
        print(f"Basic: {result['basic_sentiment']} ({result['basic_polarity']:.3f})")
        print(f"Domain factors: +{result['domain_positive_words']} -{result['domain_negative_words']} words, {result['intensifiers']} intensifiers")
        print(f"Final: {result['final_sentiment']} ({result['final_polarity']:.3f})")
        print()
```

## 💡 Pro Tips for Production Sentiment Analysis

### 1. Real-Time Sentiment Monitoring

```python
import time
from datetime import datetime
import json

class RealTimeSentimentMonitor:
    """Monitor sentiment in real-time for applications"""
    
    def __init__(self, alert_threshold=0.7):
        self.sentiment_analyzer = BasicSentimentAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.alert_threshold = alert_threshold
        self.sentiment_history = []
        
    def process_text_stream(self, texts, source="unknown"):
        """Process a stream of texts and monitor for significant sentiment changes"""
        
        results = []
        alerts = []
        
        for i, text in enumerate(texts):
            timestamp = datetime.now()
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_with_textblob([text]).iloc[0]
            
            # Analyze emotions
            emotion_result = self.emotion_analyzer.analyze_emotions([text])[0]
            
            # Create result record
            result = {
                'timestamp': timestamp,
                'text': text,
                'source': source,
                'sentiment': sentiment_result['sentiment'],
                'polarity': sentiment_result['polarity'],
                'top_emotion': emotion_result['top_emotion'],
                'emotion_confidence': emotion_result['confidence']
            }
            
            results.append(result)
            self.sentiment_history.append(result)
            
            # Check for alerts
            if abs(sentiment_result['polarity']) > self.alert_threshold:
                alert = {
                    'timestamp': timestamp,
                    'type': 'high_sentiment',
                    'sentiment': sentiment_result['sentiment'],
                    'polarity': sentiment_result['polarity'],
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'source': source
                }
                alerts.append(alert)
            
            # Emotion-based alerts
            if emotion_result['top_emotion'] in ['anger', 'fear'] and emotion_result['confidence'] > 0.8:
                alert = {
                    'timestamp': timestamp,
                    'type': 'strong_negative_emotion',
                    'emotion': emotion_result['top_emotion'],
                    'confidence': emotion_result['confidence'],
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'source': source
                }
                alerts.append(alert)
        
        return {
            'results': results,
            'alerts': alerts,
            'summary': self.generate_summary(results)
        }
    
    def generate_summary(self, results):
        """Generate summary statistics from results"""
        if not results:
            return {}
        
        sentiments = [r['sentiment'] for r in results]
        polarities = [r['polarity'] for r in results]
        emotions = [r['top_emotion'] for r in results]
        
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
        emotion_counts = pd.Series(emotions).value_counts().to_dict()
        
        return {
            'total_texts': len(results),
            'sentiment_distribution': sentiment_counts,
            'emotion_distribution': emotion_counts,
            'average_polarity': np.mean(polarities),
            'polarity_std': np.std(polarities),
            'most_common_emotion': max(emotion_counts, key=emotion_counts.get),
            'timestamp_range': {
                'start': min(r['timestamp'] for r in results),
                'end': max(r['timestamp'] for r in results)
            }
        }
    
    def export_results(self, results, filename=None):
        """Export results to JSON file"""
        if filename is None:
            filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings for JSON serialization
        exportable_results = []
        for result in results:
            export_result = result.copy()
            export_result['timestamp'] = result['timestamp'].isoformat()
            exportable_results.append(export_result)
        
        with open(filename, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        print(f"Results exported to {filename}")

# Example: Social Media Monitoring
social_media_posts = [
    "Just tried the new restaurant downtown - absolutely amazing food! 😍",
    "Worst customer service experience ever. So frustrated right now! 😡",
    "Having a great day at the beach with family. Perfect weather! ☀️",
    "Can't believe how rude that employee was. Never shopping here again! 😤",
    "Love the new update to this app. Much easier to use now! 👍",
    "This traffic is making me so anxious. Going to be late for my interview 😰",
    "Just finished a great workout. Feeling energized and happy! 💪",
    "Terrible news about the data breach. Really worried about my information 😨",
    "Beautiful sunset tonight. Sometimes you just need to appreciate nature 🌅",
    "Angry about the price increase. This is getting ridiculous! 😠"
]

# Monitor sentiment in real-time
monitor = RealTimeSentimentMonitor(alert_threshold=0.6)
monitoring_results = monitor.process_text_stream(social_media_posts, source="social_media")

print("Real-Time Sentiment Monitoring Results:")
print("=" * 50)

# Show alerts
if monitoring_results['alerts']:
    print("\n🚨 ALERTS GENERATED:")
    for alert in monitoring_results['alerts']:
        print(f"  {alert['type']}: {alert.get('sentiment', alert.get('emotion', 'N/A'))}")
        print(f"  Text: {alert['text']}")
        print(f"  Time: {alert['timestamp'].strftime('%H:%M:%S')}")
        print()

# Show summary
summary = monitoring_results['summary']
print("📊 SUMMARY STATISTICS:")
print(f"Total texts analyzed: {summary['total_texts']}")
print(f"Average polarity: {summary['average_polarity']:.3f}")
print(f"Most common emotion: {summary['most_common_emotion']}")
print("\nSentiment distribution:")
for sentiment, count in summary['sentiment_distribution'].items():
    print(f"  {sentiment}: {count}")

# Export results
monitor.export_results(monitoring_results['results'])
```

## 🏋️ Practice Exercise

### Build a Comprehensive Review Analysis System

Create a system that can analyze customer reviews across multiple dimensions:

```python
def build_comprehensive_review_analyzer():
    """
    Build a complete review analysis system
    
    Requirements:
    1. Multi-domain aspect-based sentiment analysis
    2. Emotion detection and intensity measurement
    3. Sarcasm detection and handling
    4. Real-time monitoring capabilities
    5. Trend analysis over time
    6. Actionable insights generation
    
    Bonus:
    - Multi-language support
    - Integration with review platforms (APIs)
    - Automated report generation
    - Machine learning model training on custom data
    """
    
    # Your implementation here
    pass

# Test data: Mix of product reviews from different categories
test_reviews = [
    "This laptop is absolutely fantastic! Lightning fast performance and the battery lasts all day. Worth every penny!",
    "The hotel room was spacious and clean, but the service was terrible. Staff was rude and unhelpful throughout our stay.",
    "Oh great, another software update that breaks everything. Thanks for making my phone slower! 🙄",
    "Amazing restaurant experience! The food was incredible, service was attentive, and the atmosphere was perfect for our anniversary.",
    "Worst purchase ever. The product arrived damaged and customer service refused to help. Avoid this company!",
    "The movie was okay, I guess. Not bad but nothing special. Would maybe watch again if there's nothing else on.",
    "LOVE this new skincare product! My skin has never looked better. Highly recommend to everyone!",
    "Disappointing book. The plot was predictable and the characters were one-dimensional. Expected much more from this author."
]

# Your system should provide:
# 1. Overall sentiment analysis
# 2. Aspect-based sentiment breakdown  
# 3. Emotion analysis
# 4. Sarcasm detection
# 5. Domain classification
# 6. Actionable insights for businesses
```

## 💡 Key Takeaways

1. **Sentiment analysis is multi-dimensional** - Beyond positive/negative
2. **Context matters** - Domain, culture, and sarcasm affect interpretation
3. **Aspect-based analysis provides business value** - Know what specifically customers like/dislike
4. **Emotions add depth** - Understanding specific emotions provides richer insights
5. **Real-time monitoring enables quick response** - Address issues before they escalate
6. **Combine multiple techniques** - No single approach handles all cases perfectly

## 🚀 What's Next?

You've mastered sentiment analysis and opinion mining! Next, explore [Word Embeddings and Semantic Similarity](../03_word_embeddings/01_word2vec_fundamentals.md) to understand how computers can learn the meaning of words.

**Coming up:**

- Word2Vec and GloVe embeddings
- Transformer-based embeddings (BERT, GPT)
- Semantic similarity and word analogies
- Building custom embeddings for your domain

Ready to dive into the mathematical representation of meaning? Let's continue the journey!
