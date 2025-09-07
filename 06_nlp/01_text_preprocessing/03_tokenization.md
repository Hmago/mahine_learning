# Tokenization: Breaking Text into Meaningful Pieces

## 🎯 What You'll Learn

Tokenization is like being a skilled surgeon who knows exactly where to make cuts. You'll learn how to split text into words, sentences, and even smaller pieces in the most meaningful way possible.

## 🧩 What is Tokenization?

Imagine you have a long sentence written without any spaces:

`"Thisisasentencewrittenwithoutspaces"`

Your brain can probably figure it out, but it's much easier to read when properly separated:

`"This is a sentence written without spaces"`

Tokenization does exactly this - it breaks text into individual units (called "tokens") that make sense for analysis.

## 🔍 Types of Tokenization

### 1. Word Tokenization: The Foundation

**What it does:** Splits text into individual words.

```python
text = "Hello world! How are you today?"
tokens = text.split()
print(tokens)
# Output: ['Hello', 'world!', 'How', 'are', 'you', 'today?']
```

**The Problem with Simple Splitting:**
Notice that punctuation is still attached to words (`'world!'`, `'today?'`). This might not be what we want.

**Better Word Tokenization:**

```python
import nltk
from nltk.tokenize import word_tokenize

# Download required data (run once)
nltk.download('punkt')

text = "Hello world! How are you today?"
tokens = word_tokenize(text)
print(tokens)
# Output: ['Hello', 'world', '!', 'How', 'are', 'you', 'today', '?']
```

**Real-world analogy:** It's like having a smart assistant who knows that "Dr. Smith" is one unit, but "Mr. Johnson!" should be split into "Mr.", "Johnson", and "!".

### 2. Sentence Tokenization: Finding Natural Breaks

**What it does:** Splits text into individual sentences.

```python
from nltk.tokenize import sent_tokenize

text = "Hello world! How are you today? I hope you're doing well."
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Hello world!', 'How are you today?', "I hope you're doing well."]
```

**Why it's tricky:** Consider this text:
```python
tricky_text = "Dr. Smith went to the U.S.A. He likes it there."
sentences = sent_tokenize(tricky_text)
print(sentences)
# Output: ['Dr. Smith went to the U.S.A.', 'He likes it there.']
# Notice it correctly identified that "U.S.A." doesn't end a sentence!
```

### 3. Subword Tokenization: Handling Unknown Words

**The Problem:** What if you encounter a word your model has never seen before?

```python
# Your model knows these words:
known_words = ["play", "ing", "un", "happy"]

# But encounters this new word:
new_word = "replaying"

# Subword tokenization breaks it into known pieces:
# "replaying" → ["re", "play", "ing"]
```

**Byte Pair Encoding (BPE) Example:**

```python
# This is a simplified example - real BPE is more complex
def simple_bpe_example(word, vocab):
    """Simple demonstration of subword tokenization"""
    tokens = []
    i = 0
    while i < len(word):
        # Try to find the longest known subword starting at position i
        found = False
        for length in range(len(word) - i, 0, -1):
            subword = word[i:i+length]
            if subword in vocab:
                tokens.append(subword)
                i += length
                found = True
                break
        if not found:
            tokens.append(word[i])  # Add single character if nothing found
            i += 1
    return tokens

vocab = ["play", "ing", "re", "un", "happy", "ed"]
word = "replaying"
tokens = simple_bpe_example(word, vocab)
print(f"'{word}' → {tokens}")
# Output: 'replaying' → ['re', 'play', 'ing']
```

## 🛠 Practical Tokenization Examples

### Example 1: Basic Text Processing

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def analyze_text(text):
    """Analyze text using different tokenization methods"""
    
    print(f"Original text: {text}")
    print("-" * 50)
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")
    for i, sentence in enumerate(sentences, 1):
        print(f"  Sentence {i}: {sentence}")
    
    print()
    
    # Word tokenization
    words = word_tokenize(text)
    print(f"Number of words: {len(words)}")
    print(f"Words: {words}")
    
    print()
    
    # Word tokenization without punctuation
    words_no_punct = [word for word in words if word.isalnum()]
    print(f"Words (no punctuation): {words_no_punct}")

# Test it
text = "Hello Dr. Smith! How are you? I hope you're having a great day."
analyze_text(text)
```

### Example 2: Handling Different Languages

```python
# English
english_text = "Hello world! How are you?"
english_tokens = word_tokenize(english_text)
print(f"English: {english_tokens}")

# French (requires additional setup)
french_text = "Bonjour le monde! Comment allez-vous?"
french_tokens = word_tokenize(french_text, language='french')
print(f"French: {french_tokens}")

# Spanish
spanish_text = "¡Hola mundo! ¿Cómo estás?"
spanish_tokens = word_tokenize(spanish_text, language='spanish')
print(f"Spanish: {spanish_tokens}")
```

### Example 3: Custom Tokenization

Sometimes you need special rules for your specific domain:

```python
import re

def custom_tokenizer(text, preserve_hashtags=True, preserve_mentions=True):
    """Custom tokenizer for social media text"""
    
    # First, let's identify special tokens we want to preserve
    hashtag_pattern = r'#\w+'
    mention_pattern = r'@\w+'
    
    # Find all hashtags and mentions
    hashtags = re.findall(hashtag_pattern, text) if preserve_hashtags else []
    mentions = re.findall(mention_pattern, text) if preserve_mentions else []
    
    # Replace them with placeholders
    temp_text = text
    hashtag_placeholders = {}
    mention_placeholders = {}
    
    for i, hashtag in enumerate(hashtags):
        placeholder = f"__HASHTAG_{i}__"
        hashtag_placeholders[placeholder] = hashtag
        temp_text = temp_text.replace(hashtag, placeholder, 1)
    
    for i, mention in enumerate(mentions):
        placeholder = f"__MENTION_{i}__"
        mention_placeholders[placeholder] = mention
        temp_text = temp_text.replace(mention, placeholder, 1)
    
    # Tokenize normally
    tokens = word_tokenize(temp_text)
    
    # Replace placeholders back
    final_tokens = []
    for token in tokens:
        if token in hashtag_placeholders:
            final_tokens.append(hashtag_placeholders[token])
        elif token in mention_placeholders:
            final_tokens.append(mention_placeholders[token])
        else:
            final_tokens.append(token)
    
    return final_tokens

# Test custom tokenizer
social_text = "Love this movie! #amazing @moviefan check it out!"
custom_tokens = custom_tokenizer(social_text)
print(f"Custom tokens: {custom_tokens}")

# Compare with regular tokenization
regular_tokens = word_tokenize(social_text)
print(f"Regular tokens: {regular_tokens}")
```

## 🚨 Common Tokenization Challenges

### 1. Contractions

```python
text = "I'm happy, but I can't go. We'll see what happens."
tokens = word_tokenize(text)
print(tokens)
# Output: ['I', "'m", 'happy', ',', 'but', 'I', 'ca', "n't", 'go', '.', 'We', "'ll", 'see', 'what', 'happens', '.']

# Notice how contractions are split - this might or might not be what you want
```

**Handling Contractions:**

```python
import re

def expand_contractions(text):
    """Expand common English contractions"""
    contractions = {
        "I'm": "I am",
        "can't": "cannot",
        "won't": "will not",
        "we'll": "we will",
        "you're": "you are",
        "it's": "it is",
        "they're": "they are",
        "we're": "we are",
        "I've": "I have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "I'll": "I will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "it'll": "it will",
        "we're": "we are",
        "they'll": "they will"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
        text = text.replace(contraction.lower(), expansion)
    
    return text

text = "I'm happy, but I can't go. We'll see what happens."
expanded = expand_contractions(text)
print(f"Expanded: {expanded}")
tokens = word_tokenize(expanded)
print(f"Tokens: {tokens}")
```

### 2. Handling Numbers and Dates

```python
text = "On 12/25/2023, I bought 3 items for $29.99 each."
tokens = word_tokenize(text)
print(tokens)
# Output: ['On', '12/25/2023', ',', 'I', 'bought', '3', 'items', 'for', '$', '29.99', 'each', '.']

# Notice that dates and prices are partially split
```

### 3. Domain-Specific Text

```python
# Medical text
medical_text = "Patient shows signs of COVID-19. Temperature: 101.5°F."
medical_tokens = word_tokenize(medical_text)
print(f"Medical: {medical_tokens}")

# Technical text
tech_text = "The API returns a JSON object with user_id and email_address."
tech_tokens = word_tokenize(tech_text)
print(f"Technical: {tech_tokens}")

# Legal text
legal_text = "As per Section 3.1.4 of the aforementioned agreement..."
legal_tokens = word_tokenize(legal_text)
print(f"Legal: {legal_tokens}")
```

## 🔧 Building a Comprehensive Tokenizer

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re

class SmartTokenizer:
    """A flexible tokenizer with multiple options"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def tokenize_sentences(self, text):
        """Split text into sentences"""
        return sent_tokenize(text)
    
    def tokenize_words(self, text, 
                      expand_contractions=True,
                      remove_punctuation=True,
                      preserve_hashtags=True,
                      preserve_mentions=True,
                      lowercase=True):
        """
        Flexible word tokenization with multiple options
        """
        
        # Expand contractions if requested
        if expand_contractions:
            text = self._expand_contractions(text)
        
        # Handle special social media tokens
        if preserve_hashtags or preserve_mentions:
            text, replacements = self._preserve_special_tokens(
                text, preserve_hashtags, preserve_mentions
            )
        else:
            replacements = {}
        
        # Basic tokenization
        tokens = word_tokenize(text)
        
        # Remove punctuation if requested
        if remove_punctuation:
            tokens = [token for token in tokens if token.isalnum() or token in replacements]
        
        # Restore special tokens
        if replacements:
            tokens = [replacements.get(token, token) for token in tokens]
        
        # Lowercase if requested
        if lowercase:
            tokens = [token.lower() if not (token.startswith('#') or token.startswith('@')) 
                     else token for token in tokens]
        
        return tokens
    
    def _expand_contractions(self, text):
        """Expand common contractions"""
        contractions = {
            r"I'm": "I am", r"can't": "cannot", r"won't": "will not",
            r"we'll": "we will", r"you're": "you are", r"it's": "it is",
            r"they're": "they are", r"we're": "we are", r"I've": "I have",
            r"you've": "you have", r"we've": "we have", r"they've": "they have"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _preserve_special_tokens(self, text, preserve_hashtags, preserve_mentions):
        """Preserve hashtags and mentions during tokenization"""
        replacements = {}
        counter = 0
        
        if preserve_hashtags:
            hashtags = re.findall(r'#\w+', text)
            for hashtag in hashtags:
                placeholder = f"__SPECIAL_{counter}__"
                replacements[placeholder] = hashtag
                text = text.replace(hashtag, placeholder, 1)
                counter += 1
        
        if preserve_mentions:
            mentions = re.findall(r'@\w+', text)
            for mention in mentions:
                placeholder = f"__SPECIAL_{counter}__"
                replacements[placeholder] = mention
                text = text.replace(mention, placeholder, 1)
                counter += 1
        
        # Create reverse mapping for restoration
        reverse_replacements = {v: k for k, v in replacements.items()}
        
        return text, reverse_replacements

# Example usage
tokenizer = SmartTokenizer()

# Test text
text = "I'm excited about this #AI project! @teamlead, can't wait to share results."

# Different tokenization strategies
basic_tokens = tokenizer.tokenize_words(text, 
                                      expand_contractions=False,
                                      preserve_hashtags=False,
                                      preserve_mentions=False)
print(f"Basic: {basic_tokens}")

advanced_tokens = tokenizer.tokenize_words(text,
                                         expand_contractions=True,
                                         preserve_hashtags=True,
                                         preserve_mentions=True)
print(f"Advanced: {advanced_tokens}")
```

## 🏋️ Practice Exercises

Try tokenizing these challenging examples:

```python
# Exercise 1: Scientific text
scientific_text = "The COVID-19 virus (SARS-CoV-2) has a reproduction rate (R₀) of 2.5-3.0."

# Exercise 2: Social media
social_text = "OMG! Just saw #Avengers! @Marvel you've outdone yourselves! 🔥 Can't wait for the sequel..."

# Exercise 3: Technical documentation
tech_text = "The API endpoint /api/v1/users returns user_data in JSON format. Error codes: 404, 500."

# Exercise 4: Legal text
legal_text = "Whereas Party A agrees to Section 2.1.3, and Party B accepts Terms & Conditions..."

# Your task: Use different tokenization strategies for each type of text
```

## 💡 Choosing the Right Tokenization Strategy

### For Social Media Analysis:
- Preserve hashtags and mentions
- Expand contractions for consistency
- Keep emoticons if analyzing sentiment

### For Academic/Scientific Text:
- Preserve technical terms and abbreviations
- Handle special characters (Greek letters, subscripts)
- Keep numbers and units together

### For Search Applications:
- More conservative tokenization
- Preserve proper nouns
- Keep punctuation that affects meaning

### For Machine Learning:
- Consistent tokenization across training and inference
- Consider subword tokenization for unknown words
- Balance between granularity and vocabulary size

## 💡 Key Takeaways

1. **Tokenization is not one-size-fits-all** - Different tasks need different strategies
2. **Consider your domain** - Social media, academic, and technical texts have different needs
3. **Test with real data** - Always check how your tokenization works with actual data from your domain
4. **Think about downstream tasks** - How you tokenize affects everything that comes after
5. **Handle edge cases** - Contractions, hashtags, technical terms, and special characters need special attention

Ready to learn about normalizing your tokens? Continue to [Text Normalization](./04_text_normalization.md)!
