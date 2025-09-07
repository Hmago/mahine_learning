# Advanced Text Cleaning: Handling Real-World Mess

## 🎯 What You'll Learn

Real-world text is messier than you think! In this lesson, you'll learn to handle the tricky stuff: emojis, HTML tags, URLs, and special characters that basic cleaning can't handle.

## 🌍 The Real World is Messy

While basic cleaning handles simple cases, real text data comes with:

- **Web scraping results**: Full of HTML tags and entities
- **Social media posts**: Packed with emojis, hashtags, and mentions
- **International text**: Different languages and special characters
- **User-generated content**: Typos, slang, and creative spelling

Let's learn to handle all of this!

## 🔥 Advanced Cleaning Challenges

### 1. HTML Tags and Entities

**The Problem:**
When you scrape text from websites, you get HTML mixed in with your content.

```python
html_text = """
<p>This is a <strong>great</strong> product!</p>
<div>I &amp; my family love it.</div>
<a href="http://example.com">Click here</a> for more info.
"""
```

**The Solution:**

```python
from bs4 import BeautifulSoup
import html

def clean_html(text):
    """Remove HTML tags and decode HTML entities"""
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text_only = soup.get_text()
    
    # Decode HTML entities (&amp; becomes &, &lt; becomes <, etc.)
    text_clean = html.unescape(text_only)
    
    # Clean up extra whitespace
    text_clean = ' '.join(text_clean.split())
    
    return text_clean

# Test it
clean_text = clean_html(html_text)
print(clean_text)
# Output: "This is a great product! I & my family love it. Click here for more info."
```

**Why this matters:** Web scraping is a common way to get text data, and HTML pollution can mess up your analysis.

### 2. URLs and Links

**The Problem:**
URLs add noise and rarely contain useful information for text analysis.

```python
text_with_urls = """
Check out this amazing article: https://www.example.com/amazing-article?id=123
Also see http://bit.ly/shortlink and www.another-site.org for more info!
"""
```

**The Solution:**

```python
import re

def remove_urls(text):
    """Remove URLs from text"""
    
    # Pattern to match various URL formats
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Remove URLs
    text = re.sub(url_pattern, '', text)
    text = re.sub(www_pattern, '', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text

# Test it
clean_text = remove_urls(text_with_urls)
print(clean_text)
# Output: "Check out this amazing article: Also see and for more info!"
```

### 3. Emojis and Special Characters

**The Problem:**
Emojis and special characters can either be meaningful (for sentiment analysis) or noise (for topic modeling).

```python
emoji_text = "I love this product! 😍🔥💯 It's absolutely amazing! ⭐⭐⭐⭐⭐"
```

**Solution 1: Remove All Emojis**

```python
import re

def remove_emojis(text):
    """Remove all emojis from text"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

clean_text = remove_emojis(emoji_text)
print(clean_text)
# Output: "I love this product!  It's absolutely amazing! "
```

**Solution 2: Convert Emojis to Text**

```python
# First install: pip install emoji
import emoji

def emojis_to_text(text):
    """Convert emojis to their text descriptions"""
    return emoji.demojize(text, delimiters=(" ", " "))

descriptive_text = emojis_to_text(emoji_text)
print(descriptive_text)
# Output: "I love this product!  smiling_face_with_heart-eyes fire hundred_points_symbol  It's absolutely amazing!  white_medium_star white_medium_star white_medium_star white_medium_star white_medium_star "
```

### 4. Social Media Specific Cleaning

**The Problem:**
Social media text has @mentions, #hashtags, and RT (retweet) markers.

```python
tweet = "RT @user123: Can't believe how good this movie is! #amazing #mustwatch @everyone should see it! 🔥"
```

**The Solution:**

```python
def clean_social_media(text):
    """Clean social media specific elements"""
    
    # Remove RT (retweet) markers
    text = re.sub(r'^RT\s+', '', text)
    
    # Remove @mentions but keep the word that follows
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove # from hashtags but keep the word
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text

clean_tweet = clean_social_media(tweet)
print(clean_tweet)
# Output: "Can't believe how good this movie is! amazing mustwatch should see it! 🔥"
```

## 🔧 Building a Comprehensive Cleaner

Let's combine everything into one powerful cleaning function:

```python
import re
import string
import html
from bs4 import BeautifulSoup

def advanced_text_cleaner(text, 
                         remove_html=True,
                         remove_urls=True, 
                         remove_emojis=True,
                         remove_social_media_tags=True,
                         lowercase=True,
                         remove_punctuation=True):
    """
    Advanced text cleaner with multiple options
    
    Args:
        text (str): Input text to clean
        remove_html (bool): Remove HTML tags and entities
        remove_urls (bool): Remove URLs
        remove_emojis (bool): Remove emoji characters
        remove_social_media_tags (bool): Remove @mentions and clean #hashtags
        lowercase (bool): Convert to lowercase
        remove_punctuation (bool): Remove punctuation
    
    Returns:
        str: Cleaned text
    """
    
    # Original text
    print(f"Original: {text[:100]}...")
    
    # Remove HTML
    if remove_html:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = html.unescape(text)
        print(f"After HTML removal: {text[:100]}...")
    
    # Remove URLs
    if remove_urls:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = re.sub(www_pattern, '', text)
        print(f"After URL removal: {text[:100]}...")
    
    # Clean social media
    if remove_social_media_tags:
        text = re.sub(r'^RT\s+', '', text)  # Remove RT
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove @mentions
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)  # Keep hashtag words
        print(f"After social media cleaning: {text[:100]}...")
    
    # Remove emojis
    if remove_emojis:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        print(f"After emoji removal: {text[:100]}...")
    
    # Lowercase
    if lowercase:
        text = text.lower()
        print(f"After lowercasing: {text[:100]}...")
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
        print(f"After punctuation removal: {text[:100]}...")
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    print(f"Final result: {text[:100]}...")
    return text

# Test with a complex example
messy_text = """
<p>RT @moviefan: OMG!!! Just watched the new Marvel movie 🔥🔥🔥 
SO GOOD!!! Check it out: https://marvel.com/movies #Marvel #MustWatch 
Rating: 10/10 would recommend! 😍</p>
"""

clean_result = advanced_text_cleaner(messy_text)
```

## 🤔 Choosing What to Clean

Different tasks need different cleaning approaches:

### 📊 For Sentiment Analysis
```python
# Keep emotional indicators but remove noise
sentiment_cleaner = lambda text: advanced_text_cleaner(
    text, 
    remove_emojis=False,  # Keep emojis for emotion
    remove_punctuation=False,  # Keep !!! for emphasis
    remove_social_media_tags=True
)
```

### 📑 For Topic Modeling
```python
# Remove all noise, focus on content words
topic_cleaner = lambda text: advanced_text_cleaner(
    text,
    remove_emojis=True,  # Remove emotional noise
    remove_punctuation=True,  # Focus on words
    remove_social_media_tags=True
)
```

### 🔍 For Search Applications
```python
# Moderate cleaning to preserve some context
search_cleaner = lambda text: advanced_text_cleaner(
    text,
    remove_emojis=True,
    remove_punctuation=False,  # Keep some structure
    lowercase=False  # Preserve proper nouns
)
```

## 🏋️ Practice Exercise

Clean these challenging real-world examples:

```python
# Exercise 1: Web scraped product review
web_review = """
<div class="review">
<h3>Amazing product!</h3>
<p>I &amp; my wife bought this from <a href="https://store.com">here</a>. 
It's <strong>fantastic</strong>! 😍🔥</p>
<p>Rating: ⭐⭐⭐⭐⭐ (5/5)</p>
</div>
"""

# Exercise 2: Social media post
social_post = "RT @techreview: Just tested the new iPhone! 📱✨ #Apple #iPhone #TechReview Amazing camera quality! 🔥📸 Check out my full review: https://bit.ly/iphonereview @everyone should see this!"

# Exercise 3: Email content
email_content = """
Hi there,

Thanks for your email! I'll get back to you ASAP regarding the proposal.

Best regards,
John Smith
📧 john@company.com
🌐 www.company.com
"""

# Your task: Use our advanced_text_cleaner with appropriate settings for each
```

## 💡 Key Takeaways

1. **Real-world text is messy** - Always expect HTML, URLs, emojis, and special characters
2. **One size doesn't fit all** - Different tasks need different cleaning strategies
3. **Test your cleaning** - Always check the output to make sure you're not losing important information
4. **Build modular cleaners** - Create flexible functions that can be customized for different use cases
5. **Consider your end goal** - Clean based on what you plan to do with the text

## 🔗 Quick Reference

```python
# Quick advanced cleaning template
def quick_advanced_clean(text, task_type="general"):
    """Quick cleaner with task-specific presets"""
    
    if task_type == "sentiment":
        return advanced_text_cleaner(text, remove_emojis=False, remove_punctuation=False)
    elif task_type == "topic":
        return advanced_text_cleaner(text, remove_emojis=True, remove_punctuation=True)
    elif task_type == "search":
        return advanced_text_cleaner(text, lowercase=False, remove_punctuation=False)
    else:
        return advanced_text_cleaner(text)  # Default settings
```

Ready to learn about breaking text into pieces? Continue to [Tokenization Strategies](./03_tokenization.md)!
