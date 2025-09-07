# Word2Vec Fundamentals: Learning Word Relationships

## 🎯 What You'll Learn

Word2Vec revolutionized NLP by showing that computers can learn meaningful word relationships just by analyzing how words appear together in text. You'll master both the theory and practical implementation of this groundbreaking technique.

## 🧠 The Big Idea Behind Word2Vec

**The fundamental insight:** "You shall know a word by the company it keeps" - J.R. Firth

Word2Vec learns word meanings by looking at context. If two words appear in similar contexts, they probably have similar meanings.

**Think of it like this:** Imagine you're learning a new language by watching how words are used:
- Words that appear near "delicious" → probably foods or positive descriptions
- Words that appear near "quickly" → probably actions or movements
- Words that appear near "Paris" → probably places, travel, or French things

## 🔧 The Two Architectures

### 1. Skip-gram: Predict Context from Word

**Goal:** Given a word, predict the words that appear around it.

**Think of it as:** A word trying to guess its neighbors.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

class SimpleWord2Vec:
    """A simplified implementation to understand Word2Vec internals"""
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=5, learning_rate=0.01):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Model parameters
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        
        # Weight matrices
        self.W_input = None   # Input layer weights
        self.W_output = None  # Output layer weights
        
        # Training data
        self.training_pairs = []
        
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        
        print("Building vocabulary...")
        word_counts = Counter()
        
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
        
        # Filter by minimum count
        filtered_words = [word for word, count in word_counts.items() if count >= self.min_count]
        
        # Create word mappings
        self.word_to_index = {word: idx for idx, word in enumerate(filtered_words)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.vocab_size = len(filtered_words)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {list(word_counts.most_common(10))}")
        
        return self.word_to_index
    
    def generate_training_data(self, sentences):
        """Generate (target, context) pairs for skip-gram training"""
        
        print("Generating training data...")
        self.training_pairs = []
        
        for sentence in sentences:
            # Convert sentence to indices
            sentence_indices = [self.word_to_index[word] for word in sentence 
                              if word in self.word_to_index]
            
            # Generate training pairs
            for i, target_idx in enumerate(sentence_indices):
                # Define context window
                start = max(0, i - self.window)
                end = min(len(sentence_indices), i + self.window + 1)
                
                # Generate pairs with context words
                for j in range(start, end):
                    if i != j:  # Don't include the target word itself
                        context_idx = sentence_indices[j]
                        self.training_pairs.append((target_idx, context_idx))
        
        print(f"Generated {len(self.training_pairs)} training pairs")
        return self.training_pairs
    
    def initialize_weights(self):
        """Initialize weight matrices randomly"""
        
        # Xavier initialization
        bound = np.sqrt(6.0 / (self.vocab_size + self.vector_size))
        
        self.W_input = np.random.uniform(-bound, bound, (self.vocab_size, self.vector_size))
        self.W_output = np.random.uniform(-bound, bound, (self.vector_size, self.vocab_size))
        
        print("Initialized weight matrices")
        print(f"Input weights shape: {self.W_input.shape}")
        print(f"Output weights shape: {self.W_output.shape}")
    
    def softmax(self, x):
        """Softmax function with numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward_pass(self, target_idx):
        """Forward pass through the network"""
        
        # Get input word vector (one-hot encoded)
        input_vector = self.W_input[target_idx]  # Shape: (vector_size,)
        
        # Calculate output scores
        output_scores = np.dot(input_vector, self.W_output)  # Shape: (vocab_size,)
        
        # Apply softmax
        output_probs = self.softmax(output_scores)
        
        return input_vector, output_scores, output_probs
    
    def backward_pass(self, target_idx, context_idx, input_vector, output_probs):
        """Backward pass to update weights"""
        
        # Calculate error
        error = output_probs.copy()
        error[context_idx] -= 1  # Subtract 1 from the true class
        
        # Calculate gradients
        grad_output = np.outer(input_vector, error)  # Gradient for output weights
        grad_input = np.dot(self.W_output, error)    # Gradient for input weights
        
        # Update weights
        self.W_output -= self.learning_rate * grad_output
        self.W_input[target_idx] -= self.learning_rate * grad_input
        
        # Calculate loss (cross-entropy)
        loss = -np.log(output_probs[context_idx] + 1e-10)  # Add small epsilon to avoid log(0)
        
        return loss
    
    def train(self, sentences):
        """Train the Word2Vec model"""
        
        # Build vocabulary and generate training data
        self.build_vocabulary(sentences)
        self.generate_training_data(sentences)
        self.initialize_weights()
        
        print(f"\nStarting training for {self.epochs} epochs...")
        
        total_loss_history = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            # Shuffle training pairs
            random.shuffle(self.training_pairs)
            
            for i, (target_idx, context_idx) in enumerate(self.training_pairs):
                # Forward pass
                input_vector, output_scores, output_probs = self.forward_pass(target_idx)
                
                # Backward pass
                loss = self.backward_pass(target_idx, context_idx, input_vector, output_probs)
                epoch_loss += loss
                
                # Print progress
                if i % 1000 == 0 and i > 0:
                    avg_loss = epoch_loss / (i + 1)
                    print(f"Epoch {epoch+1}, Step {i}, Average Loss: {avg_loss:.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.training_pairs)
            total_loss_history.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        print("Training completed!")
        return total_loss_history
    
    def get_word_vector(self, word):
        """Get the vector representation of a word"""
        if word not in self.word_to_index:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_idx = self.word_to_index[word]
        return self.W_input[word_idx]
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        
        try:
            vec1 = self.get_word_vector(word1)
            vec2 = self.get_word_vector(word2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                return 0
            
            return dot_product / norms
        
        except ValueError as e:
            print(f"Error calculating similarity: {e}")
            return 0
    
    def most_similar(self, word, top_n=5):
        """Find the most similar words to a given word"""
        
        if word not in self.word_to_index:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        target_vector = self.get_word_vector(word)
        similarities = []
        
        for other_word in self.word_to_index:
            if other_word != word:
                sim = self.similarity(word, other_word)
                similarities.append((other_word, sim))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def analogy(self, word_a, word_b, word_c, top_n=1):
        """Solve word analogies: word_a is to word_b as word_c is to ?"""
        
        try:
            vec_a = self.get_word_vector(word_a)
            vec_b = self.get_word_vector(word_b)
            vec_c = self.get_word_vector(word_c)
            
            # Calculate the analogy vector: b - a + c
            analogy_vector = vec_b - vec_a + vec_c
            
            # Find the most similar word to this vector
            similarities = []
            for word in self.word_to_index:
                if word not in [word_a, word_b, word_c]:
                    word_vector = self.get_word_vector(word)
                    
                    # Cosine similarity with analogy vector
                    dot_product = np.dot(analogy_vector, word_vector)
                    norms = np.linalg.norm(analogy_vector) * np.linalg.norm(word_vector)
                    
                    if norms > 0:
                        sim = dot_product / norms
                        similarities.append((word, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
        
        except ValueError as e:
            print(f"Error solving analogy: {e}")
            return []

# Example: Training Word2Vec on a small corpus
sample_sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "pets"],
    ["i", "love", "my", "cat", "and", "dog"],
    ["the", "park", "has", "many", "trees"],
    ["trees", "provide", "shade", "in", "the", "park"],
    ["my", "pet", "cat", "likes", "to", "sleep"],
    ["dogs", "like", "to", "run", "and", "play"],
    ["the", "mat", "is", "soft", "and", "warm"],
    ["pets", "need", "love", "and", "care"]
] * 50  # Repeat for more training data

# Create and train the model
print("=== Training Simple Word2Vec Model ===")
model = SimpleWord2Vec(vector_size=50, window=2, epochs=10, learning_rate=0.1)
loss_history = model.train(sample_sentences)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Word2Vec Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()

# Test word similarities
print("\n=== Word Similarities ===")
test_words = ["cat", "dog", "park", "tree"]

for word in test_words:
    if word in model.word_to_index:
        similar_words = model.most_similar(word, top_n=3)
        print(f"Words similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")
        print()

# Test analogies
print("=== Word Analogies ===")
analogies = [
    ("cat", "cats", "dog"),  # cat:cats as dog:?
    ("park", "trees", "home")  # park:trees as home:?
]

for word_a, word_b, word_c in analogies:
    result = model.analogy(word_a, word_b, word_c, top_n=3)
    print(f"{word_a} is to {word_b} as {word_c} is to:")
    for word, score in result:
        print(f"  {word}: {score:.3f}")
    print()
```

### 2. CBOW (Continuous Bag of Words): Predict Word from Context

**Goal:** Given surrounding words, predict the center word.

**Think of it as:** Context words collaborating to guess the missing word.

```python
class CBOWModel:
    """Continuous Bag of Words implementation"""
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=5, learning_rate=0.01):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Model parameters
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        
        # Weight matrices
        self.W_input = None   # Context embeddings
        self.W_output = None  # Output weights
        
        # Training data
        self.training_data = []
    
    def generate_cbow_data(self, sentences):
        """Generate (context_words, target_word) pairs for CBOW training"""
        
        print("Generating CBOW training data...")
        self.training_data = []
        
        for sentence in sentences:
            # Convert to indices
            sentence_indices = [self.word_to_index[word] for word in sentence 
                              if word in self.word_to_index]
            
            # Generate CBOW training examples
            for i in range(len(sentence_indices)):
                target_idx = sentence_indices[i]
                
                # Collect context words
                context_indices = []
                start = max(0, i - self.window)
                end = min(len(sentence_indices), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:  # Exclude target word
                        context_indices.append(sentence_indices[j])
                
                if len(context_indices) > 0:
                    self.training_data.append((context_indices, target_idx))
        
        print(f"Generated {len(self.training_data)} CBOW training examples")
        return self.training_data
    
    def forward_pass_cbow(self, context_indices):
        """Forward pass for CBOW model"""
        
        # Average the context word vectors
        context_vectors = self.W_input[context_indices]  # Shape: (context_size, vector_size)
        avg_context = np.mean(context_vectors, axis=0)   # Shape: (vector_size,)
        
        # Calculate output scores
        output_scores = np.dot(avg_context, self.W_output)  # Shape: (vocab_size,)
        output_probs = self.softmax(output_scores)
        
        return avg_context, output_scores, output_probs
    
    def train_cbow(self, sentences):
        """Train CBOW model"""
        
        # Use the same vocabulary building as skip-gram
        self.build_vocabulary(sentences)
        self.generate_cbow_data(sentences)
        self.initialize_weights()
        
        print(f"\nTraining CBOW model for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            random.shuffle(self.training_data)
            
            for i, (context_indices, target_idx) in enumerate(self.training_data):
                # Forward pass
                avg_context, output_scores, output_probs = self.forward_pass_cbow(context_indices)
                
                # Calculate loss
                loss = -np.log(output_probs[target_idx] + 1e-10)
                epoch_loss += loss
                
                # Backward pass
                error = output_probs.copy()
                error[target_idx] -= 1
                
                # Update output weights
                grad_output = np.outer(avg_context, error)
                self.W_output -= self.learning_rate * grad_output
                
                # Update input weights (distribute gradient across context words)
                grad_input = np.dot(self.W_output, error)
                for ctx_idx in context_indices:
                    self.W_input[ctx_idx] -= self.learning_rate * grad_input / len(context_indices)
                
                if i % 1000 == 0 and i > 0:
                    avg_loss = epoch_loss / (i + 1)
                    print(f"Epoch {epoch+1}, Step {i}, Average Loss: {avg_loss:.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.training_data)
            print(f"CBOW Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        print("CBOW training completed!")
    
    # Inherit similarity and analogy methods from SimpleWord2Vec
    def similarity(self, word1, word2):
        return SimpleWord2Vec.similarity(self, word1, word2)
    
    def most_similar(self, word, top_n=5):
        return SimpleWord2Vec.most_similar(self, word, top_n)
    
    def get_word_vector(self, word):
        return SimpleWord2Vec.get_word_vector(self, word)
    
    def build_vocabulary(self, sentences):
        return SimpleWord2Vec.build_vocabulary(self, sentences)
    
    def initialize_weights(self):
        return SimpleWord2Vec.initialize_weights(self)
    
    def softmax(self, x):
        return SimpleWord2Vec.softmax(self, x)

# Compare Skip-gram vs CBOW
print("\n" + "="*60)
print("COMPARING SKIP-GRAM VS CBOW")
print("="*60)

# Train CBOW model
cbow_model = CBOWModel(vector_size=50, window=2, epochs=5, learning_rate=0.1)
cbow_model.train_cbow(sample_sentences)

# Compare similarities
print("\nSimilarity Comparison:")
test_word = "cat"
if test_word in model.word_to_index and test_word in cbow_model.word_to_index:
    print(f"\nSkip-gram similarities for '{test_word}':")
    sg_similar = model.most_similar(test_word, top_n=3)
    for word, sim in sg_similar:
        print(f"  {word}: {sim:.3f}")
    
    print(f"\nCBOW similarities for '{test_word}':")
    cbow_similar = cbow_model.most_similar(test_word, top_n=3)
    for word, sim in cbow_similar:
        print(f"  {word}: {sim:.3f}")
```

## 🎯 Using Gensim for Production Word2Vec

While our implementation helps understand the internals, for real applications, use Gensim:

```python
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging

# Enable logging to see training progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class ProductionWord2Vec:
    """Production-ready Word2Vec using Gensim"""
    
    def __init__(self, vector_size=300, window=5, min_count=5, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.phrases = None
        
    def preprocess_sentences(self, sentences):
        """Advanced preprocessing including phrase detection"""
        
        # Detect common phrases (e.g., "New_York", "machine_learning")
        phrases = Phrases(sentences, min_count=5, threshold=10)
        phraser = Phraser(phrases)
        self.phrases = phraser
        
        # Apply phrase detection
        processed_sentences = []
        for sentence in sentences:
            # Convert to lowercase and apply phrases
            processed = phraser[sentence]
            processed_sentences.append(processed)
        
        print(f"Processed {len(processed_sentences)} sentences")
        print(f"Sample processed sentence: {processed_sentences[0][:10]}")
        
        return processed_sentences
    
    def train_model(self, sentences, sg=1):
        """Train Word2Vec model with Gensim
        
        Args:
            sentences: List of tokenized sentences
            sg: 1 for skip-gram, 0 for CBOW
        """
        
        print(f"Training {'Skip-gram' if sg else 'CBOW'} model...")
        
        # Preprocess sentences
        processed_sentences = self.preprocess_sentences(sentences)
        
        # Initialize and train model
        self.model = Word2Vec(
            sentences=processed_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=sg,  # 1 = skip-gram, 0 = CBOW
            epochs=self.epochs,
            alpha=0.025,  # Initial learning rate
            min_alpha=0.0001  # Final learning rate
        )
        
        print("Training completed!")
        print(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
        
        return self.model
    
    def find_similar_words(self, word, top_n=10):
        """Find words most similar to the given word"""
        
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        try:
            similar_words = self.model.wv.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
    
    def calculate_similarity(self, word1, word2):
        """Calculate similarity between two words"""
        
        try:
            similarity = self.model.wv.similarity(word1, word2)
            return similarity
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return 0
    
    def solve_analogy(self, positive, negative, top_n=1):
        """Solve word analogies using vector arithmetic
        
        Example: king - man + woman = queen
        positive = ['king', 'woman']
        negative = ['man']
        """
        
        try:
            result = self.model.wv.most_similar(
                positive=positive,
                negative=negative,
                topn=top_n
            )
            return result
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return []
    
    def get_word_vector(self, word):
        """Get the vector representation of a word"""
        
        try:
            return self.model.wv[word]
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return None
    
    def evaluate_model(self):
        """Evaluate model on standard word analogy tasks"""
        
        # Some standard analogies for evaluation
        analogies = [
            # Semantic analogies
            (['king', 'woman'], ['man'], 'queen'),
            (['paris', 'italy'], ['france'], 'rome'),
            (['big', 'biggest'], ['small'], 'smallest'),
            
            # Syntactic analogies  
            (['walk', 'walking'], ['swim'], 'swimming'),
            (['good', 'better'], ['bad'], 'worse'),
            (['cat', 'cats'], ['dog'], 'dogs')
        ]
        
        correct = 0
        total = 0
        
        print("Evaluating model on analogy tasks:")
        print("-" * 50)
        
        for positive, negative, expected in analogies:
            result = self.solve_analogy(positive, negative, top_n=1)
            
            if result:
                predicted = result[0][0]
                confidence = result[0][1]
                
                is_correct = predicted.lower() == expected.lower()
                status = "✓" if is_correct else "✗"
                
                print(f"{status} {' + '.join(positive)} - {' + '.join(negative)} = {predicted} "
                      f"(expected: {expected}, confidence: {confidence:.3f})")
                
                if is_correct:
                    correct += 1
                total += 1
            else:
                print(f"✗ Could not solve: {' + '.join(positive)} - {' + '.join(negative)}")
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nAnalogy accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return accuracy
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = Word2Vec.load(filepath)
        print(f"Model loaded from {filepath}")

# Example: Training on a larger corpus
import re

def create_larger_corpus():
    """Create a larger, more realistic corpus for training"""
    
    # Simulated sentences from different domains
    sentences = []
    
    # Technology sentences
    tech_sentences = [
        "artificial intelligence machine learning deep learning neural networks",
        "python programming language software development coding algorithms",
        "computer science data structures algorithms complexity analysis",
        "web development html css javascript frontend backend",
        "database management sql nosql mongodb postgresql",
        "cloud computing aws azure google cloud platform",
        "mobile development android ios react native flutter",
        "cybersecurity encryption authentication authorization security"
    ]
    
    # Business sentences
    business_sentences = [
        "business strategy marketing sales revenue profit growth",
        "financial management accounting budgeting investment portfolio",
        "human resources recruitment training employee development",
        "project management agile scrum methodology planning execution",
        "customer service support satisfaction feedback improvement",
        "market research analysis competition strategy positioning",
        "entrepreneurship startup venture capital funding investment",
        "supply chain logistics operations management efficiency"
    ]
    
    # Science sentences
    science_sentences = [
        "scientific research methodology experiment hypothesis analysis",
        "physics chemistry biology mathematics statistics probability",
        "medical healthcare treatment diagnosis patient therapy",
        "environmental science climate change sustainability energy",
        "psychology cognitive behavior mental health therapy",
        "genetics dna rna protein molecular biology research",
        "astronomy space exploration planets stars galaxies",
        "engineering mechanical electrical civil software systems"
    ]
    
    # Combine and expand
    all_domains = tech_sentences + business_sentences + science_sentences
    
    for sentence in all_domains:
        # Create variations
        words = sentence.split()
        sentences.append(words)
        
        # Create variations by shuffling and adding context
        for _ in range(10):
            # Add some context words
            context_words = ["the", "and", "of", "in", "to", "for", "with", "by"]
            extended = words + random.choices(context_words, k=3)
            random.shuffle(extended)
            sentences.append(extended)
    
    print(f"Created corpus with {len(sentences)} sentences")
    return sentences

# Train production model
print("\n" + "="*60)
print("TRAINING PRODUCTION WORD2VEC MODEL")
print("="*60)

# Create training corpus
training_corpus = create_larger_corpus()

# Train Skip-gram model
sg_trainer = ProductionWord2Vec(vector_size=100, window=5, min_count=2, epochs=20)
sg_model = sg_trainer.train_model(training_corpus, sg=1)

# Train CBOW model for comparison
cbow_trainer = ProductionWord2Vec(vector_size=100, window=5, min_count=2, epochs=20)
cbow_model = cbow_trainer.train_model(training_corpus, sg=0)

# Evaluate both models
print("\nSkip-gram Model Evaluation:")
sg_accuracy = sg_trainer.evaluate_model()

print("\nCBOW Model Evaluation:")
cbow_accuracy = cbow_trainer.evaluate_model()

# Compare specific similarities
print("\n" + "="*50)
print("SIMILARITY COMPARISON")
print("="*50)

test_words = ["intelligence", "programming", "business", "science"]

for word in test_words:
    print(f"\nWords similar to '{word}':")
    
    # Skip-gram similarities
    sg_similar = sg_trainer.find_similar_words(word, top_n=5)
    print("Skip-gram:")
    for sim_word, similarity in sg_similar:
        print(f"  {sim_word}: {similarity:.3f}")
    
    # CBOW similarities
    cbow_similar = cbow_trainer.find_similar_words(word, top_n=5)
    print("CBOW:")
    for sim_word, similarity in cbow_similar:
        print(f"  {sim_word}: {similarity:.3f}")
```

## 🔍 Visualizing Word Embeddings

Understanding embeddings through visualization:

```python
def visualize_embeddings(model, words_to_plot=None, method='tsne'):
    """Visualize word embeddings in 2D space"""
    
    if words_to_plot is None:
        # Select most frequent words
        word_freq = [(word, model.wv.get_vecattr(word, "count")) 
                     for word in model.wv.key_to_index.keys()]
        word_freq.sort(key=lambda x: x[1], reverse=True)
        words_to_plot = [word for word, freq in word_freq[:50]]  # Top 50 words
    
    # Get word vectors
    word_vectors = []
    labels = []
    
    for word in words_to_plot:
        if word in model.wv:
            word_vectors.append(model.wv[word])
            labels.append(word)
    
    word_vectors = np.array(word_vectors)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(word_vectors)-1))
    
    vectors_2d = reducer.fit_transform(word_vectors)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                         c=range(len(labels)), cmap='tab20', alpha=0.7)
    
    # Add word labels
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'Word Embeddings Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return vectors_2d, labels

def analyze_word_clusters(model, categories):
    """Analyze how well words cluster by semantic category"""
    
    all_similarities = []
    
    for category, words in categories.items():
        print(f"\nAnalyzing '{category}' category:")
        
        # Calculate pairwise similarities within category
        category_similarities = []
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i < j and word1 in model.wv and word2 in model.wv:
                    sim = model.wv.similarity(word1, word2)
                    category_similarities.append(sim)
                    print(f"  {word1} ↔ {word2}: {sim:.3f}")
        
        if category_similarities:
            avg_similarity = np.mean(category_similarities)
            print(f"  Average within-category similarity: {avg_similarity:.3f}")
            all_similarities.extend(category_similarities)
    
    overall_avg = np.mean(all_similarities) if all_similarities else 0
    print(f"\nOverall average within-category similarity: {overall_avg:.3f}")
    
    return all_similarities

# Visualize our trained models
print("\n" + "="*50)
print("VISUALIZING EMBEDDINGS")
print("="*50)

# Visualize Skip-gram embeddings
print("Visualizing Skip-gram embeddings...")
sg_vectors_2d, sg_labels = visualize_embeddings(sg_model, method='tsne')

# Analyze semantic clusters
semantic_categories = {
    'technology': ['intelligence', 'programming', 'computer', 'software', 'algorithm'],
    'business': ['business', 'marketing', 'financial', 'management', 'revenue'],
    'science': ['research', 'analysis', 'experiment', 'medical', 'biology'],
    'education': ['training', 'learning', 'development', 'methodology', 'study']
}

print("\nAnalyzing semantic clusters:")
cluster_similarities = analyze_word_clusters(sg_model, semantic_categories)

# Create similarity heatmap
def create_similarity_heatmap(model, words):
    """Create a heatmap of word similarities"""
    
    # Filter words that exist in vocabulary
    valid_words = [word for word in words if word in model.wv]
    
    if len(valid_words) < 2:
        print("Not enough valid words for heatmap")
        return
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(valid_words), len(valid_words)))
    
    for i, word1 in enumerate(valid_words):
        for j, word2 in enumerate(valid_words):
            similarity_matrix[i, j] = model.wv.similarity(word1, word2)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=valid_words, 
                yticklabels=valid_words,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0)
    plt.title('Word Similarity Heatmap')
    plt.tight_layout()
    plt.show()

# Create heatmap for a subset of words
sample_words = ['intelligence', 'programming', 'business', 'science', 'learning', 
                'algorithm', 'marketing', 'research', 'development', 'analysis']

print("\nCreating similarity heatmap...")
create_similarity_heatmap(sg_model, sample_words)
```

## 💡 Key Insights About Word2Vec

### 1. **Skip-gram vs CBOW Trade-offs**

**Skip-gram:**
- ✅ Better for rare words
- ✅ Better semantic representations
- ❌ Slower training
- ❌ More memory intensive

**CBOW:**
- ✅ Faster training
- ✅ Better for frequent words
- ❌ Worse with rare words
- ❌ Less detailed semantics

### 2. **Hyperparameter Impact**

```python
def compare_hyperparameters():
    """Compare different hyperparameter settings"""
    
    # Small training corpus for quick comparison
    small_corpus = create_larger_corpus()[:100]
    
    configs = [
        {'vector_size': 50, 'window': 3, 'name': 'Small vectors, small window'},
        {'vector_size': 100, 'window': 5, 'name': 'Medium vectors, medium window'},
        {'vector_size': 200, 'window': 10, 'name': 'Large vectors, large window'}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        trainer = ProductionWord2Vec(
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=1,
            epochs=5
        )
        
        model = trainer.train_model(small_corpus)
        accuracy = trainer.evaluate_model()
        
        results.append({
            'config': config['name'],
            'accuracy': accuracy,
            'vocab_size': len(model.wv.key_to_index)
        })
    
    print("\nHyperparameter Comparison Results:")
    print("-" * 50)
    for result in results:
        print(f"{result['config']}: "
              f"Accuracy={result['accuracy']:.3f}, "
              f"Vocab={result['vocab_size']}")

# Run hyperparameter comparison
compare_hyperparameters()
```

## 🎯 Best Practices for Word2Vec

### 1. **Data Preparation**
- **Clean your text** but preserve important structure
- **Use consistent tokenization** across training and inference
- **Handle rare words** appropriately (min_count parameter)
- **Detect phrases** for multi-word expressions

### 2. **Model Selection**
- **Use Skip-gram** for small datasets or when you care about rare words
- **Use CBOW** for large datasets when training speed matters
- **Start with proven hyperparameters** (vector_size=300, window=5)

### 3. **Evaluation**
- **Test on analogies** to verify semantic relationships
- **Check clustering** of semantically related words
- **Evaluate on downstream tasks** (classification, similarity)
- **Visualize embeddings** to spot issues

### 4. **Production Considerations**
- **Save models properly** with Gensim's save/load methods
- **Version your models** as you retrain on new data
- **Monitor for vocabulary drift** in new data
- **Consider incremental training** for evolving datasets

## 🏋️ Practice Exercise

**Build a Domain-Specific Word2Vec Model**

Your task: Create a Word2Vec model for a specific domain (e.g., medical, legal, financial) and evaluate its quality.

```python
def build_domain_word2vec():
    """
    Build and evaluate a domain-specific Word2Vec model
    
    Requirements:
    1. Collect or simulate domain-specific text data
    2. Train both Skip-gram and CBOW models
    3. Evaluate using domain-specific analogies
    4. Compare with general-purpose embeddings
    5. Visualize domain-specific clusters
    
    Bonus:
    - Handle domain-specific preprocessing
    - Create custom evaluation metrics
    - Build a similarity search system
    - Deploy for real-time inference
    """
    
    # Your implementation here
    pass

# Example domain data
medical_sentences = [
    ["patient", "diagnosis", "treatment", "symptoms", "medical", "history"],
    ["doctor", "physician", "nurse", "healthcare", "professional", "clinic"],
    ["medication", "prescription", "dosage", "side", "effects", "pharmacy"],
    ["surgery", "operation", "anesthesia", "recovery", "hospital", "care"]
]

# Your model should excel at medical analogies like:
# doctor : patient :: teacher : student
# symptom : diagnosis :: clue : conclusion
```

## 💡 Key Takeaways

1. **Word2Vec learns meaning from context** - Words that appear together have similar meanings
2. **Skip-gram predicts context from word** - Better for rare words and semantic details  
3. **CBOW predicts word from context** - Faster training, good for frequent words
4. **Vector arithmetic captures relationships** - king - man + woman ≈ queen
5. **Hyperparameters matter significantly** - Experiment to find the best settings
6. **Evaluation is crucial** - Use multiple methods to assess quality

## 🚀 What's Next?

You've mastered Word2Vec fundamentals! Next, explore [GloVe and FastText](./03_glove_fasttext.md) to learn alternative embedding approaches that address Word2Vec's limitations.

**Coming up:**
- GloVe: Global statistics for better representations
- FastText: Handling unknown words with subword information
- Comparing embedding approaches
- When to use which method

Ready to explore more embedding techniques? Let's continue!
