# GloVe and FastText: Advanced Embedding Techniques

## 🎯 What You'll Learn

While Word2Vec was revolutionary, it has limitations. GloVe and FastText address these issues with global statistics and subword information. You'll master these advanced techniques and learn when to use each approach.

## 🧠 Understanding the Limitations of Word2Vec

Word2Vec learns from local context windows, but it misses some important patterns:

- **Global statistics**: It doesn't use overall word co-occurrence patterns
- **Unknown words**: Can't handle words not seen during training
- **Morphology**: Ignores word structure (prefixes, suffixes, roots)
- **Rare words**: Struggles with words that appear infrequently

## 🌐 GloVe: Global Vectors for Word Representation

**The key insight:** Combine the benefits of global matrix factorization (like LSA) with local context methods (like Word2Vec).

**Think of it as:** Instead of just looking at nearby words, GloVe looks at the entire document collection to understand how often words appear together globally.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time

class GloVeModel:
    """Implementation of GloVe (Global Vectors) algorithm"""
    
    def __init__(self, vector_size=100, window_size=5, min_count=5, 
                 learning_rate=0.05, epochs=100, x_max=100, alpha=0.75):
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.x_max = x_max  # Cutoff for weighting function
        self.alpha = alpha  # Exponent for weighting function
        
        # Model parameters
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
        # Embedding matrices
        self.W = None  # Main word vectors
        self.W_tilde = None  # Context word vectors
        self.b = None  # Main word biases
        self.b_tilde = None  # Context word biases
        
        # Co-occurrence matrix
        self.cooccurrence_matrix = None
        
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        
        print("Building vocabulary...")
        word_counts = Counter()
        
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
        
        # Filter by minimum count
        vocab_words = [word for word, count in word_counts.items() 
                      if count >= self.min_count]
        
        # Create mappings
        self.word_to_id = {word: idx for idx, word in enumerate(vocab_words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(vocab_words)
        
        print(f"Vocabulary size: {self.vocab_size}")
        return self.word_to_id
    
    def build_cooccurrence_matrix(self, sentences):
        """Build co-occurrence matrix from sentences"""
        
        print("Building co-occurrence matrix...")
        cooccur_dict = defaultdict(float)
        
        # Count co-occurrences
        for sentence in sentences:
            # Convert to IDs
            sentence_ids = [self.word_to_id[word] for word in sentence 
                          if word in self.word_to_id]
            
            # Count co-occurrences within window
            for i, center_id in enumerate(sentence_ids):
                start = max(0, i - self.window_size)
                end = min(len(sentence_ids), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_id = sentence_ids[j]
                        distance = abs(i - j)
                        # Weight by inverse distance
                        weight = 1.0 / distance
                        cooccur_dict[(center_id, context_id)] += weight
        
        # Convert to sparse matrix format
        rows, cols, data = [], [], []
        for (word_i, word_j), count in cooccur_dict.items():
            rows.append(word_i)
            cols.append(word_j)
            data.append(count)
        
        self.cooccurrence_matrix = coo_matrix(
            (data, (rows, cols)), 
            shape=(self.vocab_size, self.vocab_size)
        )
        
        print(f"Co-occurrence matrix shape: {self.cooccurrence_matrix.shape}")
        print(f"Non-zero entries: {self.cooccurrence_matrix.nnz}")
        
        return self.cooccurrence_matrix
    
    def weighting_function(self, x):
        """GloVe weighting function"""
        return np.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0)
    
    def initialize_parameters(self):
        """Initialize model parameters randomly"""
        
        # Initialize word vectors
        self.W = np.random.normal(0, 0.1, (self.vocab_size, self.vector_size))
        self.W_tilde = np.random.normal(0, 0.1, (self.vocab_size, self.vector_size))
        
        # Initialize biases
        self.b = np.random.normal(0, 0.1, self.vocab_size)
        self.b_tilde = np.random.normal(0, 0.1, self.vocab_size)
        
        print("Initialized parameters")
    
    def train(self, sentences):
        """Train GloVe model"""
        
        # Build vocabulary and co-occurrence matrix
        self.build_vocabulary(sentences)
        self.build_cooccurrence_matrix(sentences)
        self.initialize_parameters()
        
        # Convert sparse matrix to dense for easier computation
        # Note: In practice, you'd want to keep it sparse for large vocabularies
        cooccur_dense = self.cooccurrence_matrix.toarray()
        
        print(f"Starting GloVe training for {self.epochs} epochs...")
        
        # Get non-zero entries for training
        nonzero_indices = np.nonzero(cooccur_dense)
        num_pairs = len(nonzero_indices[0])
        
        loss_history = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            # Shuffle training pairs
            indices = np.random.permutation(num_pairs)
            
            for idx in indices:
                i = nonzero_indices[0][idx]
                j = nonzero_indices[1][idx]
                x_ij = cooccur_dense[i, j]
                
                if x_ij > 0:  # Skip zero entries
                    # Forward pass
                    diff = (np.dot(self.W[i], self.W_tilde[j]) + 
                           self.b[i] + self.b_tilde[j] - np.log(x_ij))
                    
                    # Apply weighting function
                    weight = self.weighting_function(x_ij)
                    
                    # Calculate loss
                    loss = weight * (diff ** 2)
                    epoch_loss += loss
                    
                    # Gradients
                    grad_factor = weight * diff
                    
                    # Update main word vector
                    grad_W_i = grad_factor * self.W_tilde[j]
                    self.W[i] -= self.learning_rate * grad_W_i
                    
                    # Update context word vector
                    grad_W_tilde_j = grad_factor * self.W[i]
                    self.W_tilde[j] -= self.learning_rate * grad_W_tilde_j
                    
                    # Update biases
                    self.b[i] -= self.learning_rate * grad_factor
                    self.b_tilde[j] -= self.learning_rate * grad_factor
            
            avg_loss = epoch_loss / num_pairs
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("GloVe training completed!")
        return loss_history
    
    def get_word_vector(self, word):
        """Get final word vector (sum of main and context vectors)"""
        if word not in self.word_to_id:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        word_id = self.word_to_id[word]
        # GloVe final vector is sum of main and context vectors
        return self.W[word_id] + self.W_tilde[word_id]
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        try:
            vec1 = self.get_word_vector(word1)
            vec2 = self.get_word_vector(word2)
            
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            return dot_product / norms if norms > 0 else 0
        except KeyError:
            return 0
    
    def most_similar(self, word, top_n=5):
        """Find most similar words"""
        if word not in self.word_to_id:
            return []
        
        target_vector = self.get_word_vector(word)
        similarities = []
        
        for other_word in self.word_to_id:
            if other_word != word:
                sim = self.similarity(word, other_word)
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

# Create sample corpus for GloVe training
def create_sample_corpus():
    """Create a sample corpus for training"""
    
    sentences = []
    
    # Add various sentences
    base_sentences = [
        "the cat sat on the mat",
        "dogs like to run and play",
        "cats and dogs are pets",
        "the park has many trees", 
        "trees provide shade and oxygen",
        "animals need food and water",
        "pets require love and care",
        "the sun shines bright today",
        "rain helps plants grow well",
        "flowers bloom in spring season"
    ]
    
    # Repeat and add variations
    for _ in range(100):  # Repeat for more training data
        for sentence in base_sentences:
            sentences.append(sentence.split())
    
    return sentences

# Train GloVe model
print("=" * 60)
print("TRAINING GLOVE MODEL")
print("=" * 60)

# Create training data
training_sentences = create_sample_corpus()

# Train GloVe
glove_model = GloVeModel(vector_size=50, window_size=3, epochs=50, learning_rate=0.1)
glove_loss = glove_model.train(training_sentences)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(glove_loss)
plt.title('GloVe Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()

# Test GloVe similarities
print("\nGloVe Word Similarities:")
test_words = ["cat", "dog", "tree", "sun"]

for word in test_words:
    if word in glove_model.word_to_id:
        similar = glove_model.most_similar(word, top_n=3)
        print(f"\nWords similar to '{word}':")
        for sim_word, similarity in similar:
            print(f"  {sim_word}: {similarity:.3f}")
```

## 🚀 FastText: Handling Unknown Words with Subword Information

**The key insight:** Words are made up of smaller parts (subwords) that carry meaning. By learning representations for these parts, we can handle unknown words.

**Think of it as:** Instead of treating "running" as one unit, FastText sees "run" + "ning" and can understand new words like "jumping" by recognizing the "ing" pattern.

```python
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
import re

class CustomFastText:
    """Custom FastText implementation with subword analysis"""
    
    def __init__(self, vector_size=100, window=5, min_count=1, 
                 min_n=3, max_n=6, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.min_n = min_n  # Minimum subword length
        self.max_n = max_n  # Maximum subword length
        self.epochs = epochs
        self.model = None
        
    def generate_subwords(self, word):
        """Generate character n-grams for a word"""
        
        # Add boundary markers
        padded_word = f"<{word}>"
        subwords = set()
        
        # Generate n-grams
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded_word) - n + 1):
                subword = padded_word[i:i+n]
                subwords.add(subword)
        
        return list(subwords)
    
    def train_model(self, sentences):
        """Train FastText model using Gensim"""
        
        print("Training FastText model...")
        
        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            min_n=self.min_n,
            max_n=self.max_n,
            epochs=self.epochs,
            sg=1,  # Use skip-gram
            workers=4
        )
        
        print(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
        print("Training completed!")
        
        return self.model
    
    def handle_unknown_word(self, word):
        """Demonstrate how FastText handles unknown words"""
        
        if word in self.model.wv:
            print(f"'{word}' is in vocabulary")
            return self.model.wv[word]
        else:
            print(f"'{word}' is NOT in vocabulary, but FastText can still handle it!")
            
            # FastText can still provide a vector by using subword information
            try:
                vector = self.model.wv[word]  # This works even for unknown words!
                return vector
            except KeyError:
                print(f"Even FastText couldn't handle '{word}'")
                return None
    
    def analyze_subwords(self, word):
        """Analyze the subwords that FastText uses for a word"""
        
        print(f"\nSubword analysis for '{word}':")
        
        # Generate subwords
        subwords = self.generate_subwords(word)
        print(f"Generated {len(subwords)} subwords:")
        
        # Show subwords in chunks
        for i in range(0, len(subwords), 10):
            chunk = subwords[i:i+10]
            print(f"  {chunk}")
        
        # Try to get vector (works even if word wasn't in training)
        try:
            vector = self.model.wv[word]
            print(f"✓ FastText can provide a vector for '{word}'")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector norm: {np.linalg.norm(vector):.3f}")
        except:
            print(f"✗ Could not get vector for '{word}'")
    
    def morphology_test(self):
        """Test FastText's understanding of morphology"""
        
        print("\n" + "="*50)
        print("MORPHOLOGY TEST")
        print("="*50)
        
        # Test words with common prefixes/suffixes
        test_groups = {
            'ing_words': ['running', 'jumping', 'swimming', 'walking'],
            'ed_words': ['walked', 'jumped', 'played', 'worked'],
            'un_words': ['unhappy', 'unknown', 'unable', 'unfair'],
            'tion_words': ['action', 'creation', 'education', 'nation']
        }
        
        for group_name, words in test_groups.items():
            print(f"\n{group_name.upper()}:")
            
            # Calculate similarities within group
            similarities = []
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i < j:
                        try:
                            sim = self.model.wv.similarity(word1, word2)
                            similarities.append(sim)
                            print(f"  {word1} ↔ {word2}: {sim:.3f}")
                        except:
                            print(f"  {word1} ↔ {word2}: Could not calculate")
            
            if similarities:
                avg_sim = np.mean(similarities)
                print(f"  Average within-group similarity: {avg_sim:.3f}")
    
    def out_of_vocabulary_test(self):
        """Test FastText's ability to handle unseen words"""
        
        print("\n" + "="*50)
        print("OUT-OF-VOCABULARY TEST")
        print("="*50)
        
        # Create words that likely weren't in training
        oov_words = [
            'superfantastic',  # Made-up word with recognizable parts
            'unbelievable',    # Real word that might not be in small corpus
            'preprocessing',   # Technical term
            'antidisestablishmentarianism',  # Very long word
            'running123',      # Word with numbers
            'covid19'          # Modern term
        ]
        
        for word in oov_words:
            print(f"\nTesting: '{word}'")
            
            # Check if it's in vocabulary
            in_vocab = word in self.model.wv
            print(f"  In vocabulary: {in_vocab}")
            
            # Try to get vector
            try:
                vector = self.model.wv[word]
                print(f"  ✓ Got vector of shape {vector.shape}")
                
                # Find similar words
                try:
                    similar = self.model.wv.most_similar(word, topn=3)
                    print(f"  Similar words:")
                    for sim_word, similarity in similar:
                        print(f"    {sim_word}: {similarity:.3f}")
                except:
                    print(f"  Could not find similar words")
                    
            except Exception as e:
                print(f"  ✗ Could not get vector: {e}")

# Train FastText model
print("\n" + "="*60)
print("TRAINING FASTTEXT MODEL")
print("="*60)

# Create a more diverse corpus for FastText
def create_diverse_corpus():
    """Create a corpus with various morphological forms"""
    
    sentences = []
    
    base_words = {
        'run': ['run', 'running', 'ran', 'runs', 'runner'],
        'jump': ['jump', 'jumping', 'jumped', 'jumps', 'jumper'],
        'walk': ['walk', 'walking', 'walked', 'walks', 'walker'],
        'play': ['play', 'playing', 'played', 'plays', 'player'],
        'work': ['work', 'working', 'worked', 'works', 'worker'],
        'teach': ['teach', 'teaching', 'taught', 'teaches', 'teacher'],
        'learn': ['learn', 'learning', 'learned', 'learns', 'learner']
    }
    
    # Create sentences with various forms
    for base, forms in base_words.items():
        for form in forms:
            # Create multiple sentences with this form
            sentence_templates = [
                f"I like to {form}",
                f"The {form} is good",
                f"We saw {form} yesterday",
                f"Everyone enjoys {form}",
                f"This {form} makes me happy"
            ]
            
            for template in sentence_templates:
                # Only use valid sentences (basic grammar check)
                if form.endswith('ing') and 'to ' in template:
                    continue  # Skip "to running" etc.
                sentences.append(template.split())
    
    # Add some additional sentences
    additional = [
        "the cat sat on the mat".split(),
        "dogs like to run in the park".split(),
        "children love playing games".split(),
        "teachers help students learn".split(),
        "workers build many buildings".split()
    ] * 20
    
    sentences.extend(additional)
    
    print(f"Created corpus with {len(sentences)} sentences")
    return sentences

# Create training corpus
fasttext_corpus = create_diverse_corpus()

# Train FastText model
fasttext_trainer = CustomFastText(vector_size=100, min_n=3, max_n=6, epochs=20)
fasttext_model = fasttext_trainer.train_model(fasttext_corpus)

# Test subword analysis
print("\nAnalyzing subwords:")
fasttext_trainer.analyze_subwords("running")
fasttext_trainer.analyze_subwords("fantastic")

# Test morphology understanding
fasttext_trainer.morphology_test()

# Test out-of-vocabulary handling
fasttext_trainer.out_of_vocabulary_test()
```

## 📊 Comparing Word2Vec, GloVe, and FastText

Let's systematically compare all three approaches:

```python
from gensim.models import Word2Vec
import pandas as pd

class EmbeddingComparison:
    """Compare different embedding approaches"""
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.models = {}
        self.results = {}
        
    def train_all_models(self):
        """Train Word2Vec, GloVe, and FastText models"""
        
        print("Training all embedding models...")
        
        # Train Word2Vec (Skip-gram)
        print("\n1. Training Word2Vec...")
        w2v_model = Word2Vec(
            sentences=self.corpus,
            vector_size=100,
            window=5,
            min_count=2,
            epochs=20,
            sg=1,  # Skip-gram
            workers=4
        )
        self.models['Word2Vec'] = w2v_model
        
        # Train FastText
        print("\n2. Training FastText...")
        ft_model = FastText(
            sentences=self.corpus,
            vector_size=100,
            window=5,
            min_count=2,
            epochs=20,
            sg=1,
            workers=4
        )
        self.models['FastText'] = ft_model
        
        # Train GloVe (using our custom implementation)
        print("\n3. Training GloVe...")
        glove_model = GloVeModel(vector_size=100, window_size=5, epochs=30)
        glove_model.train(self.corpus)
        self.models['GloVe'] = glove_model
        
        print("All models trained!")
    
    def compare_similarities(self, test_words):
        """Compare word similarities across models"""
        
        print("\n" + "="*60)
        print("SIMILARITY COMPARISON")
        print("="*60)
        
        for word in test_words:
            print(f"\nWords similar to '{word}':")
            
            for model_name, model in self.models.items():
                print(f"\n{model_name}:")
                
                try:
                    if model_name == 'GloVe':
                        similar = model.most_similar(word, top_n=3)
                    else:
                        similar = model.wv.most_similar(word, topn=3)
                    
                    for sim_word, similarity in similar:
                        print(f"  {sim_word}: {similarity:.3f}")
                        
                except Exception as e:
                    print(f"  Error: {e}")
    
    def compare_analogies(self, analogies):
        """Compare analogy performance across models"""
        
        print("\n" + "="*60)
        print("ANALOGY COMPARISON")
        print("="*60)
        
        results = {model_name: [] for model_name in self.models.keys()}
        
        for positive, negative, expected in analogies:
            print(f"\nAnalogy: {' + '.join(positive)} - {' + '.join(negative)} = ?")
            print(f"Expected: {expected}")
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'GloVe':
                        # Implement analogy for GloVe
                        if all(word in model.word_to_id for word in positive + negative):
                            # Simple vector arithmetic for GloVe
                            result_vec = sum(model.get_word_vector(word) for word in positive)
                            result_vec -= sum(model.get_word_vector(word) for word in negative)
                            
                            # Find closest word
                            best_word = None
                            best_sim = -1
                            
                            for word in model.word_to_id:
                                if word not in positive + negative:
                                    word_vec = model.get_word_vector(word)
                                    sim = np.dot(result_vec, word_vec) / (np.linalg.norm(result_vec) * np.linalg.norm(word_vec))
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_word = word
                            
                            result = [(best_word, best_sim)] if best_word else []
                        else:
                            result = []
                    else:
                        result = model.wv.most_similar(
                            positive=positive,
                            negative=negative,
                            topn=1
                        )
                    
                    if result:
                        predicted = result[0][0]
                        confidence = result[0][1]
                        is_correct = predicted.lower() == expected.lower()
                        
                        status = "✓" if is_correct else "✗"
                        print(f"  {model_name}: {status} {predicted} ({confidence:.3f})")
                        
                        results[model_name].append(is_correct)
                    else:
                        print(f"  {model_name}: ✗ No result")
                        results[model_name].append(False)
                        
                except Exception as e:
                    print(f"  {model_name}: ✗ Error: {e}")
                    results[model_name].append(False)
        
        # Calculate accuracies
        print(f"\nAnalogy Accuracies:")
        for model_name, correct_list in results.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                print(f"  {model_name}: {accuracy:.3f} ({sum(correct_list)}/{len(correct_list)})")
    
    def compare_oov_handling(self, oov_words):
        """Compare out-of-vocabulary word handling"""
        
        print("\n" + "="*60)
        print("OUT-OF-VOCABULARY COMPARISON")
        print("="*60)
        
        for word in oov_words:
            print(f"\nTesting: '{word}'")
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'GloVe':
                        if word in model.word_to_id:
                            vector = model.get_word_vector(word)
                            print(f"  {model_name}: ✓ In vocabulary")
                        else:
                            print(f"  {model_name}: ✗ Not in vocabulary, cannot handle")
                    else:
                        # Word2Vec and FastText
                        if word in model.wv:
                            print(f"  {model_name}: ✓ In vocabulary")
                        else:
                            if model_name == 'FastText':
                                # FastText can handle OOV words
                                try:
                                    vector = model.wv[word]
                                    print(f"  {model_name}: ✓ Not in vocab, but can handle via subwords")
                                except:
                                    print(f"  {model_name}: ✗ Cannot handle")
                            else:
                                print(f"  {model_name}: ✗ Not in vocabulary, cannot handle")
                                
                except Exception as e:
                    print(f"  {model_name}: ✗ Error: {e}")
    
    def performance_analysis(self):
        """Analyze computational performance"""
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        test_word = "running"
        num_iterations = 1000
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            try:
                for _ in range(num_iterations):
                    if model_name == 'GloVe':
                        if test_word in model.word_to_id:
                            vector = model.get_word_vector(test_word)
                    else:
                        if test_word in model.wv:
                            vector = model.wv[test_word]
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
                
                print(f"{model_name}: {avg_time:.3f} ms per lookup")
                
            except Exception as e:
                print(f"{model_name}: Error during performance test: {e}")

# Run comprehensive comparison
print("\n" + "="*80)
print("COMPREHENSIVE EMBEDDING COMPARISON")
print("="*80)

# Create comparison corpus
comparison_corpus = create_diverse_corpus()

# Initialize comparison
comparison = EmbeddingComparison(comparison_corpus)

# Train all models
comparison.train_all_models()

# Compare similarities
test_words = ["running", "teacher", "player"]
comparison.compare_similarities(test_words)

# Compare analogies
test_analogies = [
    (['running', 'runner'], ['teaching'], 'teacher'),
    (['walk', 'walking'], ['run'], 'running'),
    (['play', 'player'], ['work'], 'worker')
]
comparison.compare_analogies(test_analogies)

# Test OOV handling
oov_test_words = ['preprocessing', 'unbelievable', 'fantastic']
comparison.compare_oov_handling(oov_test_words)

# Performance analysis
comparison.performance_analysis()
```

## 🎯 When to Use Which Embedding Method

### Decision Matrix

```python
def choose_embedding_method(dataset_size, vocabulary_size, oov_frequency, 
                          computation_budget, morphology_importance):
    """Help choose the best embedding method based on requirements"""
    
    scores = {
        'Word2Vec': 0,
        'GloVe': 0,
        'FastText': 0
    }
    
    # Dataset size consideration
    if dataset_size == 'small':
        scores['Word2Vec'] += 2
        scores['FastText'] += 3  # Better for small datasets
    elif dataset_size == 'medium':
        scores['Word2Vec'] += 3
        scores['GloVe'] += 2
        scores['FastText'] += 2
    else:  # large
        scores['GloVe'] += 3  # Better global statistics
        scores['Word2Vec'] += 2
        scores['FastText'] += 1
    
    # Vocabulary size
    if vocabulary_size == 'small':
        scores['FastText'] += 3  # Handles rare words better
        scores['Word2Vec'] += 1
    elif vocabulary_size == 'medium':
        scores['Word2Vec'] += 2
        scores['GloVe'] += 2
        scores['FastText'] += 2
    else:  # large
        scores['GloVe'] += 3
        scores['Word2Vec'] += 2
        scores['FastText'] += 1  # Memory intensive
    
    # Out-of-vocabulary frequency
    if oov_frequency == 'high':
        scores['FastText'] += 5  # Only FastText handles OOV well
    elif oov_frequency == 'medium':
        scores['FastText'] += 3
        scores['Word2Vec'] += 1
        scores['GloVe'] += 1
    else:  # low
        scores['Word2Vec'] += 2
        scores['GloVe'] += 2
        scores['FastText'] += 1
    
    # Computation budget
    if computation_budget == 'low':
        scores['Word2Vec'] += 3
        scores['GloVe'] += 2
        scores['FastText'] += 1  # Most expensive
    elif computation_budget == 'medium':
        scores['Word2Vec'] += 2
        scores['GloVe'] += 2
        scores['FastText'] += 2
    else:  # high
        scores['FastText'] += 3
        scores['GloVe'] += 2
        scores['Word2Vec'] += 1
    
    # Morphology importance
    if morphology_importance == 'high':
        scores['FastText'] += 4  # Only FastText captures morphology
    elif morphology_importance == 'medium':
        scores['FastText'] += 2
        scores['Word2Vec'] += 1
        scores['GloVe'] += 1
    else:  # low
        scores['Word2Vec'] += 1
        scores['GloVe'] += 1
    
    # Find best method
    best_method = max(scores.keys(), key=lambda x: scores[x])
    
    print("Embedding Method Recommendation:")
    print(f"Best choice: {best_method}")
    print("\nScores:")
    for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {score}")
    
    return best_method, scores

# Example usage
print("\n" + "="*50)
print("EMBEDDING METHOD SELECTION")
print("="*50)

# Scenario 1: Small dataset, many unknown words
print("\nScenario 1: Small dataset with many technical terms")
choose_embedding_method(
    dataset_size='small',
    vocabulary_size='small', 
    oov_frequency='high',
    computation_budget='medium',
    morphology_importance='high'
)

# Scenario 2: Large dataset, stable vocabulary
print("\nScenario 2: Large news corpus with stable vocabulary")
choose_embedding_method(
    dataset_size='large',
    vocabulary_size='large',
    oov_frequency='low', 
    computation_budget='high',
    morphology_importance='low'
)
```

## 💡 Production Best Practices

### 1. **Model Selection Guidelines**

```python
class EmbeddingProductionGuide:
    """Best practices for production embedding systems"""
    
    @staticmethod
    def data_preprocessing_tips():
        """Data preprocessing recommendations"""
        
        tips = {
            'Word2Vec': [
                "Clean but preserve important punctuation",
                "Handle phrase detection (New_York, machine_learning)",
                "Consistent tokenization across train/test",
                "Consider subsampling frequent words"
            ],
            'GloVe': [
                "More aggressive cleaning acceptable",
                "Focus on global co-occurrence patterns",
                "Consider lemmatization for better statistics",
                "Handle rare words carefully"
            ],
            'FastText': [
                "Minimal cleaning to preserve morphology",
                "Keep original word forms",
                "Handle multiple languages in same corpus",
                "Consider character-level noise"
            ]
        }
        
        for method, method_tips in tips.items():
            print(f"\n{method} preprocessing tips:")
            for tip in method_tips:
                print(f"  • {tip}")
    
    @staticmethod
    def hyperparameter_recommendations():
        """Recommended hyperparameters for different scenarios"""
        
        scenarios = {
            'General Purpose': {
                'Word2Vec': {'vector_size': 300, 'window': 5, 'min_count': 5},
                'GloVe': {'vector_size': 300, 'window_size': 5, 'min_count': 5},
                'FastText': {'vector_size': 300, 'window': 5, 'min_n': 3, 'max_n': 6}
            },
            'Small Dataset': {
                'Word2Vec': {'vector_size': 100, 'window': 3, 'min_count': 1},
                'GloVe': {'vector_size': 100, 'window_size': 3, 'min_count': 1},
                'FastText': {'vector_size': 100, 'window': 3, 'min_n': 3, 'max_n': 6}
            },
            'Large Dataset': {
                'Word2Vec': {'vector_size': 500, 'window': 10, 'min_count': 10},
                'GloVe': {'vector_size': 500, 'window_size': 10, 'min_count': 10},
                'FastText': {'vector_size': 500, 'window': 10, 'min_n': 3, 'max_n': 6}
            }
        }
        
        for scenario, methods in scenarios.items():
            print(f"\n{scenario}:")
            for method, params in methods.items():
                print(f"  {method}: {params}")
    
    @staticmethod
    def evaluation_checklist():
        """Evaluation checklist for embedding quality"""
        
        checklist = [
            "✓ Test on word similarity benchmarks (SimLex-999, WordSim-353)",
            "✓ Evaluate on analogy tasks (Google analogies, BATS)",
            "✓ Check clustering of semantic categories",
            "✓ Visualize embeddings with t-SNE/UMAP",
            "✓ Test on downstream tasks (classification, NER)",
            "✓ Analyze bias in embeddings (gender, racial, cultural)",
            "✓ Check coverage of domain-specific vocabulary",
            "✓ Measure inference speed and memory usage",
            "✓ Test robustness to typos and variations",
            "✓ Validate cross-lingual capabilities (if applicable)"
        ]
        
        print("Embedding Evaluation Checklist:")
        for item in checklist:
            print(f"  {item}")

# Display production guidance
print("\n" + "="*60)
print("PRODUCTION BEST PRACTICES")
print("="*60)

guide = EmbeddingProductionGuide()

guide.data_preprocessing_tips()
print("\n" + "-"*50)
guide.hyperparameter_recommendations() 
print("\n" + "-"*50)
guide.evaluation_checklist()
```

## 🏋️ Practice Exercise

**Build a Multi-Method Embedding System**

Create a system that can use different embedding methods based on the input characteristics:

```python
def build_adaptive_embedding_system():
    """
    Build an embedding system that adapts to different scenarios
    
    Requirements:
    1. Implement all three methods (Word2Vec, GloVe, FastText)
    2. Create automatic method selection based on data characteristics
    3. Build evaluation suite with multiple metrics
    4. Handle both in-vocabulary and out-of-vocabulary words
    5. Create visualization tools for embedding analysis
    
    Bonus:
    - Ensemble methods combining multiple embeddings
    - Dynamic vocabulary expansion
    - Real-time similarity search
    - Multi-language support
    """
    
    # Your implementation here
    pass

# Test scenarios for your system
test_scenarios = [
    {
        'name': 'Medical Text',
        'corpus': ['patient diagnosis treatment medicine doctor hospital'],
        'expected_challenges': ['technical terms', 'abbreviations', 'new drugs']
    },
    {
        'name': 'Social Media',
        'corpus': ['lol amazing food selfie happy vacation'],
        'expected_challenges': ['informal language', 'typos', 'new slang']
    },
    {
        'name': 'Legal Documents', 
        'corpus': ['contract agreement party liability terms conditions'],
        'expected_challenges': ['formal language', 'long compounds', 'precision required']
    }
]
```

## 💡 Key Takeaways

1. **GloVe uses global statistics** - Better at capturing overall word relationships
2. **FastText handles subwords** - Only method that can handle unknown words
3. **Word2Vec is simple and effective** - Good baseline for most applications
4. **Choose based on your specific needs** - Dataset size, vocabulary, OOV frequency
5. **Preprocessing matters differently** - Each method has different requirements
6. **Evaluation is method-specific** - Different strengths require different tests

## 🚀 What's Next?

You've mastered the three fundamental embedding approaches! Next, explore [Document Embeddings and Sentence Representations](./04_document_embeddings.md) to learn how to represent entire documents and sentences.

**Coming up:**

- Doc2Vec: Extending embeddings to documents
- Sentence-BERT: Modern sentence representations
- Paragraph vectors and document similarity
- Applications in document retrieval and clustering

Ready to scale up from words to documents? Let's continue!
