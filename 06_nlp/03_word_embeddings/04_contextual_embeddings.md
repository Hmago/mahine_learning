# Contextual Embeddings and BERT

## 🎯 What You'll Learn

Traditional embeddings give each word a single vector, but words have different meanings in different contexts. You'll master contextual embeddings that adapt to context, with hands-on implementation of ELMo and BERT concepts.

## 🧠 The Context Problem

Traditional embeddings have a fundamental limitation: **one vector per word type**

**Example Problem:**
- "I'll **bank** the money" (financial institution)
- "I'll **bank** left at the river" (turn/lean)
- "The river **bank** is muddy" (riverbank)

All three uses of "bank" get the same vector in Word2Vec/GloVe, but they mean completely different things!

## 🔄 ELMo: Embeddings from Language Models

**The key insight:** Use a language model to create context-dependent embeddings. The same word gets different representations based on its surrounding context.

**Think of it as:** Instead of having one "bank" in your vocabulary, you have different "banks" depending on the sentence they appear in.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
from sklearn.manifold import TSNE
import seaborn as sns

class SimpleELMo:
    """Simplified ELMo implementation for educational purposes"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Vocabulary mappings
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size_actual = 0
        
        # Models
        self.forward_lstm = None
        self.backward_lstm = None
        self.embedding_layer = None
        
        # Trained embeddings cache
        self.contextual_embeddings = {}
        
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        
        print("Building vocabulary...")
        word_counts = defaultdict(int)
        
        for sentence in sentences:
            for word in sentence:
                word_counts[word.lower()] += 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[:self.vocab_size-2]]
        
        # Add special tokens
        vocab_words = ['<UNK>', '<PAD>'] + vocab_words
        
        # Create mappings
        self.word_to_id = {word: idx for idx, word in enumerate(vocab_words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size_actual = len(vocab_words)
        
        print(f"Vocabulary size: {self.vocab_size_actual}")
        return self.word_to_id
    
    def sentences_to_ids(self, sentences):
        """Convert sentences to token IDs"""
        
        id_sentences = []
        for sentence in sentences:
            ids = []
            for word in sentence:
                word_lower = word.lower()
                word_id = self.word_to_id.get(word_lower, 0)  # 0 is <UNK>
                ids.append(word_id)
            id_sentences.append(ids)
        
        return id_sentences

class BiLSTMLanguageModel(nn.Module):
    """Bidirectional LSTM Language Model (simplified ELMo)"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Forward LSTM
        self.forward_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Backward LSTM  
        self.backward_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        
        # Output projections
        self.forward_projection = nn.Linear(hidden_dim, vocab_size)
        self.backward_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids):
        """Forward pass through the model"""
        
        # Get embeddings
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embeddings = self.dropout(embeddings)
        
        # Forward LSTM
        forward_out, _ = self.forward_lstm(embeddings)
        forward_predictions = self.forward_projection(forward_out)
        
        # Backward LSTM (reverse the sequence)
        backward_embeddings = torch.flip(embeddings, dims=[1])
        backward_out, _ = self.backward_lstm(backward_embeddings)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back
        backward_predictions = self.backward_projection(backward_out)
        
        # Store outputs for contextual embeddings
        self.last_forward_states = forward_out
        self.last_backward_states = backward_out
        self.last_embeddings = embeddings
        
        return forward_predictions, backward_predictions
    
    def get_contextual_embedding(self, layer_weights=None):
        """Get contextual embeddings from different layers"""
        
        if layer_weights is None:
            # Default: equal weights for all representations
            layer_weights = [1.0, 1.0, 1.0]  # [embedding, forward, backward]
        
        # Normalize weights
        layer_weights = np.array(layer_weights)
        layer_weights = layer_weights / layer_weights.sum()
        
        # Combine representations
        contextual_emb = (
            layer_weights[0] * self.last_embeddings +
            layer_weights[1] * self.last_forward_states +
            layer_weights[2] * self.last_backward_states
        )
        
        return contextual_emb

class ELMoDataset(Dataset):
    """Dataset for training ELMo"""
    
    def __init__(self, id_sentences, max_length=50):
        self.sentences = id_sentences
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # Truncate or pad
        if len(sentence) > self.max_length:
            sentence = sentence[:self.max_length]
        else:
            sentence = sentence + [1] * (self.max_length - len(sentence))  # 1 is <PAD>
        
        input_ids = torch.tensor(sentence[:-1], dtype=torch.long)  # Input
        forward_targets = torch.tensor(sentence[1:], dtype=torch.long)  # Next word
        backward_targets = torch.tensor(sentence[:-1], dtype=torch.long)  # Previous word
        
        return input_ids, forward_targets, backward_targets

class ELMoTrainer:
    """Trainer for ELMo model"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore padding
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, forward_targets, backward_targets) in enumerate(dataloader):
            
            # Forward pass
            forward_pred, backward_pred = self.model(input_ids)
            
            # Calculate losses
            forward_loss = self.criterion(
                forward_pred.reshape(-1, forward_pred.size(-1)),
                forward_targets.reshape(-1)
            )
            backward_loss = self.criterion(
                backward_pred.reshape(-1, backward_pred.size(-1)),
                backward_targets.reshape(-1)
            )
            
            # Total loss
            loss = forward_loss + backward_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(self, dataloader, epochs=10):
        """Train the model"""
        
        print(f"Training ELMo for {epochs} epochs...")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")

# Create sample corpus for ELMo training
def create_elmo_corpus():
    """Create corpus with words having multiple meanings"""
    
    sentences = [
        # Bank (financial)
        "I need to go to the bank to deposit money".split(),
        "The bank offers great interest rates".split(),
        "She works at the central bank".split(),
        "The bank approved my loan application".split(),
        
        # Bank (river)
        "We sat on the river bank".split(),
        "The bank was covered with flowers".split(),
        "Fish swim near the bank".split(),
        "The muddy bank was slippery".split(),
        
        # Bank (turn/lean)
        "The airplane will bank left".split(),
        "Bank the fire with more wood".split(),
        "Don't bank on that happening".split(),
        
        # Other ambiguous words
        "The bat flew at night".split(),
        "He swung the baseball bat".split(),
        "The apple fell from the tree".split(),
        "She has an Apple computer".split(),
        "The key opens the door".split(),
        "This is the key point".split(),
        
        # Context sentences
        "The weather is nice today".split(),
        "I like to read books".split(),
        "Music makes me happy".split(),
        "Learning is important".split(),
    ] * 20  # Repeat for more training data
    
    return sentences

# Demonstrate ELMo training and contextual embeddings
print("=" * 60)
print("ELMO CONTEXTUAL EMBEDDINGS DEMONSTRATION")
print("=" * 60)

# Create corpus
elmo_corpus = create_elmo_corpus()

# Initialize ELMo
elmo = SimpleELMo(vocab_size=1000, embedding_dim=64, hidden_dim=128)
elmo.build_vocabulary(elmo_corpus)

# Convert to IDs
id_sentences = elmo.sentences_to_ids(elmo_corpus)

# Create model and dataset
model = BiLSTMLanguageModel(
    vocab_size=elmo.vocab_size_actual,
    embedding_dim=elmo.embedding_dim,
    hidden_dim=elmo.hidden_dim
)

dataset = ELMoDataset(id_sentences, max_length=20)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train model (simplified training)
trainer = ELMoTrainer(model, learning_rate=0.01)

# Train for fewer epochs for demonstration
print("\nTraining simplified ELMo model...")
trainer.train(dataloader, epochs=3)

# Test contextual embeddings
def test_contextual_embeddings():
    """Test how the same word gets different embeddings in different contexts"""
    
    print("\n" + "=" * 50)
    print("CONTEXTUAL EMBEDDING ANALYSIS")
    print("=" * 50)
    
    # Test sentences with the word "bank"
    test_sentences = [
        "I went to the bank".split(),
        "The river bank is muddy".split(),
        "The plane will bank left".split()
    ]
    
    model.eval()
    embeddings_by_context = {}
    
    for sentence in test_sentences:
        print(f"\nAnalyzing: '{' '.join(sentence)}'")
        
        # Convert to IDs
        ids = []
        for word in sentence:
            word_id = elmo.word_to_id.get(word.lower(), 0)
            ids.append(word_id)
        
        # Pad to fixed length
        max_len = 10
        if len(ids) < max_len:
            ids.extend([1] * (max_len - len(ids)))  # Pad with <PAD>
        else:
            ids = ids[:max_len]
        
        input_tensor = torch.tensor([ids], dtype=torch.long)
        
        # Get contextual embeddings
        with torch.no_grad():
            _ = model(input_tensor)  # Run forward pass
            contextual_emb = model.get_contextual_embedding()
        
        # Find position of "bank" if it exists
        bank_id = elmo.word_to_id.get('bank', -1)
        if bank_id != -1:
            try:
                bank_position = ids.index(bank_id)
                bank_embedding = contextual_emb[0, bank_position].numpy()
                
                context_key = ' '.join(sentence)
                embeddings_by_context[context_key] = bank_embedding
                
                print(f"  Found 'bank' at position {bank_position}")
                print(f"  Embedding norm: {np.linalg.norm(bank_embedding):.3f}")
                
            except ValueError:
                print("  'bank' not found in this sentence")
    
    # Compare embeddings across contexts
    if len(embeddings_by_context) > 1:
        print(f"\nComparing 'bank' embeddings across contexts:")
        
        contexts = list(embeddings_by_context.keys())
        embeddings = list(embeddings_by_context.values())
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  '{contexts[i]}' vs '{contexts[j]}': {similarity:.3f}")

# Run contextual embedding test
test_contextual_embeddings()
```

## 🤖 BERT: Bidirectional Encoder Representations from Transformers

BERT revolutionized NLP by using the transformer architecture to create deep, bidirectional contextual embeddings:

```python
class SimpleBERT:
    """Simplified BERT implementation for educational purposes"""
    
    def __init__(self, vocab_size=10000, hidden_size=256, num_attention_heads=8, 
                 num_layers=6, max_position_embeddings=512):
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1, 
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        # Vocabulary
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Model components (simplified)
        self.embeddings = {}
        self.attention_weights = {}
        
    def build_vocabulary(self, sentences):
        """Build BERT vocabulary with special tokens"""
        
        print("Building BERT vocabulary...")
        
        # Start with special tokens
        self.word_to_id = self.special_tokens.copy()
        
        # Count words
        word_counts = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                word_counts[word.lower()] += 1
        
        # Add most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        current_id = len(self.special_tokens)
        for word, count in sorted_words:
            if current_id >= self.vocab_size:
                break
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                current_id += 1
        
        # Create reverse mapping
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        print(f"BERT vocabulary size: {len(self.word_to_id)}")
        
    def tokenize_for_bert(self, sentence):
        """Tokenize sentence for BERT input"""
        
        tokens = ['[CLS]']  # Start with CLS token
        
        for word in sentence:
            word_lower = word.lower()
            if word_lower in self.word_to_id:
                tokens.append(word_lower)
            else:
                tokens.append('[UNK]')
        
        tokens.append('[SEP]')  # End with SEP token
        
        return tokens
    
    def create_masked_tokens(self, tokens, mask_prob=0.15):
        """Create masked tokens for BERT training"""
        
        masked_tokens = tokens.copy()
        labels = [-1] * len(tokens)  # -1 means ignore
        
        for i in range(1, len(tokens) - 1):  # Skip [CLS] and [SEP]
            if np.random.random() < mask_prob:
                labels[i] = self.word_to_id[tokens[i]]  # Store original token
                
                # 80% of the time, replace with [MASK]
                if np.random.random() < 0.8:
                    masked_tokens[i] = '[MASK]'
                # 10% of the time, replace with random token
                elif np.random.random() < 0.5:
                    random_token = np.random.choice(list(self.word_to_id.keys()))
                    masked_tokens[i] = random_token
                # 10% of the time, keep original (but still predict)
        
        return masked_tokens, labels

class MultiHeadAttention:
    """Simplified Multi-Head Attention mechanism"""
    
    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Initialize weight matrices (simplified)
        self.W_q = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.W_k = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.W_v = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        self.W_o = np.random.normal(0, 0.02, (hidden_size, hidden_size))
        
    def attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        
        # Calculate attention scores
        scores = np.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.head_size)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x, axis=-1):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass through multi-head attention"""
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear transformations
        Q = np.matmul(hidden_states, self.W_q)
        K = np.matmul(hidden_states, self.W_k)
        V = np.matmul(hidden_states, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output, attention_weights = self.attention(Q, K, V, attention_mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size
        )
        
        # Final linear transformation
        output = np.matmul(attention_output, self.W_o)
        
        return output, attention_weights

class BERTEmbeddings:
    """BERT embedding layer (token + position + segment embeddings)"""
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Initialize embedding matrices
        self.token_embeddings = np.random.normal(0, 0.02, (vocab_size, hidden_size))
        self.position_embeddings = np.random.normal(0, 0.02, (max_position_embeddings, hidden_size))
        self.segment_embeddings = np.random.normal(0, 0.02, (2, hidden_size))  # 2 segments
        
    def forward(self, token_ids, segment_ids=None):
        """Create BERT embeddings"""
        
        seq_length = len(token_ids)
        
        # Token embeddings
        token_emb = self.token_embeddings[token_ids]
        
        # Position embeddings
        position_ids = np.arange(seq_length)
        position_emb = self.position_embeddings[position_ids]
        
        # Segment embeddings (default to 0 if not provided)
        if segment_ids is None:
            segment_ids = np.zeros(seq_length, dtype=int)
        segment_emb = self.segment_embeddings[segment_ids]
        
        # Sum all embeddings
        embeddings = token_emb + position_emb + segment_emb
        
        return embeddings

class BERTDemo:
    """Demonstrate BERT concepts"""
    
    def __init__(self):
        self.bert = SimpleBERT(vocab_size=5000, hidden_size=128)
        self.attention = MultiHeadAttention(hidden_size=128, num_heads=8)
        self.embeddings = None
        
    def train_bert_concepts(self, sentences):
        """Demonstrate BERT training concepts"""
        
        print("=" * 60)
        print("BERT CONCEPTS DEMONSTRATION")
        print("=" * 60)
        
        # Build vocabulary
        self.bert.build_vocabulary(sentences)
        
        # Initialize embeddings
        self.embeddings = BERTEmbeddings(
            vocab_size=len(self.bert.word_to_id),
            hidden_size=self.bert.hidden_size,
            max_position_embeddings=512
        )
        
        print("\n1. TOKENIZATION")
        print("-" * 30)
        
        # Demonstrate tokenization
        example_sentence = "The bank approved my loan".split()
        tokens = self.bert.tokenize_for_bert(example_sentence)
        print(f"Original: {example_sentence}")
        print(f"BERT tokens: {tokens}")
        
        # Convert to IDs
        token_ids = [self.bert.word_to_id[token] for token in tokens]
        print(f"Token IDs: {token_ids}")
        
        print("\n2. MASKED LANGUAGE MODELING")
        print("-" * 30)
        
        # Demonstrate masking
        masked_tokens, labels = self.bert.create_masked_tokens(tokens)
        print(f"Original tokens: {tokens}")
        print(f"Masked tokens: {masked_tokens}")
        print(f"Labels (original tokens for masked positions): {labels}")
        
        print("\n3. EMBEDDINGS")
        print("-" * 30)
        
        # Demonstrate embeddings
        embeddings = self.embeddings.forward(token_ids)
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Each token gets a {self.bert.hidden_size}-dimensional vector")
        
        print("\n4. ATTENTION VISUALIZATION")
        print("-" * 30)
        
        # Demonstrate attention
        embeddings_batch = embeddings.reshape(1, len(tokens), -1)
        attention_output, attention_weights = self.attention.forward(embeddings_batch)
        
        print(f"Attention output shape: {attention_output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Show attention patterns
        self.visualize_attention(tokens, attention_weights[0, 0])  # First head
        
        return embeddings, attention_weights
    
    def visualize_attention(self, tokens, attention_matrix):
        """Visualize attention patterns"""
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            annot=True,
            fmt='.2f',
            cmap='Blues'
        )
        
        plt.title('BERT Attention Patterns (First Head)')
        plt.xlabel('Key/Value Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def compare_contextual_embeddings(self):
        """Compare how BERT handles the same word in different contexts"""
        
        print("\n" + "=" * 50)
        print("CONTEXTUAL COMPARISON")
        print("=" * 50)
        
        # Test sentences with ambiguous words
        test_sentences = [
            "I went to the bank to deposit money".split(),
            "We sat by the river bank".split(), 
            "The plane will bank to the left".split()
        ]
        
        embeddings_by_context = {}
        
        for sentence in test_sentences:
            print(f"\nProcessing: '{' '.join(sentence)}'")
            
            # Tokenize and get embeddings
            tokens = self.bert.tokenize_for_bert(sentence)
            token_ids = [self.bert.word_to_id.get(token, 1) for token in tokens]
            
            # Get embeddings
            sentence_embeddings = self.embeddings.forward(token_ids)
            
            # Find 'bank' token
            if 'bank' in tokens:
                bank_position = tokens.index('bank')
                bank_embedding = sentence_embeddings[bank_position]
                
                context_key = ' '.join(sentence)
                embeddings_by_context[context_key] = bank_embedding
                
                print(f"  Found 'bank' at position {bank_position}")
                print(f"  Embedding norm: {np.linalg.norm(bank_embedding):.3f}")
        
        # Compare similarities
        if len(embeddings_by_context) > 1:
            print(f"\nContextual similarity comparison:")
            
            contexts = list(embeddings_by_context.keys())
            embeddings = list(embeddings_by_context.values())
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    print(f"  Context {i+1} vs Context {j+1}: {similarity:.3f}")
                    print(f"    '{contexts[i][:30]}...' vs '{contexts[j][:30]}...'")

# Demonstrate BERT concepts
demo = BERTDemo()

# Create sample sentences
bert_sentences = [
    "The bank approved my loan application".split(),
    "We walked along the river bank".split(),
    "The airplane will bank left soon".split(),
    "She opened a bank account".split(),
    "The muddy bank was slippery".split(),
    "I like to eat apples".split(),
    "Apple makes great computers".split(),
    "The key opens the door".split(),
    "This is the key point".split(),
    "Music makes me happy".split()
] * 10

# Run demonstration
embeddings, attention = demo.train_bert_concepts(bert_sentences)
demo.compare_contextual_embeddings()
```

## 🔧 Fine-tuning BERT for Specific Tasks

One of BERT's greatest strengths is its ability to be fine-tuned for specific tasks:

```python
class BERTFineTuner:
    """Demonstrate BERT fine-tuning concepts"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
        self.task_heads = {}
        
    def add_classification_head(self, num_classes, hidden_size=128):
        """Add classification head for sentiment analysis, etc."""
        
        # Simple classification head (linear layer)
        classification_weights = np.random.normal(0, 0.02, (hidden_size, num_classes))
        self.task_heads['classification'] = classification_weights
        
        print(f"Added classification head for {num_classes} classes")
    
    def add_token_classification_head(self, num_labels, hidden_size=128):
        """Add token classification head for NER, POS tagging, etc."""
        
        # Token-level classification
        token_weights = np.random.normal(0, 0.02, (hidden_size, num_labels))
        self.task_heads['token_classification'] = token_weights
        
        print(f"Added token classification head for {num_labels} labels")
    
    def add_qa_head(self, hidden_size=128):
        """Add question answering head"""
        
        # Start and end position prediction
        start_weights = np.random.normal(0, 0.02, (hidden_size, 1))
        end_weights = np.random.normal(0, 0.02, (hidden_size, 1))
        
        self.task_heads['qa_start'] = start_weights
        self.task_heads['qa_end'] = end_weights
        
        print("Added question answering head")
    
    def sentiment_analysis_example(self):
        """Demonstrate sentiment analysis fine-tuning"""
        
        print("\n" + "=" * 50)
        print("SENTIMENT ANALYSIS FINE-TUNING")
        print("=" * 50)
        
        # Add classification head for sentiment (positive/negative)
        self.add_classification_head(num_classes=2)
        
        # Example sentences with sentiment labels
        examples = [
            ("I love this movie", 1),  # Positive
            ("This is terrible", 0),   # Negative
            ("Great performance", 1),  # Positive
            ("Boring and slow", 0),    # Negative
        ]
        
        print("\nExample predictions (random weights):")
        
        for text, true_label in examples:
            # Tokenize
            tokens = self.bert_model.tokenize_for_bert(text.split())
            
            # Get BERT embeddings (use [CLS] token for classification)
            token_ids = [self.bert_model.word_to_id.get(token, 1) for token in tokens]
            embeddings = BERTEmbeddings(
                vocab_size=len(self.bert_model.word_to_id),
                hidden_size=128,
                max_position_embeddings=512
            ).forward(token_ids)
            
            # Use [CLS] token embedding for classification
            cls_embedding = embeddings[0]  # First token is [CLS]
            
            # Apply classification head
            logits = np.dot(cls_embedding, self.task_heads['classification'])
            
            # Softmax for probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            predicted_label = np.argmax(probs)
            
            sentiment = "Positive" if predicted_label == 1 else "Negative"
            confidence = probs[predicted_label]
            
            print(f"  Text: '{text}'")
            print(f"  True: {'Positive' if true_label == 1 else 'Negative'}")
            print(f"  Predicted: {sentiment} (confidence: {confidence:.3f})")
            print()
    
    def named_entity_recognition_example(self):
        """Demonstrate NER fine-tuning"""
        
        print("\n" + "=" * 50)
        print("NAMED ENTITY RECOGNITION FINE-TUNING")
        print("=" * 50)
        
        # NER labels: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
        ner_labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        self.add_token_classification_head(num_labels=len(ner_labels))
        
        # Example sentence
        sentence = "John works at Google in California".split()
        true_labels = ['B-PER', 'O', 'O', 'B-ORG', 'O', 'B-LOC']
        
        print(f"Example sentence: {' '.join(sentence)}")
        print(f"True NER labels: {true_labels}")
        
        # Tokenize for BERT
        tokens = self.bert_model.tokenize_for_bert(sentence)
        print(f"BERT tokens: {tokens}")
        
        # Get embeddings for each token
        token_ids = [self.bert_model.word_to_id.get(token, 1) for token in tokens]
        embeddings = BERTEmbeddings(
            vocab_size=len(self.bert_model.word_to_id),
            hidden_size=128,
            max_position_embeddings=512
        ).forward(token_ids)
        
        print(f"\nToken-level predictions (random weights):")
        
        # Predict for each token (skip [CLS] and [SEP])
        for i, token in enumerate(tokens[1:-1], 1):  # Skip [CLS] and [SEP]
            token_embedding = embeddings[i]
            
            # Apply token classification head
            logits = np.dot(token_embedding, self.task_heads['token_classification'])
            predicted_label_idx = np.argmax(logits)
            predicted_label = ner_labels[predicted_label_idx]
            confidence = np.max(np.exp(logits) / np.sum(np.exp(logits)))
            
            print(f"  {token}: {predicted_label} (confidence: {confidence:.3f})")
    
    def question_answering_example(self):
        """Demonstrate question answering fine-tuning"""
        
        print("\n" + "=" * 50)
        print("QUESTION ANSWERING FINE-TUNING")
        print("=" * 50)
        
        # Add QA head
        self.add_qa_head()
        
        # Example QA pair
        context = "John works at Google in California. He loves his job."
        question = "Where does John work?"
        answer = "Google"
        
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"True Answer: {answer}")
        
        # Combine question and context for BERT
        # Format: [CLS] question [SEP] context [SEP]
        question_tokens = question.split()
        context_tokens = context.split()
        
        combined_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
        
        print(f"Combined tokens: {combined_tokens}")
        
        # Get embeddings
        token_ids = [self.bert_model.word_to_id.get(token, 1) for token in combined_tokens]
        embeddings = BERTEmbeddings(
            vocab_size=len(self.bert_model.word_to_id),
            hidden_size=128,
            max_position_embeddings=512
        ).forward(token_ids)
        
        print(f"\nStart/End position predictions (random weights):")
        
        # Predict start and end positions
        start_logits = []
        end_logits = []
        
        for i, embedding in enumerate(embeddings):
            start_score = np.dot(embedding, self.task_heads['qa_start'].flatten())
            end_score = np.dot(embedding, self.task_heads['qa_end'].flatten())
            
            start_logits.append(start_score)
            end_logits.append(end_score)
        
        # Find best start and end positions
        start_pos = np.argmax(start_logits)
        end_pos = np.argmax(end_logits)
        
        print(f"  Predicted start position: {start_pos} (token: '{combined_tokens[start_pos]}')")
        print(f"  Predicted end position: {end_pos} (token: '{combined_tokens[end_pos]}')")
        
        if start_pos <= end_pos:
            predicted_answer = ' '.join(combined_tokens[start_pos:end_pos+1])
            print(f"  Predicted answer: '{predicted_answer}'")
        else:
            print("  Invalid span predicted")

# Demonstrate fine-tuning
print("\n" + "=" * 60)
print("BERT FINE-TUNING DEMONSTRATION")
print("=" * 60)

# Initialize fine-tuner with our BERT model
fine_tuner = BERTFineTuner(demo.bert)

# Run different fine-tuning examples
fine_tuner.sentiment_analysis_example()
fine_tuner.named_entity_recognition_example()
fine_tuner.question_answering_example()
```

## 🎯 BERT vs Traditional Embeddings Comparison

Let's create a comprehensive comparison:

```python
class EmbeddingComparison:
    """Compare traditional vs contextual embeddings"""
    
    def __init__(self):
        self.results = {}
        
    def polysemy_test(self):
        """Test how different methods handle polysemous words"""
        
        print("\n" + "=" * 60)
        print("POLYSEMY HANDLING COMPARISON")
        print("=" * 60)
        
        # Test cases with polysemous words
        test_cases = [
            {
                'word': 'bank',
                'contexts': [
                    "I deposited money at the bank",
                    "We sat by the river bank", 
                    "The plane will bank left"
                ],
                'expected_differences': "High - completely different meanings"
            },
            {
                'word': 'apple',
                'contexts': [
                    "I ate a red apple",
                    "Apple released a new iPhone",
                    "The apple tree is blooming"
                ],
                'expected_differences': "Medium - fruit vs company vs tree"
            },
            {
                'word': 'key',
                'contexts': [
                    "I lost my house key",
                    "This is the key point",
                    "Press the escape key"
                ],
                'expected_differences': "High - physical object vs important point vs keyboard"
            }
        ]
        
        for test_case in test_cases:
            word = test_case['word']
            contexts = test_case['contexts']
            expected = test_case['expected_differences']
            
            print(f"\nTesting word: '{word}'")
            print(f"Expected differences: {expected}")
            
            print("\nContexts:")
            for i, context in enumerate(contexts, 1):
                print(f"  {i}. {context}")
            
            # Traditional embeddings (same vector for all contexts)
            print("\nTraditional embeddings:")
            print("  All contexts get identical vector for 'bank'")
            print("  Similarity between any two contexts: 1.000")
            
            # Contextual embeddings (different vectors)
            print("\nContextual embeddings (simulated):")
            # Simulate different contextual embeddings
            contextual_embeddings = []
            for _ in contexts:
                # Create different random vectors to simulate contextual differences
                embedding = np.random.normal(0, 1, 100)
                contextual_embeddings.append(embedding)
            
            # Calculate similarities between contexts
            for i in range(len(contextual_embeddings)):
                for j in range(i+1, len(contextual_embeddings)):
                    sim = np.dot(contextual_embeddings[i], contextual_embeddings[j]) / (
                        np.linalg.norm(contextual_embeddings[i]) * 
                        np.linalg.norm(contextual_embeddings[j])
                    )
                    print(f"  Context {i+1} vs Context {j+1}: {sim:.3f}")
    
    def context_sensitivity_analysis(self):
        """Analyze how context affects word representations"""
        
        print("\n" + "=" * 60)
        print("CONTEXT SENSITIVITY ANALYSIS")  
        print("=" * 60)
        
        # Example: How context changes meaning
        examples = [
            {
                'target_word': 'light',
                'contexts': [
                    'The light is bright',      # illumination
                    'The bag is very light',    # weight
                    'Light the candle',         # verb: to ignite
                    'He has light hair'         # color
                ]
            },
            {
                'target_word': 'run',
                'contexts': [
                    'I like to run fast',       # physical activity
                    'Run the program',          # execute
                    'The river runs south',     # flow/direction
                    'She runs the company'      # manage
                ]
            }
        ]
        
        for example in examples:
            word = example['target_word']
            contexts = example['contexts']
            
            print(f"\nTarget word: '{word}'")
            print("Demonstrating context sensitivity:")
            
            for i, context in enumerate(contexts, 1):
                print(f"  {i}. {context}")
                # In a real contextual model, each would produce different embeddings
                print(f"     → Contextual embedding_{i} (unique vector)")
            
            print(f"\n  Traditional: Same vector for all uses of '{word}'")
            print(f"  Contextual: {len(contexts)} different vectors for '{word}'")
    
    def create_evaluation_summary(self):
        """Create comprehensive evaluation summary"""
        
        print("\n" + "=" * 80)
        print("EMBEDDING COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = {
            'Aspect': [
                'Polysemy Handling',
                'Context Sensitivity', 
                'Training Data Requirements',
                'Computational Cost',
                'Memory Usage',
                'Inference Speed',
                'Out-of-Vocabulary',
                'Transfer Learning',
                'Interpretability'
            ],
            'Traditional (Word2Vec/GloVe)': [
                'Poor - same vector for all meanings',
                'None - no context awareness',
                'Moderate - word co-occurrence',
                'Low - simple training',
                'Low - one vector per word',
                'Fast - simple lookup',
                'Cannot handle (except FastText)',
                'Limited - task-specific training',
                'High - clear word vectors'
            ],
            'Contextual (ELMo/BERT)': [
                'Excellent - different vectors per context',
                'High - context-dependent vectors',
                'High - large text corpora',
                'High - complex architecture',
                'High - multiple layers',
                'Slower - forward pass required',
                'Good - subword tokenization',
                'Excellent - pre-train then fine-tune',
                'Lower - complex interactions'
            ]
        }
        
        # Print comparison table
        for i, aspect in enumerate(comparison_data['Aspect']):
            print(f"\n{aspect}:")
            print(f"  Traditional: {comparison_data['Traditional (Word2Vec/GloVe)'][i]}")
            print(f"  Contextual:  {comparison_data['Contextual (ELMo/BERT)'][i]}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = [
            "Use BERT/contextual embeddings when:",
            "  • Working with polysemous words",
            "  • Context is crucial for understanding",
            "  • You have sufficient computational resources",
            "  • Transfer learning is important",
            "",
            "Use traditional embeddings when:",
            "  • Working with limited resources",
            "  • Speed is critical",
            "  • Words have relatively stable meanings",
            "  • You need interpretable word representations"
        ]
        
        for rec in recommendations:
            print(rec)

# Run comprehensive comparison
comparison = EmbeddingComparison()
comparison.polysemy_test()
comparison.context_sensitivity_analysis()
comparison.create_evaluation_summary()
```

## 🏋️ Practice Exercise

## Practice Exercise: Build a Contextual Word Sense Disambiguation System

Create a system that can determine which meaning of a polysemous word is being used:

```python
def build_word_sense_disambiguation_system():
    """
    Build a system that disambiguates word senses using contextual embeddings
    
    Requirements:
    1. Implement contextual embedding extraction
    2. Create sense clustering for polysemous words
    3. Build classification system for sense prediction
    4. Evaluate on word sense disambiguation tasks
    5. Compare with traditional disambiguation methods
    6. Handle unknown senses and words
    
    Bonus:
    - Multi-lingual word sense disambiguation
    - Domain-specific sense adaptation
    - Interactive sense discovery
    - Real-time disambiguation API
    """
    
    # Your implementation here
    pass

# Test your system with these challenging cases
test_cases = [
    {
        'word': 'bark',
        'senses': ['dog sound', 'tree covering', 'ship type'],
        'test_sentences': [
            "The dog's bark was loud",
            "The bark of the oak tree was rough", 
            "The bark sailed across the ocean"
        ]
    },
    {
        'word': 'spring',
        'senses': ['season', 'water source', 'mechanical device', 'jump'],
        'test_sentences': [
            "Spring is my favorite season",
            "We found a natural spring",
            "The spring in the watch broke",
            "She can spring very high"
        ]
    }
]
```

## 💡 Key Takeaways

1. **Contextual embeddings solve polysemy** - Same word, different meanings in different contexts
2. **ELMo uses bidirectional LSTMs** - Context from both directions
3. **BERT uses transformer attention** - Every word attends to every other word
4. **Fine-tuning is powerful** - Pre-trained BERT adapts to specific tasks
5. **Context window matters** - Larger context = better understanding
6. **Computational trade-offs exist** - More powerful = more expensive

## 🚀 What's Next?

You've mastered contextual embeddings! Next, explore [Word Embedding Applications](./06_applications.md) to see how to apply these techniques to real-world problems.

**Coming up:**

- Semantic search systems
- Document clustering and classification
- Recommendation systems using embeddings
- Bias detection and mitigation in embeddings

Ready to build real applications? Let's continue!
