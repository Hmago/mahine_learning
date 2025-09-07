# Neural Networks for Text Classification

## 🎯 What You'll Learn

Neural networks have revolutionized text classification by automatically learning complex patterns that traditional algorithms might miss. You'll discover when and how to use deep learning for text, from simple feedforward networks to sophisticated architectures.

## 🧠 Why Neural Networks for Text?

Think of traditional ML as a skilled craftsperson with specific tools, while neural networks are like having an entire workshop that can create custom tools for each job. They excel at:

- **Learning complex patterns** → Finding subtle relationships between words
- **Handling large vocabularies** → Processing millions of unique words
- **Capturing context** → Understanding word meaning based on surrounding text
- **Transfer learning** → Using pre-trained knowledge from massive datasets

## 🏗️ The Neural Network Hierarchy for Text

### 1. Feedforward Networks: The Foundation

**What they do:** Process text as a bag of features through layers of neurons.

**Think of it as:** A sophisticated voting system where each word contributes to the final decision through multiple rounds of weighted voting.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralTextClassifier:
    """A simple neural network for text classification"""
    
    def __init__(self, max_words=10000, max_length=100, embedding_dim=100):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def preprocess_texts(self, texts, labels=None, is_training=True):
        """Convert texts to sequences and encode labels"""
        
        if is_training:
            # Fit tokenizer on training data
            self.tokenizer.fit_on_texts(texts)
            
            # Encode labels
            if labels is not None:
                encoded_labels = self.label_encoder.fit_transform(labels)
                self.num_classes = len(self.label_encoder.classes_)
                labels_categorical = tf.keras.utils.to_categorical(encoded_labels, self.num_classes)
            else:
                labels_categorical = None
        else:
            # Transform labels for test data
            if labels is not None:
                encoded_labels = self.label_encoder.transform(labels)
                labels_categorical = tf.keras.utils.to_categorical(encoded_labels, self.num_classes)
            else:
                labels_categorical = None
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, 
                                       padding='post', truncating='post')
        
        return padded_sequences, labels_categorical
    
    def build_model(self):
        """Build a simple feedforward neural network"""
        
        self.model = Sequential([
            # Embedding layer - converts words to dense vectors
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # Global average pooling - averages all word embeddings
            GlobalAveragePooling1D(),
            
            # Hidden layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """Train the neural network"""
        
        # Preprocess data
        X, y = self.preprocess_texts(texts, labels, is_training=True)
        
        # Build model
        self.build_model()
        
        print(f"Model architecture:")
        self.model.summary()
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X, _ = self.preprocess_texts(texts, is_training=False)
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example: Movie Review Sentiment Analysis
movie_reviews = [
    "This movie is absolutely incredible! The acting is superb and the plot is engaging throughout.",
    "Terrible film. The story makes no sense and the acting is wooden and unconvincing.",
    "A decent movie with some good moments, but nothing spectacular. Worth watching once.",
    "Awful cinematography and poor character development. Complete waste of time.",
    "Outstanding performance by the lead actor. One of the best films I've seen this year!",
    "The movie has its moments but overall feels rushed and poorly executed.",
    "Brilliant storytelling and amazing visual effects. Highly recommend to everyone!",
    "Disappointing sequel that doesn't live up to the original. Skip this one.",
    "Entertaining and well-made film with great attention to detail throughout.",
    "Poor script and terrible pacing. The director clearly had no vision for this project."
] * 20  # Repeat to have more training data

sentiments = ["positive", "negative", "neutral", "negative", "positive", 
              "neutral", "positive", "negative", "positive", "negative"] * 20

# Create and train the neural classifier
nn_classifier = SimpleNeuralTextClassifier(max_words=5000, max_length=50)
history = nn_classifier.train(movie_reviews, sentiments, epochs=15)

# Plot training progress
nn_classifier.plot_training_history(history)

# Test with new reviews
test_reviews = [
    "This film exceeded all my expectations! Absolutely amazing cinematography and acting.",
    "Not bad, but nothing special. Average movie that's okay for a weekend watch.",
    "Horrible movie with terrible acting and a nonsensical plot. Don't waste your time."
]

predictions, probabilities = nn_classifier.predict(test_reviews)

print("\nNeural Network Predictions:")
for review, pred, probs in zip(test_reviews, predictions, probabilities):
    print(f"Review: '{review[:50]}...'")
    print(f"Prediction: {pred}")
    print("Probabilities:")
    for i, class_name in enumerate(nn_classifier.label_encoder.classes_):
        print(f"  {class_name}: {probs[i]:.3f}")
    print()
```

### 2. Convolutional Neural Networks (CNNs): The Pattern Detectors

**What they do:** Use filters to detect local patterns in text, like n-grams, but learn them automatically.

**Think of it as:** A detective with multiple magnifying glasses, each specialized to find different types of clues in the text.

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

class CNNTextClassifier:
    """CNN-based text classifier for capturing local patterns"""
    
    def __init__(self, max_words=10000, max_length=100, embedding_dim=100):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def build_cnn_model(self):
        """Build a CNN architecture for text classification"""
        
        self.model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # Multiple CNN layers with different filter sizes
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_multi_filter_cnn(self):
        """Build a CNN with multiple filter sizes (like Kim's CNN)"""
        from tensorflow.keras.layers import Input, Concatenate
        from tensorflow.keras.models import Model
        
        # Input layer
        input_layer = Input(shape=(self.max_length,))
        embedding = Embedding(self.max_words, self.embedding_dim)(input_layer)
        
        # Multiple filter sizes
        filter_sizes = [3, 4, 5]
        conv_blocks = []
        
        for filter_size in filter_sizes:
            conv = Conv1D(filters=100, kernel_size=filter_size, activation='relu')(embedding)
            pool = MaxPooling1D(pool_size=self.max_length - filter_size + 1)(conv)
            flatten = Flatten()(pool)
            conv_blocks.append(flatten)
        
        # Concatenate all filter outputs
        concat = Concatenate()(conv_blocks)
        
        # Dense layers
        dense = Dense(128, activation='relu')(concat)
        dropout = Dropout(0.5)(dense)
        output = Dense(self.num_classes, activation='softmax')(dropout)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

# Example: News Category Classification with CNN
news_texts = [
    "Apple announces breakthrough in artificial intelligence technology for mobile devices",
    "Stock market reaches all-time high as investors show confidence in economic recovery",
    "Olympic swimmer breaks world record in 200-meter freestyle competition",
    "New medical research offers hope for patients with chronic heart disease",
    "Cryptocurrency prices surge following major institutional adoption announcement",
    "Professional basketball team signs star player to record-breaking contract",
    "Scientists develop revolutionary gene therapy treatment for rare genetic disorders",
    "Tech startup raises $100 million in Series B funding for AI platform development"
] * 25  # More training data

news_labels = ["technology", "business", "sports", "health", 
               "business", "sports", "health", "technology"] * 25

# Create CNN classifier
cnn_classifier = CNNTextClassifier(max_words=5000, max_length=50)

# Preprocess data
X, y = cnn_classifier.preprocess_texts(news_texts, news_labels, is_training=True)

# Build and train model
cnn_classifier.build_multi_filter_cnn()
print("CNN Model Architecture:")
cnn_classifier.model.summary()

# Train the model
history = cnn_classifier.model.fit(
    X, y,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    verbose=1
)

# Visualize learned filters
def visualize_cnn_filters(model, word_index, max_words_to_show=10):
    """Visualize what the CNN filters learned"""
    
    # Get the embedding weights
    embedding_layer = model.layers[1]  # Assuming embedding is second layer
    embeddings = embedding_layer.get_weights()[0]
    
    # Get conv layer weights
    conv_layer = model.layers[2]  # First conv layer
    conv_weights = conv_layer.get_weights()[0]  # Shape: (filter_size, embedding_dim, num_filters)
    
    print("CNN Filter Analysis:")
    print(f"Filter shape: {conv_weights.shape}")
    print(f"Number of filters: {conv_weights.shape[2]}")
    
    # Analyze first few filters
    for filter_idx in range(min(3, conv_weights.shape[2])):
        filter_weights = conv_weights[:, :, filter_idx]
        print(f"\nFilter {filter_idx + 1}:")
        print(f"  Weight magnitude: {np.linalg.norm(filter_weights):.3f}")

# Analyze the trained CNN
visualize_cnn_filters(cnn_classifier.model, cnn_classifier.tokenizer.word_index)
```

### 3. Recurrent Neural Networks (RNNs): The Memory Masters

**What they do:** Process text sequentially, maintaining memory of previous words to understand context.

**Think of it as:** A reader who remembers everything they've read so far and uses that memory to understand each new word.

```python
from tensorflow.keras.layers import LSTM, GRU, Bidirectional

class RNNTextClassifier:
    """RNN-based text classifier for sequential understanding"""
    
    def __init__(self, max_words=10000, max_length=100, embedding_dim=100):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def build_lstm_model(self):
        """Build an LSTM model for sequence classification"""
        
        self.model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # LSTM layers
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_bidirectional_lstm(self):
        """Build a bidirectional LSTM that reads text in both directions"""
        
        self.model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
            Bidirectional(LSTM(32, dropout=0.2)),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_gru_model(self):
        """Build a GRU model (faster alternative to LSTM)"""
        
        self.model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # GRU layers
            GRU(128, return_sequences=True, dropout=0.2),
            GRU(64, dropout=0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

# Example: Long Text Classification with Context
long_texts = [
    """The latest artificial intelligence breakthrough has significant implications for the technology 
    industry. Companies are investing billions in machine learning research, hoping to gain competitive 
    advantages in areas like natural language processing, computer vision, and automated decision making. 
    This technological revolution is transforming how businesses operate across all sectors.""",
    
    """Economic indicators suggest that the stock market may experience volatility in the coming months. 
    Analysts are closely monitoring inflation rates, employment figures, and corporate earnings reports. 
    The Federal Reserve's monetary policy decisions will likely influence investor sentiment and market 
    trends throughout the remainder of the fiscal year.""",
    
    """The championship game drew millions of viewers as two legendary teams competed for the title. 
    Star athletes demonstrated exceptional skill and determination throughout the intense competition. 
    Record-breaking performances and dramatic moments made this one of the most memorable sporting events 
    in recent history, cementing its place in sports folklore.""",
    
    """Medical researchers have made significant progress in developing new treatments for chronic diseases. 
    Clinical trials are showing promising results for innovative therapies that could improve patient 
    outcomes and quality of life. The healthcare industry is optimistic about the potential impact of 
    these breakthroughs on public health and disease prevention strategies."""
] * 30  # Repeat for more training data

long_labels = ["technology", "business", "sports", "health"] * 30

# Create RNN classifier
rnn_classifier = RNNTextClassifier(max_words=5000, max_length=100)

# Preprocess data
X, y = rnn_classifier.preprocess_texts(long_texts, long_labels, is_training=True)

# Try different RNN architectures
print("Training LSTM Model:")
rnn_classifier.build_lstm_model()
lstm_history = rnn_classifier.model.fit(X, y, validation_split=0.2, epochs=15, batch_size=16, verbose=1)

print("\nTraining Bidirectional LSTM Model:")
rnn_classifier.build_bidirectional_lstm()
bi_lstm_history = rnn_classifier.model.fit(X, y, validation_split=0.2, epochs=15, batch_size=16, verbose=1)

# Compare performance
def compare_models_performance():
    """Compare different neural architectures"""
    
    models = {
        'Simple NN': SimpleNeuralTextClassifier(),
        'CNN': CNNTextClassifier(),
        'RNN (LSTM)': RNNTextClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Simple NN':
            history = model.train(long_texts, long_labels, epochs=10)
        else:
            X, y = model.preprocess_texts(long_texts, long_labels, is_training=True)
            if name == 'CNN':
                model.build_multi_filter_cnn()
            else:  # RNN
                model.build_lstm_model()
            
            history = model.model.fit(X, y, validation_split=0.2, epochs=10, 
                                    batch_size=16, verbose=0)
        
        # Get final validation accuracy
        final_val_acc = history.history['val_accuracy'][-1]
        results[name] = final_val_acc
        
        print(f"{name} final validation accuracy: {final_val_acc:.3f}")
    
    # Display results
    print("\nModel Comparison Results:")
    print("-" * 40)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, accuracy) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name}: {accuracy:.3f}")

# Run comparison (uncomment to execute)
# compare_models_performance()
```

## 🎯 Choosing the Right Neural Architecture

### Decision Framework

```python
def choose_neural_architecture(text_length, context_importance, speed_requirement, data_size):
    """Help choose the best neural network architecture"""
    
    recommendations = []
    
    # Text length consideration
    if text_length == 'short':  # < 50 words
        recommendations.append(('Simple NN', 3))
        recommendations.append(('CNN', 3))
    elif text_length == 'medium':  # 50-200 words
        recommendations.append(('CNN', 3))
        recommendations.append(('RNN', 2))
    else:  # > 200 words
        recommendations.append(('RNN', 3))
        recommendations.append(('Bidirectional RNN', 2))
    
    # Context importance
    if context_importance == 'high':
        recommendations.append(('RNN', 3))
        recommendations.append(('Bidirectional RNN', 3))
    else:
        recommendations.append(('CNN', 2))
        recommendations.append(('Simple NN', 1))
    
    # Speed requirement
    if speed_requirement == 'high':
        recommendations.append(('Simple NN', 3))
        recommendations.append(('CNN', 2))
    else:
        recommendations.append(('RNN', 1))
    
    # Data size
    if data_size < 1000:
        recommendations.append(('Simple NN', 2))
    elif data_size < 10000:
        recommendations.append(('CNN', 2))
    else:
        recommendations.append(('RNN', 2))
    
    # Calculate scores
    scores = {}
    for arch, score in recommendations:
        scores[arch] = scores.get(arch, 0) + score
    
    best_architecture = max(scores.keys(), key=lambda x: scores[x])
    
    print(f"Recommended architecture: {best_architecture}")
    print("\nArchitecture scores:")
    for arch, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {arch}: {score}")
    
    return best_architecture

# Example usage
recommended_arch = choose_neural_architecture(
    text_length='medium',
    context_importance='high',
    speed_requirement='medium',
    data_size=5000
)
```

## 🚀 Transfer Learning: Standing on Giants' Shoulders

### Using Pre-trained Embeddings

```python
def load_pretrained_embeddings(embedding_path, word_index, embedding_dim=100):
    """Load pre-trained word embeddings (like GloVe)"""
    
    print("Loading pre-trained embeddings...")
    embeddings_index = {}
    
    # This would load actual GloVe embeddings
    # For demo, we'll create random embeddings
    vocab_size = len(word_index) + 1
    embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim))
    
    print(f"Created embedding matrix of shape: {embedding_matrix.shape}")
    return embedding_matrix

class TransferLearningClassifier:
    """Text classifier using pre-trained embeddings"""
    
    def __init__(self, embedding_path=None):
        self.embedding_path = embedding_path
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.label_encoder = LabelEncoder()
        
    def build_model_with_pretrained_embeddings(self, embedding_matrix):
        """Build model with pre-trained embeddings"""
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.model = Sequential([
            # Embedding layer with pre-trained weights
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=False,  # Freeze pre-trained embeddings
                input_length=self.max_length
            ),
            
            # Your choice of architecture
            LSTM(64, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

# Usage example
transfer_classifier = TransferLearningClassifier()
# embedding_matrix = load_pretrained_embeddings('glove.6B.100d.txt', word_index)
# model = transfer_classifier.build_model_with_pretrained_embeddings(embedding_matrix)
```

## 📊 Advanced Evaluation and Interpretation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def comprehensive_neural_evaluation(model, X_test, y_test, class_names, tokenizer):
    """Comprehensive evaluation of neural text classifier"""
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("Neural Network Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Neural Network Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Prediction confidence analysis
    confidences = np.max(predictions, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean Confidence: {np.mean(confidences):.3f}')
    plt.legend()
    plt.show()
    
    # Error analysis
    incorrect_indices = np.where(predicted_classes != true_classes)[0]
    
    print(f"\nError Analysis: {len(incorrect_indices)} misclassifications")
    print("Sample errors:")
    
    for i, idx in enumerate(incorrect_indices[:3]):
        confidence = confidences[idx]
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        
        # Decode text (simplified)
        text_length = np.sum(X_test[idx] != 0)  # Count non-zero tokens
        
        print(f"\nError {i+1}:")
        print(f"  Text length: {text_length} tokens")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print(f"  Confidence: {confidence:.3f}")

# Example usage (would need actual trained model and test data)
# comprehensive_neural_evaluation(model, X_test, y_test, class_names, tokenizer)
```

## 💡 Pro Tips for Neural Text Classification

### 1. Hyperparameter Tuning

```python
import keras_tuner as kt

def build_tunable_model(hp):
    """Build a hyperparameter-tunable model"""
    
    model = Sequential()
    
    # Tunable embedding dimension
    embedding_dim = hp.Int('embedding_dim', min_value=50, max_value=200, step=50)
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length))
    
    # Tunable architecture choice
    architecture = hp.Choice('architecture', values=['lstm', 'gru', 'cnn'])
    
    if architecture == 'lstm':
        units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        model.add(LSTM(units, dropout=hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    elif architecture == 'gru':
        units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
        model.add(GRU(units, dropout=hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    else:  # CNN
        filters = hp.Int('cnn_filters', min_value=32, max_value=128, step=32)
        kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(GlobalAveragePooling1D())
    
    # Tunable dense layers
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    # Tunable learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Hyperparameter search
tuner = kt.RandomSearch(
    build_tunable_model,
    objective='val_accuracy',
    max_trials=20,
    directory='hyperparam_search',
    project_name='text_classification'
)

# Search for best hyperparameters
# tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# best_model = tuner.get_best_models(num_models=1)[0]
```

### 2. Early Stopping and Model Checkpointing

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_with_callbacks(model, X_train, y_train, X_val, y_val):
    """Train model with advanced callbacks"""
    
    callbacks = [
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # High number, early stopping will handle it
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

## 🏋️ Practice Exercise

### Build a Multi-Class News Classifier

Create a neural network system that can classify news articles into multiple categories with high accuracy.

```python
def build_advanced_news_classifier():
    """
    Build an advanced neural network for news classification
    
    Requirements:
    1. Compare CNN, LSTM, and hybrid architectures
    2. Implement proper data preprocessing
    3. Use transfer learning with pre-trained embeddings
    4. Add comprehensive evaluation metrics
    5. Create prediction confidence analysis
    
    Bonus:
    - Implement attention mechanisms
    - Add ensemble of different architectures
    - Create interpretability visualizations
    - Build a Streamlit demo
    """
    
    # Your implementation here
    pass

# Test data with longer, more complex articles
complex_test_articles = [
    """Artificial intelligence researchers at leading technology companies have announced 
    breakthrough developments in natural language processing and computer vision. The new 
    algorithms demonstrate unprecedented accuracy in understanding human speech and interpreting 
    visual data, with potential applications spanning autonomous vehicles, medical diagnosis, 
    and automated customer service systems.""",
    
    """Global financial markets experienced significant volatility following the central bank's 
    unexpected policy announcement. Investment analysts are reassessing portfolio strategies 
    as interest rate changes impact various sectors differently. The technology and healthcare 
    industries showed resilience, while traditional manufacturing faced headwinds."""
]

# Your classifier should handle these complex, multi-sentence articles
```

## 💡 Key Takeaways

1. **Neural networks excel with large datasets** - Need thousands of examples to shine
2. **CNNs capture local patterns** - Great for feature detection in text
3. **RNNs understand sequence** - Essential for context-dependent classification
4. **Transfer learning accelerates training** - Use pre-trained embeddings when possible
5. **Proper evaluation is crucial** - Monitor both accuracy and confidence
6. **Hyperparameter tuning matters** - Can significantly improve performance

## 🚀 What's Next?

You've mastered neural approaches to text classification! Next, explore [Sentiment Analysis and Opinion Mining](./03_sentiment_analysis.md) to build systems that understand emotions and opinions in text.

**Coming up:**
- Advanced sentiment analysis techniques
- Aspect-based sentiment analysis
- Emotion detection beyond positive/negative
- Real-world applications in business intelligence

Ready to dive into the emotional side of text? Let's continue!
