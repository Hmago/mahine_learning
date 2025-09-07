# Advanced Classification Techniques

## 🎯 What You'll Learn

Advanced classification goes beyond basic algorithms to handle real-world challenges like imbalanced data, multi-label classification, and few-shot learning. You'll master cutting-edge techniques that power modern NLP applications.

## 🧠 Beyond Basic Classification: Real-World Challenges

In production systems, you'll encounter:

- **Imbalanced datasets** → 99% normal emails, 1% spam
- **Multi-label problems** → News articles with multiple topics
- **Few-shot learning** → New categories with minimal examples
- **Hierarchical classification** → Categories with sub-categories
- **Multi-lingual classification** → Handling multiple languages

## 🔥 Modern Classification Architectures

### 1. Transformer-Based Classification: The State-of-the-Art

**What makes transformers special:** They understand context through attention mechanisms, allowing them to focus on relevant parts of text for classification.

**Think of it as:** A super-intelligent reader who can simultaneously pay attention to every word while understanding how they all relate to each other.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class TransformerTextClassifier:
    """Advanced text classification using transformer models"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.classifier = None
        
    def create_dataset(self, texts, labels=None):
        """Create a PyTorch dataset for training/inference"""
        
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                item = {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
                
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                
                return item
        
        return TextDataset(texts, labels, self.tokenizer)
    
    def train_model(self, train_texts, train_labels, val_texts=None, val_labels=None, 
                   output_dir="./results", epochs=3, batch_size=16):
        """Train the transformer model"""
        
        # Create datasets
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = self.create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_accuracy" if val_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    def predict(self, texts, use_pipeline=True):
        """Make predictions on new texts"""
        
        if use_pipeline:
            # Use pipeline for easy inference
            if self.classifier is None:
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            results = []
            for text in texts:
                result = self.classifier(text)
                results.append({
                    'text': text,
                    'predicted_label': result[0]['label'],
                    'confidence': result[0]['score']
                })
            
            return results
        else:
            # Manual prediction
            dataset = self.create_dataset(texts)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
            
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            
            return predictions

# Example: Advanced News Classification with BERT
news_articles = [
    "Apple announces groundbreaking AI chip that revolutionizes mobile computing performance",
    "Stock market soars as Federal Reserve signals potential interest rate cuts",
    "Olympic swimmer breaks world record in thrilling 200-meter freestyle final",
    "New cancer treatment shows 95% success rate in clinical trials",
    "Cryptocurrency market experiences massive volatility amid regulatory concerns",
    "Professional basketball season kicks off with record ticket sales",
    "Medical breakthrough offers hope for patients with rare genetic disorders",
    "Tech startup raises $100 million for quantum computing research",
    "Economic indicators suggest steady growth despite global uncertainties",
    "Tennis championship features youngest finalist in tournament history"
] * 10  # Repeat for more training data

news_categories = [0, 1, 2, 3, 1, 2, 3, 0, 1, 2] * 10  # 0=tech, 1=business, 2=sports, 3=health
category_names = ['technology', 'business', 'sports', 'health']

# Split data
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    news_articles, news_categories, test_size=0.2, random_state=42, stratify=news_categories
)

# Create and train transformer classifier
transformer_classifier = TransformerTextClassifier(
    model_name="distilbert-base-uncased",
    num_labels=4
)

# Note: Training transformers requires significant computational resources
# For demonstration, we'll show the setup
print("Transformer classifier setup complete!")
print(f"Model: {transformer_classifier.model_name}")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# trainer = transformer_classifier.train_model(
#     train_texts, train_labels, val_texts, val_labels
# )

# Test with new articles
test_articles = [
    "Revolutionary artificial intelligence breakthrough enables real-time language translation",
    "Global markets react positively to unexpected economic growth reports",
    "World Cup final attracts billions of viewers in historic match",
    "Gene therapy treatment shows remarkable results for inherited blindness"
]

# predictions = transformer_classifier.predict(test_articles)
print("\nTransformer predictions would be shown here after training...")
```

### 2. Multi-Label Classification: When Text Belongs to Multiple Categories

**Challenge:** A news article can be about both technology AND business.

**Solution:** Independent binary classifiers for each label or specialized multi-label architectures.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

class MultiLabelTextClassifier:
    """Handle texts that can belong to multiple categories"""
    
    def __init__(self, approach='binary_relevance'):
        self.approach = approach
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.mlb = MultiLabelBinarizer()
        self.classifier = None
        
    def prepare_multilabel_data(self, texts, label_sets):
        """Prepare data for multi-label classification"""
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Binarize labels
        y = self.mlb.fit_transform(label_sets)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Labels: {self.mlb.classes_}")
        
        return X, y
    
    def train_binary_relevance(self, X, y):
        """Train using binary relevance approach"""
        
        # Each label gets its own binary classifier
        self.classifier = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        
        self.classifier.fit(X, y)
        print("Binary relevance model trained!")
        
        return self.classifier
    
    def train_neural_multilabel(self, X, y):
        """Train neural network for multi-label classification"""
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        # Convert sparse matrix to dense for neural network
        X_dense = X.toarray()
        
        # Build neural network
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(y.shape[1], activation='sigmoid')  # Sigmoid for multi-label
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Binary crossentropy for multi-label
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_dense, y,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        self.classifier = model
        print("Neural multi-label model trained!")
        
        return model, history
    
    def predict_multilabel(self, texts, threshold=0.5):
        """Make multi-label predictions"""
        
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        if hasattr(self.classifier, 'predict'):
            # Scikit-learn classifier
            predictions = self.classifier.predict(X)
        else:
            # Neural network
            X_dense = X.toarray()
            predictions = self.classifier.predict(X_dense)
            predictions = (predictions > threshold).astype(int)
        
        # Convert back to label sets
        predicted_labels = self.mlb.inverse_transform(predictions)
        
        results = []
        for text, labels, probs in zip(texts, predicted_labels, predictions):
            # Calculate confidence scores
            if hasattr(self.classifier, 'predict_proba'):
                prob_scores = self.classifier.predict_proba(X[len(results):len(results)+1])[0]
                confidence_scores = {
                    label: prob_scores[i][1] for i, label in enumerate(self.mlb.classes_)
                }
            else:
                confidence_scores = {
                    label: probs[i] for i, label in enumerate(self.mlb.classes_)
                }
            
            results.append({
                'text': text,
                'predicted_labels': list(labels),
                'confidence_scores': confidence_scores
            })
        
        return results
    
    def evaluate_multilabel(self, X_test, y_test):
        """Comprehensive evaluation of multi-label classifier"""
        
        predictions = self.classifier.predict(X_test)
        
        # Overall metrics
        from sklearn.metrics import hamming_loss, jaccard_score, f1_score
        
        hamming = hamming_loss(y_test, predictions)
        jaccard = jaccard_score(y_test, predictions, average='samples')
        f1_micro = f1_score(y_test, predictions, average='micro')
        f1_macro = f1_score(y_test, predictions, average='macro')
        
        print("Multi-Label Classification Metrics:")
        print(f"Hamming Loss: {hamming:.3f} (lower is better)")
        print(f"Jaccard Score: {jaccard:.3f} (higher is better)")
        print(f"F1-Score (micro): {f1_micro:.3f}")
        print(f"F1-Score (macro): {f1_macro:.3f}")
        
        # Per-label analysis
        print("\nPer-Label Classification Report:")
        print(classification_report(y_test, predictions, target_names=self.mlb.classes_))
        
        # Confusion matrices for each label
        cm_multilabel = multilabel_confusion_matrix(y_test, predictions)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (label, cm) in enumerate(zip(self.mlb.classes_, cm_multilabel)):
            if i < len(axes):
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
                axes[i].set_title(f'Confusion Matrix: {label}')
                axes[i].set_ylabel('True Label')
                axes[i].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }

# Example: Multi-Label News Classification
multilabel_articles = [
    "Apple's new AI chip revolutionizes mobile technology while boosting stock prices significantly",  # tech + business
    "Olympic basketball tournament breaks viewership records with innovative streaming technology",      # sports + tech
    "Medical AI startup raises $50 million to develop cancer detection algorithms",                     # health + tech + business
    "Professional tennis player announces retirement due to recurring health issues",                   # sports + health
    "Federal Reserve's interest rate decision impacts technology sector investments",                    # business + tech
    "New sports medicine research reveals breakthrough in injury prevention and recovery",              # health + sports
    "Cryptocurrency market volatility affects tech company funding and investment strategies",          # business + tech
    "Olympic athlete uses advanced medical technology for performance enhancement and health monitoring", # sports + health + tech
    "Healthcare startups see increased investment following successful clinical trial results",          # health + business
    "Major tech conference discusses future of artificial intelligence in financial markets"            # tech + business
]

# Multi-label targets (each article can have multiple labels)
multilabel_targets = [
    ['technology', 'business'],
    ['sports', 'technology'],
    ['health', 'technology', 'business'],
    ['sports', 'health'],
    ['business', 'technology'],
    ['health', 'sports'],
    ['business', 'technology'],
    ['sports', 'health', 'technology'],
    ['health', 'business'],
    ['technology', 'business']
]

# Expand dataset for training
multilabel_articles_expanded = multilabel_articles * 10
multilabel_targets_expanded = multilabel_targets * 10

# Create multi-label classifier
multilabel_classifier = MultiLabelTextClassifier()

# Prepare data
X, y = multilabel_classifier.prepare_multilabel_data(
    multilabel_articles_expanded, 
    multilabel_targets_expanded
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train using binary relevance
multilabel_classifier.train_binary_relevance(X_train, y_train)

# Evaluate
metrics = multilabel_classifier.evaluate_multilabel(X_test, y_test)

# Test predictions
test_multilabel_articles = [
    "Revolutionary AI breakthrough in medical diagnosis attracts major investment from tech giants",
    "World Cup final streamed live using cutting-edge technology platform",
    "Biotech company's stock soars after successful drug trial results"
]

predictions = multilabel_classifier.predict_multilabel(test_multilabel_articles)

print("\nMulti-Label Predictions:")
for pred in predictions:
    print(f"Text: '{pred['text']}'")
    print(f"Predicted Labels: {pred['predicted_labels']}")
    print("Confidence Scores:")
    for label, score in pred['confidence_scores'].items():
        if score > 0.3:  # Only show confident predictions
            print(f"  {label}: {score:.3f}")
    print()
```

### 3. Few-Shot Learning: Learning from Limited Examples

**Challenge:** You have only 5 examples of a new category but need to classify it accurately.

**Solution:** Meta-learning approaches and transfer learning from pre-trained models.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FewShotTextClassifier:
    """Classify text with very few training examples"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.support_embeddings = {}
        self.support_labels = {}
        self.prototype_embeddings = {}
        
    def create_support_set(self, support_texts, support_labels):
        """Create embeddings for few-shot examples (support set)"""
        
        print("Creating support set embeddings...")
        
        # Group texts by label
        label_groups = {}
        for text, label in zip(support_texts, support_labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(text)
        
        # Create embeddings for each group
        for label, texts in label_groups.items():
            embeddings = self.model.encode(texts)
            self.support_embeddings[label] = embeddings
            self.support_labels[label] = texts
            
            # Create prototype (centroid) for this class
            self.prototype_embeddings[label] = np.mean(embeddings, axis=0)
        
        print(f"Support set created for {len(label_groups)} classes:")
        for label, texts in label_groups.items():
            print(f"  {label}: {len(texts)} examples")
    
    def predict_prototypical(self, query_texts):
        """Classify using prototypical networks approach"""
        
        if not self.prototype_embeddings:
            raise ValueError("Support set must be created first!")
        
        # Encode query texts
        query_embeddings = self.model.encode(query_texts)
        
        results = []
        
        for i, (text, query_emb) in enumerate(zip(query_texts, query_embeddings)):
            # Calculate distances to all prototypes
            distances = {}
            similarities = {}
            
            for label, prototype in self.prototype_embeddings.items():
                # Cosine similarity
                similarity = cosine_similarity([query_emb], [prototype])[0][0]
                similarities[label] = similarity
                
                # Euclidean distance
                distance = np.linalg.norm(query_emb - prototype)
                distances[label] = distance
            
            # Predict based on highest similarity
            predicted_label = max(similarities.keys(), key=lambda k: similarities[k])
            confidence = similarities[predicted_label]
            
            results.append({
                'text': text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'similarities': similarities,
                'distances': distances
            })
        
        return results
    
    def predict_knn(self, query_texts, k=3):
        """Classify using k-nearest neighbors in embedding space"""
        
        if not self.support_embeddings:
            raise ValueError("Support set must be created first!")
        
        # Combine all support embeddings and labels
        all_support_embeddings = []
        all_support_labels = []
        
        for label, embeddings in self.support_embeddings.items():
            all_support_embeddings.extend(embeddings)
            all_support_labels.extend([label] * len(embeddings))
        
        all_support_embeddings = np.array(all_support_embeddings)
        
        # Encode query texts
        query_embeddings = self.model.encode(query_texts)
        
        results = []
        
        for text, query_emb in zip(query_texts, query_embeddings):
            # Calculate similarities to all support examples
            similarities = cosine_similarity([query_emb], all_support_embeddings)[0]
            
            # Get k nearest neighbors
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_labels = [all_support_labels[i] for i in top_k_indices]
            top_k_similarities = [similarities[i] for i in top_k_indices]
            
            # Vote based on k nearest neighbors
            label_votes = {}
            for label, sim in zip(top_k_labels, top_k_similarities):
                if label not in label_votes:
                    label_votes[label] = []
                label_votes[label].append(sim)
            
            # Calculate weighted vote
            label_scores = {}
            for label, sims in label_votes.items():
                label_scores[label] = np.mean(sims)  # Average similarity
            
            predicted_label = max(label_scores.keys(), key=lambda k: label_scores[k])
            confidence = label_scores[predicted_label]
            
            results.append({
                'text': text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'knn_labels': top_k_labels,
                'knn_similarities': top_k_similarities,
                'label_scores': label_scores
            })
        
        return results
    
    def evaluate_few_shot(self, query_texts, query_labels, method='prototypical'):
        """Evaluate few-shot classification performance"""
        
        if method == 'prototypical':
            predictions = self.predict_prototypical(query_texts)
        else:
            predictions = self.predict_knn(query_texts)
        
        # Calculate accuracy
        correct = 0
        total = len(query_labels)
        
        predicted_labels = [pred['predicted_label'] for pred in predictions]
        
        for true_label, pred_label in zip(query_labels, predicted_labels):
            if true_label == pred_label:
                correct += 1
        
        accuracy = correct / total
        
        print(f"Few-Shot Classification Results ({method}):")
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Detailed results
        print("\nDetailed Results:")
        for i, (pred, true_label) in enumerate(zip(predictions, query_labels)):
            status = "✓" if pred['predicted_label'] == true_label else "✗"
            print(f"{status} Query {i+1}: True={true_label}, Pred={pred['predicted_label']} (conf={pred['confidence']:.3f})")
        
        return accuracy, predictions

# Example: Few-Shot Document Classification
# Scenario: Classify customer support tickets with only a few examples per category

# Support set (few examples for each class)
support_texts = [
    # Technical issues (2 examples)
    "My app keeps crashing when I try to upload photos. Can you help fix this?",
    "The website won't load on my browser. I'm getting error 500 messages.",
    
    # Billing issues (2 examples)  
    "I was charged twice for my subscription this month. Please refund the duplicate charge.",
    "My credit card was declined but I know there are sufficient funds. What's wrong?",
    
    # Account issues (2 examples)
    "I can't log into my account. I forgot my password and the reset isn't working.",
    "Someone else is using my account. I need to change my password immediately.",
    
    # Product questions (2 examples)
    "What's the difference between the premium and basic plans? Which one should I choose?",
    "Do you offer a student discount? I'm currently enrolled in university."
]

support_labels = [
    'technical', 'technical',
    'billing', 'billing', 
    'account', 'account',
    'product', 'product'
]

# Query set (new tickets to classify)
query_texts = [
    "The mobile app freezes every time I try to save my work.",
    "I need a refund for last month's payment that was processed in error.",
    "I'm locked out of my account after too many failed login attempts.",
    "Can you explain what features are included in the enterprise package?",
    "My subscription keeps auto-renewing even though I cancelled it.",
    "The dashboard is not loading any data. Everything appears blank.",
    "How do I upgrade my account to get access to premium features?",
    "I suspect my account has been compromised. Please help secure it."
]

query_labels = ['technical', 'billing', 'account', 'product', 'billing', 'technical', 'product', 'account']

# Create few-shot classifier
few_shot_classifier = FewShotTextClassifier()

# Create support set
few_shot_classifier.create_support_set(support_texts, support_labels)

# Test both methods
print("=== Prototypical Networks Approach ===")
accuracy_proto, predictions_proto = few_shot_classifier.evaluate_few_shot(
    query_texts, query_labels, method='prototypical'
)

print("\n=== K-Nearest Neighbors Approach ===")
accuracy_knn, predictions_knn = few_shot_classifier.evaluate_few_shot(
    query_texts, query_labels, method='knn'
)

# Compare methods
print(f"\n=== Method Comparison ===")
print(f"Prototypical Networks: {accuracy_proto:.3f}")
print(f"K-Nearest Neighbors: {accuracy_knn:.3f}")

# Show confidence analysis
print("\n=== Confidence Analysis ===")
proto_confidences = [pred['confidence'] for pred in predictions_proto]
knn_confidences = [pred['confidence'] for pred in predictions_knn]

print(f"Prototypical - Avg confidence: {np.mean(proto_confidences):.3f}, Std: {np.std(proto_confidences):.3f}")
print(f"KNN - Avg confidence: {np.mean(knn_confidences):.3f}, Std: {np.std(knn_confidences):.3f}")
```

### 4. Hierarchical Classification: Multi-Level Categories

**Challenge:** Categories have sub-categories (e.g., Sports → Football → NFL → Team News).

**Solution:** Hierarchical classifiers that predict at multiple levels simultaneously.

```python
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
import matplotlib.pyplot as plt

class HierarchicalTextClassifier:
    """Handle hierarchical classification with multiple levels"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.hierarchy = {}
        self.classifiers = {}
        self.hierarchy_graph = nx.DiGraph()
        
    def define_hierarchy(self, hierarchy_dict):
        """Define the category hierarchy"""
        
        self.hierarchy = hierarchy_dict
        
        # Build graph representation
        def add_nodes_recursive(parent, children):
            for child in children:
                self.hierarchy_graph.add_edge(parent, child)
                if isinstance(children[child], dict):
                    add_nodes_recursive(child, children[child])
        
        for root, children in hierarchy_dict.items():
            self.hierarchy_graph.add_node(root)
            if isinstance(children, dict):
                add_nodes_recursive(root, children)
        
        print(f"Hierarchy defined with {len(self.hierarchy_graph.nodes)} categories")
        print(f"Levels: {len(list(nx.topological_sort(self.hierarchy_graph)))}")
    
    def prepare_hierarchical_data(self, texts, hierarchical_labels):
        """Prepare data for hierarchical classification"""
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Organize labels by hierarchy level
        level_labels = {}
        
        for text_idx, label_path in enumerate(hierarchical_labels):
            # label_path is like ['sports', 'football', 'nfl']
            for level, label in enumerate(label_path):
                if level not in level_labels:
                    level_labels[level] = []
                
                # Pad with None for shorter paths
                while len(level_labels[level]) <= text_idx:
                    level_labels[level].append(None)
                
                level_labels[level][text_idx] = label
        
        return X, level_labels
    
    def train_hierarchical_classifiers(self, X, level_labels):
        """Train classifiers for each level of hierarchy"""
        
        for level, labels in level_labels.items():
            # Remove None values and corresponding samples
            valid_indices = [i for i, label in enumerate(labels) if label is not None]
            X_level = X[valid_indices]
            y_level = [labels[i] for i in valid_indices]
            
            if len(set(y_level)) > 1:  # Only train if multiple classes exist
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                classifier.fit(X_level, y_level)
                self.classifiers[level] = classifier
                
                print(f"Level {level} classifier trained with {len(set(y_level))} classes: {set(y_level)}")
    
    def predict_hierarchical(self, texts, strategy='top_down'):
        """Make hierarchical predictions"""
        
        X = self.vectorizer.transform(texts)
        results = []
        
        for i, text in enumerate(texts):
            sample = X[i:i+1]
            
            if strategy == 'top_down':
                prediction_path = self._predict_top_down(sample)
            else:  # independent
                prediction_path = self._predict_independent(sample)
            
            results.append({
                'text': text,
                'predicted_path': prediction_path,
                'strategy': strategy
            })
        
        return results
    
    def _predict_top_down(self, sample):
        """Predict using top-down approach (each level depends on previous)"""
        
        path = []
        current_level = 0
        
        while current_level in self.classifiers:
            classifier = self.classifiers[current_level]
            
            # Get prediction and probability
            prediction = classifier.predict(sample)[0]
            probabilities = classifier.predict_proba(sample)[0]
            confidence = max(probabilities)
            
            # Add to path if confident enough
            if confidence > 0.5:  # Confidence threshold
                path.append({
                    'level': current_level,
                    'category': prediction,
                    'confidence': confidence
                })
                current_level += 1
            else:
                break  # Stop if not confident
        
        return path
    
    def _predict_independent(self, sample):
        """Predict each level independently"""
        
        path = []
        
        for level in sorted(self.classifiers.keys()):
            classifier = self.classifiers[level]
            
            prediction = classifier.predict(sample)[0]
            probabilities = classifier.predict_proba(sample)[0]
            confidence = max(probabilities)
            
            path.append({
                'level': level,
                'category': prediction,
                'confidence': confidence
            })
        
        return path
    
    def visualize_hierarchy(self):
        """Visualize the category hierarchy"""
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.hierarchy_graph, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.hierarchy_graph, pos, 
                              node_color='lightblue', 
                              node_size=2000, 
                              alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.hierarchy_graph, pos, 
                              edge_color='gray', 
                              arrows=True, 
                              arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.hierarchy_graph, pos, 
                               font_size=10, 
                               font_weight='bold')
        
        plt.title("Category Hierarchy")
        plt.axis('off')
        plt.show()
    
    def evaluate_hierarchical(self, texts, true_paths, strategy='top_down'):
        """Evaluate hierarchical classification"""
        
        predictions = self.predict_hierarchical(texts, strategy)
        
        # Level-wise accuracy
        level_accuracies = {}
        
        for pred, true_path in zip(predictions, true_paths):
            pred_path = pred['predicted_path']
            
            for level in range(max(len(pred_path), len(true_path))):
                if level not in level_accuracies:
                    level_accuracies[level] = {'correct': 0, 'total': 0}
                
                level_accuracies[level]['total'] += 1
                
                # Check if prediction exists and is correct
                if (level < len(pred_path) and 
                    level < len(true_path) and 
                    pred_path[level]['category'] == true_path[level]):
                    level_accuracies[level]['correct'] += 1
        
        # Calculate accuracies
        print(f"Hierarchical Classification Evaluation ({strategy}):")
        print("-" * 50)
        
        for level, stats in level_accuracies.items():
            accuracy = stats['correct'] / stats['total']
            print(f"Level {level}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # Overall path accuracy (exact match)
        exact_matches = 0
        for pred, true_path in zip(predictions, true_paths):
            pred_categories = [step['category'] for step in pred['predicted_path']]
            if pred_categories == true_path:
                exact_matches += 1
        
        path_accuracy = exact_matches / len(predictions)
        print(f"\nExact Path Accuracy: {path_accuracy:.3f} ({exact_matches}/{len(predictions)})")
        
        return level_accuracies, path_accuracy

# Example: News Article Hierarchical Classification
# Hierarchy: Root → Main Category → Sub Category → Specific Topic

hierarchy_structure = {
    'news': {
        'technology': {
            'artificial_intelligence': ['machine_learning', 'deep_learning', 'nlp'],
            'mobile_technology': ['smartphones', 'apps', 'tablets'],
            'computing': ['hardware', 'software', 'cloud']
        },
        'sports': {
            'football': ['nfl', 'college', 'international'],
            'basketball': ['nba', 'college', 'international'],
            'olympics': ['summer', 'winter', 'paralympics']
        },
        'business': {
            'finance': ['markets', 'banking', 'crypto'],
            'corporate': ['mergers', 'earnings', 'leadership'],
            'economics': ['policy', 'indicators', 'trade']
        }
    }
}

# Sample hierarchical data
hierarchical_texts = [
    "New AI algorithm achieves breakthrough in natural language understanding",
    "Stock market reaches record high as tech companies report strong earnings",
    "Olympic swimmer breaks world record in 200-meter freestyle",
    "Apple releases new iPhone with advanced machine learning capabilities",
    "Federal Reserve announces interest rate policy changes affecting markets",
    "NBA championship series breaks television viewership records",
    "Cryptocurrency market experiences volatility amid regulatory concerns",
    "Google's latest AI model shows remarkable progress in language processing",
    "Professional football season kicks off with new safety protocols",
    "Banking sector stocks surge following positive economic indicators"
]

hierarchical_labels = [
    ['technology', 'artificial_intelligence', 'nlp'],
    ['business', 'finance', 'markets'],
    ['sports', 'olympics', 'summer'],
    ['technology', 'mobile_technology', 'smartphones'],
    ['business', 'economics', 'policy'],
    ['sports', 'basketball', 'nba'],
    ['business', 'finance', 'crypto'],
    ['technology', 'artificial_intelligence', 'machine_learning'],
    ['sports', 'football', 'nfl'],
    ['business', 'finance', 'banking']
]

# Create hierarchical classifier
hier_classifier = HierarchicalTextClassifier()

# Define hierarchy (simplified for this example)
simple_hierarchy = {
    'root': ['technology', 'sports', 'business']
}
hier_classifier.define_hierarchy(simple_hierarchy)

# Prepare data
X, level_labels = hier_classifier.prepare_hierarchical_data(hierarchical_texts, hierarchical_labels)

# Train classifiers
hier_classifier.train_hierarchical_classifiers(X, level_labels)

# Test predictions
test_hierarchical_texts = [
    "Revolutionary neural network achieves human-level performance in language tasks",
    "World Cup final attracts billions of viewers worldwide",
    "Tech startup raises $100 million in Series B funding round"
]

test_hierarchical_labels = [
    ['technology', 'artificial_intelligence', 'deep_learning'],
    ['sports', 'football', 'international'],
    ['business', 'corporate', 'earnings']
]

# Make predictions
hier_predictions = hier_classifier.predict_hierarchical(test_hierarchical_texts, 'top_down')

print("Hierarchical Classification Results:")
print("=" * 50)
for i, pred in enumerate(hier_predictions):
    print(f"Text: '{pred['text']}'")
    print("Predicted path:")
    for step in pred['predicted_path']:
        print(f"  Level {step['level']}: {step['category']} (confidence: {step['confidence']:.3f})")
    print(f"True path: {test_hierarchical_labels[i]}")
    print()

# Evaluate
level_acc, path_acc = hier_classifier.evaluate_hierarchical(
    test_hierarchical_texts, 
    test_hierarchical_labels, 
    'top_down'
)
```

## 💡 Production-Ready Classification System

### Building a Complete Classification Pipeline

```python
class ProductionTextClassifier:
    """Production-ready text classification system"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.preprocessing_pipeline = None
        self.postprocessing_pipeline = None
        
    def create_preprocessing_pipeline(self):
        """Create comprehensive preprocessing pipeline"""
        
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        
        def advanced_text_preprocessing(texts):
            """Advanced text preprocessing function"""
            processed = []
            
            for text in texts:
                # Handle missing values
                if pd.isna(text) or text is None:
                    text = ""
                
                text = str(text)
                
                # Language detection (simplified)
                # In production, use langdetect library
                
                # Normalize text
                text = text.lower()
                
                # Remove noise
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                text = re.sub(r'@\w+|#\w+', '', text)
                text = re.sub(r'\d+', 'NUMBER', text)
                
                # Handle contractions
                contractions = {
                    "won't": "will not", "can't": "cannot", "n't": " not",
                    "'re": " are", "'ve": " have", "'ll": " will",
                    "'d": " would", "'m": " am"
                }
                
                for contraction, expansion in contractions.items():
                    text = text.replace(contraction, expansion)
                
                processed.append(text)
            
            return processed
        
        self.preprocessing_pipeline = Pipeline([
            ('text_preprocessing', FunctionTransformer(advanced_text_preprocessing)),
            ('vectorizer', TfidfVectorizer(
                max_features=self.config.get('max_features', 10000),
                ngram_range=self.config.get('ngram_range', (1, 2)),
                stop_words='english',
                min_df=self.config.get('min_df', 2),
                max_df=self.config.get('max_df', 0.95)
            ))
        ])
        
        return self.preprocessing_pipeline
    
    def train_ensemble_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble of multiple models"""
        
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        
        # Individual models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        # Train individual models
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            if X_val is not None and y_val is not None:
                val_score = model.score(X_val, y_val)
                print(f"{name} validation score: {val_score:.3f}")
        
        # Create ensemble
        ensemble = VotingClassifier([
            ('lr', models['logistic_regression']),
            ('rf', models['random_forest']),
            ('svm', models['svm']),
            ('nb', models['naive_bayes'])
        ], voting='soft')
        
        print("Training ensemble...")
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        
        if X_val is not None and y_val is not None:
            ensemble_score = ensemble.score(X_val, y_val)
            print(f"Ensemble validation score: {ensemble_score:.3f}")
        
        return ensemble
    
    def predict_with_confidence(self, texts, model_name='ensemble', confidence_threshold=0.5):
        """Make predictions with confidence estimation"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        
        # Preprocess texts
        X = self.preprocessing_pipeline.transform(texts)
        
        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            confidence = max(probs)
            
            # Determine if prediction is reliable
            is_reliable = confidence >= confidence_threshold
            
            # Get class probabilities
            class_probs = dict(zip(model.classes_, probs))
            
            results.append({
                'text': text,
                'predicted_class': pred,
                'confidence': confidence,
                'is_reliable': is_reliable,
                'class_probabilities': class_probs,
                'model_used': model_name
            })
        
        return results
    
    def batch_predict(self, texts, batch_size=1000):
        """Handle large-scale batch prediction"""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.predict_with_confidence(batch)
            results.extend(batch_results)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return results
    
    def monitor_model_drift(self, new_texts, new_labels=None):
        """Monitor for model drift in production"""
        
        # Feature drift detection
        X_new = self.preprocessing_pipeline.transform(new_texts)
        
        # Simple approach: compare feature distributions
        if hasattr(self, 'training_feature_stats'):
            # Compare mean and std of features
            new_mean = np.mean(X_new.toarray(), axis=0)
            new_std = np.std(X_new.toarray(), axis=0)
            
            drift_score = np.mean(np.abs(new_mean - self.training_feature_stats['mean']))
            
            print(f"Feature drift score: {drift_score:.3f}")
            
            if drift_score > 0.1:  # Threshold
                print("⚠️ Potential feature drift detected!")
        
        # Performance drift (if labels available)
        if new_labels is not None:
            predictions = self.predict_with_confidence(new_texts)
            predicted_labels = [pred['predicted_class'] for pred in predictions]
            
            accuracy = accuracy_score(new_labels, predicted_labels)
            print(f"Current performance: {accuracy:.3f}")
            
            if hasattr(self, 'baseline_performance'):
                performance_drop = self.baseline_performance - accuracy
                if performance_drop > 0.05:  # 5% drop threshold
                    print("⚠️ Performance drift detected!")

# Configuration for production system
production_config = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'confidence_threshold': 0.7
}

# Example usage
print("Setting up production classification system...")
prod_classifier = ProductionTextClassifier(production_config)
prod_classifier.create_preprocessing_pipeline()

# This would be used with real training data in production
print("Production system ready for training and deployment!")
```

## 💡 Key Takeaways

1. **Transformers are the new standard** - But require computational resources
2. **Multi-label classification is common** - Real texts often belong to multiple categories
3. **Few-shot learning enables rapid deployment** - New categories without extensive retraining
4. **Hierarchical classification matches real-world structure** - Categories often have natural hierarchies
5. **Production systems need monitoring** - Detect and handle model drift
6. **Ensemble methods improve robustness** - Combine multiple approaches for better results

## 🚀 What's Next?

You've mastered advanced classification techniques! Next, explore [Word Embeddings and Semantic Similarity](../03_word_embeddings/01_word2vec_fundamentals.md) to understand how machines learn the meaning of words.

**Coming up:**

- Word2Vec: Learning word relationships
- GloVe: Global vector representations  
- BERT: Contextual embeddings
- Building semantic search systems

Ready to explore the mathematical representation of meaning? Let's continue!
