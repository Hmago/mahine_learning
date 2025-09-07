# Document Embeddings and Sentence Representations

## 🎯 What You'll Learn

Moving beyond individual words, you'll learn how to create meaningful representations for entire sentences, paragraphs, and documents. This opens up possibilities for document similarity, clustering, and retrieval systems.

## 🧠 The Challenge: From Words to Documents

Word embeddings work great for individual words, but what about longer text? How do you represent the meaning of an entire sentence or document?

**Naive approaches:**
- **Average word vectors**: Simple but loses word order
- **Weighted average**: Better but still ignores structure
- **Concatenation**: Preserves order but creates huge vectors

**Better approaches:**
- **Doc2Vec**: Learn document representations directly
- **Sentence transformers**: Modern neural approaches
- **Universal Sentence Encoder**: Google's approach

## 📄 Doc2Vec: Paragraph Vector Algorithm

**The key insight:** Just like Word2Vec learns word representations, Doc2Vec learns document representations by predicting words in a document context.

**Think of it as:** Each document gets its own "ID" that helps predict the words it contains. The model learns what makes each document unique.

```python
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
import random

class DocumentEmbeddingSystem:
    """Comprehensive system for document embeddings"""
    
    def __init__(self):
        self.doc2vec_model = None
        self.tfidf_vectorizer = None
        self.documents = []
        self.document_tags = []
        self.embeddings = {}
        
    def prepare_documents(self, documents, tags=None):
        """Prepare documents for training"""
        
        self.documents = documents
        
        # Generate tags if not provided
        if tags is None:
            self.document_tags = [f"DOC_{i}" for i in range(len(documents))]
        else:
            self.document_tags = tags
            
        print(f"Prepared {len(documents)} documents for training")
        
    def create_tagged_documents(self):
        """Create TaggedDocument objects for Doc2Vec"""
        
        tagged_docs = []
        for i, doc in enumerate(self.documents):
            # Tokenize document
            words = doc.lower().split()
            # Create tagged document
            tagged_doc = TaggedDocument(words=words, tags=[self.document_tags[i]])
            tagged_docs.append(tagged_doc)
            
        return tagged_docs
    
    def train_doc2vec(self, vector_size=100, window=5, min_count=1, epochs=100):
        """Train Doc2Vec model"""
        
        print("Training Doc2Vec model...")
        
        # Prepare tagged documents
        tagged_docs = self.create_tagged_documents()
        
        # Train Doc2Vec model
        self.doc2vec_model = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            dm=1,  # Distributed Memory (PV-DM)
            workers=4
        )
        
        print(f"Doc2Vec training completed!")
        print(f"Vocabulary size: {len(self.doc2vec_model.wv.key_to_index)}")
        
        # Store embeddings
        self.embeddings['doc2vec'] = {}
        for tag in self.document_tags:
            self.embeddings['doc2vec'][tag] = self.doc2vec_model.dv[tag]
            
        return self.doc2vec_model
    
    def train_tfidf_baseline(self):
        """Train TF-IDF baseline for comparison"""
        
        print("Training TF-IDF baseline...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        # Store embeddings
        self.embeddings['tfidf'] = {}
        for i, tag in enumerate(self.document_tags):
            self.embeddings['tfidf'][tag] = tfidf_matrix[i].toarray().flatten()
            
        print("TF-IDF baseline training completed!")
        return self.tfidf_vectorizer
    
    def average_word_embeddings(self, word_embeddings):
        """Create document embeddings by averaging word embeddings"""
        
        print("Creating averaged word embeddings...")
        
        self.embeddings['averaged'] = {}
        
        for i, doc in enumerate(self.documents):
            words = doc.lower().split()
            word_vectors = []
            
            for word in words:
                if word in word_embeddings:
                    word_vectors.append(word_embeddings[word])
            
            if word_vectors:
                # Average the word vectors
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Random vector if no words found
                doc_vector = np.random.normal(0, 0.1, 100)
                
            self.embeddings['averaged'][self.document_tags[i]] = doc_vector
            
        print("Averaged embeddings created!")
    
    def weighted_word_embeddings(self, word_embeddings, tfidf_weights=None):
        """Create TF-IDF weighted document embeddings"""
        
        print("Creating weighted word embeddings...")
        
        # Use TF-IDF weights if available
        if tfidf_weights is None and self.tfidf_vectorizer is not None:
            tfidf_matrix = self.tfidf_vectorizer.transform(self.documents)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        self.embeddings['weighted'] = {}
        
        for i, doc in enumerate(self.documents):
            words = doc.lower().split()
            weighted_vectors = []
            total_weight = 0
            
            for word in words:
                if word in word_embeddings:
                    # Get TF-IDF weight if available
                    if self.tfidf_vectorizer is not None:
                        try:
                            word_idx = list(feature_names).index(word)
                            weight = tfidf_matrix[i, word_idx]
                        except ValueError:
                            weight = 1.0
                    else:
                        weight = 1.0
                    
                    weighted_vectors.append(weight * word_embeddings[word])
                    total_weight += weight
            
            if weighted_vectors and total_weight > 0:
                # Weighted average
                doc_vector = np.sum(weighted_vectors, axis=0) / total_weight
            else:
                # Random vector if no words found
                doc_vector = np.random.normal(0, 0.1, 100)
                
            self.embeddings['weighted'][self.document_tags[i]] = doc_vector
            
        print("Weighted embeddings created!")
    
    def compare_methods(self, query_doc, top_k=5):
        """Compare different embedding methods on a query"""
        
        print(f"\nFinding similar documents to: '{query_doc[:50]}...'")
        
        # Create query embeddings
        query_embeddings = {}
        
        # Doc2Vec query
        if self.doc2vec_model:
            query_words = query_doc.lower().split()
            query_embeddings['doc2vec'] = self.doc2vec_model.infer_vector(query_words)
        
        # TF-IDF query
        if self.tfidf_vectorizer:
            query_tfidf = self.tfidf_vectorizer.transform([query_doc])
            query_embeddings['tfidf'] = query_tfidf.toarray().flatten()
        
        # Averaged and weighted queries would need word embeddings
        # (Assuming they're available from training)
        
        results = {}
        
        for method, query_emb in query_embeddings.items():
            similarities = []
            
            for tag, doc_emb in self.embeddings[method].items():
                # Calculate cosine similarity
                sim = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                similarities.append((tag, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            results[method] = similarities[:top_k]
            
            print(f"\n{method.upper()} Results:")
            for i, (tag, sim) in enumerate(similarities[:top_k], 1):
                doc_idx = self.document_tags.index(tag)
                doc_preview = self.documents[doc_idx][:50] + "..."
                print(f"  {i}. {tag}: {sim:.3f} - {doc_preview}")
        
        return results
    
    def visualize_embeddings(self, method='doc2vec', figsize=(12, 8)):
        """Visualize document embeddings using t-SNE"""
        
        if method not in self.embeddings:
            print(f"Method '{method}' not available")
            return
        
        print(f"Visualizing {method} embeddings...")
        
        # Get embeddings and labels
        embeddings_matrix = []
        labels = []
        
        for tag, embedding in self.embeddings[method].items():
            embeddings_matrix.append(embedding)
            labels.append(tag)
        
        embeddings_matrix = np.array(embeddings_matrix)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
        embeddings_2d = tsne.fit_transform(embeddings_matrix)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Color by document index for variety
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, c=[colors[i]], s=100, alpha=0.7)
            plt.annotate(labels[i], (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.title(f'{method.upper()} Document Embeddings (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate_clustering(self, true_categories=None):
        """Evaluate how well embeddings capture document categories"""
        
        if true_categories is None:
            print("No true categories provided for evaluation")
            return
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        
        print("\nClustering Evaluation:")
        print("=" * 40)
        
        n_clusters = len(set(true_categories))
        
        for method, embeddings_dict in self.embeddings.items():
            # Get embeddings matrix
            embeddings_matrix = np.array(list(embeddings_dict.values()))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            predicted_clusters = kmeans.fit_predict(embeddings_matrix)
            
            # Calculate metrics
            ari_score = adjusted_rand_score(true_categories, predicted_clusters)
            silhouette = silhouette_score(embeddings_matrix, predicted_clusters)
            
            print(f"\n{method.upper()}:")
            print(f"  Adjusted Rand Index: {ari_score:.3f}")
            print(f"  Silhouette Score: {silhouette:.3f}")

# Create sample document corpus
def create_document_corpus():
    """Create a diverse document corpus for testing"""
    
    documents = [
        # Technology documents
        "Machine learning algorithms help computers learn from data without explicit programming.",
        "Artificial intelligence systems can process natural language and understand human speech.",
        "Deep learning neural networks have revolutionized computer vision and image recognition.",
        "Cloud computing provides scalable infrastructure for modern software applications.",
        
        # Science documents  
        "Quantum physics explores the behavior of matter and energy at atomic scales.",
        "Climate change affects global weather patterns and ocean temperatures worldwide.",
        "Genetic engineering techniques allow scientists to modify DNA sequences precisely.",
        "Space exploration missions have discovered thousands of exoplanets in distant galaxies.",
        
        # Health documents
        "Regular exercise and balanced nutrition promote cardiovascular health and longevity.",
        "Vaccination programs have successfully eliminated many infectious diseases globally.",
        "Mental health awareness campaigns reduce stigma and encourage treatment seeking.",
        "Medical research advances have led to breakthrough treatments for cancer patients.",
        
        # Business documents
        "E-commerce platforms have transformed retail shopping and consumer purchasing habits.",
        "Financial markets respond to economic indicators and geopolitical events rapidly.",
        "Supply chain management requires coordination across multiple vendors and regions.",
        "Digital marketing strategies leverage social media to reach target audiences effectively."
    ]
    
    categories = [
        'tech', 'tech', 'tech', 'tech',
        'science', 'science', 'science', 'science', 
        'health', 'health', 'health', 'health',
        'business', 'business', 'business', 'business'
    ]
    
    return documents, categories

# Demonstrate document embeddings
print("=" * 60)
print("DOCUMENT EMBEDDINGS DEMONSTRATION")
print("=" * 60)

# Create corpus
docs, categories = create_document_corpus()

# Initialize system
doc_system = DocumentEmbeddingSystem()
doc_system.prepare_documents(docs)

# Train different methods
doc_system.train_doc2vec(vector_size=100, epochs=50)
doc_system.train_tfidf_baseline()

# Create word embeddings for comparison (simplified)
from gensim.models import Word2Vec

# Train simple Word2Vec for averaging
sentences = [doc.split() for doc in docs]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, epochs=20)
word_embeddings = {word: w2v_model.wv[word] for word in w2v_model.wv.key_to_index}

# Create averaged embeddings
doc_system.average_word_embeddings(word_embeddings)
doc_system.weighted_word_embeddings(word_embeddings)

# Test with query
query = "artificial intelligence algorithms process natural language understanding"
results = doc_system.compare_methods(query, top_k=3)

# Visualize embeddings
doc_system.visualize_embeddings('doc2vec')

# Evaluate clustering
doc_system.evaluate_clustering(categories)
```

## 🤖 Modern Approach: Sentence Transformers

Sentence transformers use transformer architectures (like BERT) to create high-quality sentence embeddings:

```python
# Note: This requires sentence-transformers library
# pip install sentence-transformers

class ModernSentenceEmbeddings:
    """Modern sentence embeddings using transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Load pre-trained sentence transformer model"""
        
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
            
        except ImportError:
            print("sentence-transformers not installed. Using fallback...")
            self.create_simple_transformer_embedding()
            
    def create_simple_transformer_embedding(self):
        """Simple transformer-like embedding as fallback"""
        
        print("Creating simple transformer-like embeddings...")
        
        # This is a simplified version for demonstration
        # In practice, use the actual sentence-transformers library
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        self.simple_model = {
            'vectorizer': TfidfVectorizer(max_features=1000, stop_words='english'),
            'svd': TruncatedSVD(n_components=384)  # Approximate sentence-BERT size
        }
        
    def encode_sentences(self, sentences):
        """Encode sentences to embeddings"""
        
        if self.model is not None:
            # Use actual sentence transformer
            embeddings = self.model.encode(sentences)
            
        else:
            # Use fallback method
            tfidf_matrix = self.simple_model['vectorizer'].fit_transform(sentences)
            embeddings = self.simple_model['svd'].fit_transform(tfidf_matrix)
        
        # Cache embeddings
        for i, sentence in enumerate(sentences):
            self.embeddings_cache[sentence] = embeddings[i]
            
        return embeddings
    
    def semantic_search(self, query, corpus, top_k=5):
        """Perform semantic search on corpus"""
        
        print(f"Searching for: '{query}'")
        
        # Encode query and corpus
        all_sentences = [query] + corpus
        embeddings = self.encode_sentences(all_sentences)
        
        query_embedding = embeddings[0]
        corpus_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(corpus_embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity, corpus[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} results:")
        for rank, (idx, sim, doc) in enumerate(similarities[:top_k], 1):
            print(f"{rank}. ({sim:.3f}) {doc[:80]}...")
            
        return similarities[:top_k]
    
    def sentence_similarity_matrix(self, sentences):
        """Create similarity matrix between sentences"""
        
        embeddings = self.encode_sentences(sentences)
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarity_matrix[i, j] = sim
                else:
                    similarity_matrix[i, j] = 1.0
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, sentences, figsize=(10, 8)):
        """Visualize sentence similarity matrix"""
        
        similarity_matrix = self.sentence_similarity_matrix(sentences)
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            xticklabels=[f"S{i+1}" for i in range(len(sentences))],
            yticklabels=[f"S{i+1}" for i in range(len(sentences))]
        )
        
        plt.title('Sentence Similarity Matrix')
        plt.tight_layout()
        plt.show()
        
        # Print sentence mapping
        print("\nSentence mapping:")
        for i, sentence in enumerate(sentences):
            print(f"S{i+1}: {sentence[:60]}...")

# Demonstrate modern sentence embeddings
print("\n" + "=" * 60)
print("MODERN SENTENCE EMBEDDINGS")
print("=" * 60)

# Initialize modern system
modern_system = ModernSentenceEmbeddings()
modern_system.load_model()

# Test sentences
test_sentences = [
    "Machine learning helps computers learn from data.",
    "AI systems can understand human language naturally.", 
    "Dogs are loyal pets that love their owners.",
    "Cats are independent animals that enjoy solitude.",
    "The weather today is sunny and warm.",
    "Climate change affects global temperatures."
]

# Semantic search
search_query = "pets and animals"
modern_system.semantic_search(search_query, test_sentences, top_k=3)

# Visualize similarities
modern_system.visualize_similarity_matrix(test_sentences)
```

## 🔄 Universal Sentence Encoder (USE)

Google's Universal Sentence Encoder provides high-quality, general-purpose sentence embeddings:

```python
class UniversalSentenceEncoder:
    """Wrapper for Universal Sentence Encoder"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """Load Universal Sentence Encoder"""
        
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            print("Loading Universal Sentence Encoder...")
            
            # USE model URL
            model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
            self.model = hub.load(model_url)
            self.model_loaded = True
            
            print("USE model loaded successfully!")
            
        except ImportError:
            print("TensorFlow Hub not available. Creating simulation...")
            self.create_use_simulation()
    
    def create_use_simulation(self):
        """Create USE-like embeddings as simulation"""
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        print("Creating USE simulation...")
        
        self.use_simulation = {
            'vectorizer': TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),
                stop_words='english'
            ),
            'svd': TruncatedSVD(n_components=512)  # USE embedding size
        }
        
        self.model_loaded = True
    
    def encode(self, texts):
        """Encode texts to embeddings"""
        
        if not self.model_loaded:
            self.load_model()
        
        if hasattr(self, 'use_simulation'):
            # Use simulation
            tfidf_matrix = self.use_simulation['vectorizer'].fit_transform(texts)
            embeddings = self.use_simulation['svd'].fit_transform(tfidf_matrix)
            
        else:
            # Use real USE model
            embeddings = self.model(texts).numpy()
        
        return embeddings
    
    def similarity_search(self, query, documents, threshold=0.5):
        """Find similar documents above threshold"""
        
        # Encode all texts
        all_texts = [query] + documents
        embeddings = self.encode(all_texts)
        
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]
        
        # Find similarities
        results = []
        for i, doc_emb in enumerate(doc_embs):
            similarity = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            
            if similarity >= threshold:
                results.append((i, similarity, documents[i]))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(results)} documents above threshold {threshold}")
        for i, (idx, sim, doc) in enumerate(results):
            print(f"{i+1}. ({sim:.3f}) {doc[:70]}...")
            
        return results

# Test Universal Sentence Encoder
print("\n" + "=" * 60)
print("UNIVERSAL SENTENCE ENCODER")
print("=" * 60)

use_encoder = UniversalSentenceEncoder()

# Test documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models require large amounts of training data",
    "Climate change is affecting weather patterns around the world",
    "Artificial intelligence can help solve complex problems",
    "Deep learning algorithms are inspired by neural networks in the brain"
]

# Search query
query = "AI and machine learning technologies"

# Perform similarity search
results = use_encoder.similarity_search(query, documents, threshold=0.3)
```

## 📊 Comprehensive Evaluation Framework

Let's build a framework to systematically evaluate different document embedding approaches:

```python
class DocumentEmbeddingEvaluator:
    """Comprehensive evaluation of document embedding methods"""
    
    def __init__(self):
        self.methods = {}
        self.datasets = {}
        self.results = {}
        
    def add_method(self, name, embedding_function):
        """Add an embedding method to evaluate"""
        self.methods[name] = embedding_function
        
    def add_dataset(self, name, documents, labels, queries=None):
        """Add a dataset for evaluation"""
        self.datasets[name] = {
            'documents': documents,
            'labels': labels,
            'queries': queries or []
        }
    
    def evaluate_clustering(self, dataset_name):
        """Evaluate clustering performance"""
        
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found")
            return
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        dataset = self.datasets[dataset_name]
        documents = dataset['documents']
        true_labels = dataset['labels']
        n_clusters = len(set(true_labels))
        
        print(f"\nClustering Evaluation on {dataset_name}")
        print("=" * 50)
        
        clustering_results = {}
        
        for method_name, embed_func in self.methods.items():
            print(f"\nEvaluating {method_name}...")
            
            try:
                # Get embeddings
                embeddings = embed_func(documents)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                predicted_labels = kmeans.fit_predict(embeddings)
                
                # Calculate metrics
                ari = adjusted_rand_score(true_labels, predicted_labels)
                nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                
                clustering_results[method_name] = {
                    'ARI': ari,
                    'NMI': nmi
                }
                
                print(f"  ARI: {ari:.3f}")
                print(f"  NMI: {nmi:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                clustering_results[method_name] = {'ARI': 0, 'NMI': 0}
        
        return clustering_results
    
    def evaluate_retrieval(self, dataset_name):
        """Evaluate document retrieval performance"""
        
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found")
            return
        
        dataset = self.datasets[dataset_name]
        documents = dataset['documents']
        queries = dataset['queries']
        
        if not queries:
            print("No queries provided for retrieval evaluation")
            return
        
        print(f"\nRetrieval Evaluation on {dataset_name}")
        print("=" * 50)
        
        retrieval_results = {}
        
        for method_name, embed_func in self.methods.items():
            print(f"\nEvaluating {method_name}...")
            
            try:
                # Get document embeddings
                doc_embeddings = embed_func(documents)
                
                method_scores = []
                
                for query_info in queries:
                    query_text = query_info['text']
                    relevant_docs = query_info['relevant']
                    
                    # Get query embedding
                    query_embedding = embed_func([query_text])[0]
                    
                    # Calculate similarities
                    similarities = []
                    for i, doc_emb in enumerate(doc_embeddings):
                        sim = np.dot(query_embedding, doc_emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                        )
                        similarities.append((i, sim))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Calculate precision@k
                    top_k = min(5, len(similarities))
                    retrieved = [idx for idx, _ in similarities[:top_k]]
                    
                    precision = len(set(retrieved) & set(relevant_docs)) / top_k
                    method_scores.append(precision)
                
                avg_precision = np.mean(method_scores)
                retrieval_results[method_name] = {'Precision@5': avg_precision}
                
                print(f"  Average Precision@5: {avg_precision:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                retrieval_results[method_name] = {'Precision@5': 0}
        
        return retrieval_results
    
    def evaluate_semantic_similarity(self, sentence_pairs):
        """Evaluate semantic similarity performance"""
        
        print(f"\nSemantic Similarity Evaluation")
        print("=" * 50)
        
        similarity_results = {}
        
        for method_name, embed_func in self.methods.items():
            print(f"\nEvaluating {method_name}...")
            
            try:
                correlations = []
                
                for pair_info in sentence_pairs:
                    sent1, sent2, human_score = pair_info
                    
                    # Get embeddings
                    embeddings = embed_func([sent1, sent2])
                    emb1, emb2 = embeddings[0], embeddings[1]
                    
                    # Calculate similarity
                    model_sim = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    
                    correlations.append((model_sim, human_score))
                
                # Calculate correlation with human judgments
                model_scores = [sim for sim, _ in correlations]
                human_scores = [score for _, score in correlations]
                
                correlation = np.corrcoef(model_scores, human_scores)[0, 1]
                similarity_results[method_name] = {'Correlation': correlation}
                
                print(f"  Correlation with human scores: {correlation:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                similarity_results[method_name] = {'Correlation': 0}
        
        return similarity_results
    
    def create_summary_report(self):
        """Create a comprehensive evaluation report"""
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 80)
        
        # Gather all results
        all_results = {}
        
        for method_name in self.methods.keys():
            all_results[method_name] = {}
        
        # Add clustering results if available
        if hasattr(self, 'clustering_results'):
            for method, scores in self.clustering_results.items():
                all_results[method].update(scores)
        
        # Add retrieval results if available  
        if hasattr(self, 'retrieval_results'):
            for method, scores in self.retrieval_results.items():
                all_results[method].update(scores)
        
        # Create summary table
        if all_results:
            import pandas as pd
            
            df = pd.DataFrame(all_results).T
            print("\nMethod Comparison:")
            print(df.round(3))
            
            # Find best methods
            print("\nBest Methods by Metric:")
            for metric in df.columns:
                best_method = df[metric].idxmax()
                best_score = df[metric].max()
                print(f"  {metric}: {best_method} ({best_score:.3f})")

# Example evaluation
print("\n" + "=" * 60)
print("COMPREHENSIVE EVALUATION EXAMPLE")
print("=" * 60)

# Create evaluator
evaluator = DocumentEmbeddingEvaluator()

# Add some simple embedding methods for demonstration
def simple_tfidf_embeddings(documents):
    """Simple TF-IDF embeddings"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    return vectorizer.fit_transform(documents).toarray()

def simple_averaged_embeddings(documents):
    """Simple averaged word embeddings"""
    # This is a simplified version
    embeddings = []
    for doc in documents:
        # Random embedding as placeholder
        embedding = np.random.normal(0, 1, 100)
        embeddings.append(embedding)
    return np.array(embeddings)

# Add methods
evaluator.add_method('TF-IDF', simple_tfidf_embeddings)
evaluator.add_method('Averaged', simple_averaged_embeddings)

# Create test dataset
test_docs, test_labels = create_document_corpus()

# Add dataset
evaluator.add_dataset('test_corpus', test_docs, test_labels)

# Run evaluation
clustering_results = evaluator.evaluate_clustering('test_corpus')

# Store results for summary
evaluator.clustering_results = clustering_results

# Create summary
evaluator.create_summary_report()
```

## 🏋️ Practice Exercise

**Build a Smart Document Search System**

Create a document search system that combines multiple embedding approaches:

```python
def build_smart_document_search():
    """
    Build a comprehensive document search system
    
    Requirements:
    1. Implement multiple embedding methods (Doc2Vec, averaged embeddings, TF-IDF)
    2. Create a query interface that handles different types of queries
    3. Implement result fusion from multiple methods
    4. Add relevance feedback to improve results
    5. Create evaluation metrics for search quality
    6. Handle both short queries and long document queries
    
    Bonus:
    - Real-time indexing for new documents
    - Multilingual support
    - Semantic clustering of results
    - Query expansion using synonyms
    """
    
    # Your implementation here
    pass

# Test your system with different document types
test_scenarios = [
    {
        'name': 'Scientific Papers',
        'query_types': ['technical terms', 'abstract concepts', 'methodology'],
        'challenges': ['domain vocabulary', 'complex language', 'long documents']
    },
    {
        'name': 'News Articles', 
        'query_types': ['events', 'people', 'locations'],
        'challenges': ['time-sensitive', 'proper nouns', 'evolving vocabulary']
    },
    {
        'name': 'Legal Documents',
        'query_types': ['legal concepts', 'case citations', 'statutes'],
        'challenges': ['formal language', 'precise terminology', 'long references']
    }
]
```

## 💡 Key Takeaways

1. **Doc2Vec learns document-specific representations** - Each document gets a unique vector
2. **Averaging word embeddings is simple but effective** - Good baseline approach
3. **TF-IDF weighting improves simple averaging** - Emphasizes important words
4. **Modern transformers excel at semantic understanding** - BERT-based approaches are powerful
5. **Different methods excel in different scenarios** - Choose based on your specific needs
6. **Evaluation should be task-specific** - Clustering, retrieval, and similarity require different metrics

## 🚀 What's Next?

You've mastered document-level representations! Next, explore [Contextual Embeddings and BERT](./05_contextual_embeddings.md) to learn how modern transformer models create context-aware word representations.

**Coming up:**

- ELMo: Context-aware embeddings
- BERT: Bidirectional transformers  
- Fine-tuning embeddings for specific tasks
- Handling polysemy and context sensitivity

Ready to dive into the transformer revolution? Let's continue!
