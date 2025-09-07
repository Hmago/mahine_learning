# Word Embedding Applications

## 🎯 What You'll Learn

Now that you understand word embeddings, let's build real-world applications! You'll create semantic search engines, recommendation systems, document clustering tools, and learn to detect and mitigate bias in embeddings.

## 🔍 Semantic Search Systems

Traditional search looks for exact word matches. Semantic search understands meaning and finds conceptually similar content even with different words.

**Think of it as:** Instead of searching for "car", semantic search also finds "automobile", "vehicle", "sedan" because it understands they're related concepts.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from collections import defaultdict
import time

class SemanticSearchEngine:
    """Advanced semantic search using word embeddings"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.document_embeddings = {}
        self.documents = []
        self.document_metadata = []
        self.tfidf_vectorizer = None
        self.hybrid_weights = {'semantic': 0.7, 'lexical': 0.3}
        
    def load_embeddings(self, embedding_path=None):
        """Load pre-trained word embeddings"""
        
        if embedding_path:
            print(f"Loading embeddings from {embedding_path}")
            # In practice, load from file
            # self.embedding_model = load_embeddings(embedding_path)
        else:
            print("Creating sample embeddings...")
            self.create_sample_embeddings()
    
    def create_sample_embeddings(self):
        """Create sample embeddings for demonstration"""
        
        # Sample vocabulary with related words
        vocab = {
            # Technology cluster
            'computer': np.array([0.8, 0.1, 0.0, 0.1, 0.0]),
            'laptop': np.array([0.7, 0.2, 0.0, 0.1, 0.0]),
            'software': np.array([0.6, 0.3, 0.0, 0.1, 0.0]),
            'technology': np.array([0.9, 0.1, 0.0, 0.0, 0.0]),
            'programming': np.array([0.5, 0.4, 0.0, 0.1, 0.0]),
            
            # Health cluster
            'health': np.array([0.0, 0.8, 0.1, 0.1, 0.0]),
            'medicine': np.array([0.1, 0.7, 0.2, 0.0, 0.0]),
            'doctor': np.array([0.0, 0.6, 0.3, 0.1, 0.0]),
            'hospital': np.array([0.0, 0.5, 0.4, 0.1, 0.0]),
            'treatment': np.array([0.0, 0.7, 0.2, 0.1, 0.0]),
            
            # Education cluster
            'education': np.array([0.0, 0.1, 0.8, 0.1, 0.0]),
            'school': np.array([0.0, 0.0, 0.7, 0.2, 0.1]),
            'student': np.array([0.0, 0.0, 0.6, 0.3, 0.1]),
            'learning': np.array([0.1, 0.0, 0.7, 0.2, 0.0]),
            'teacher': np.array([0.0, 0.0, 0.5, 0.4, 0.1]),
            
            # Business cluster
            'business': np.array([0.1, 0.0, 0.1, 0.8, 0.0]),
            'company': np.array([0.0, 0.0, 0.0, 0.9, 0.1]),
            'finance': np.array([0.0, 0.0, 0.0, 0.7, 0.3]),
            'market': np.array([0.0, 0.0, 0.0, 0.6, 0.4]),
            'investment': np.array([0.0, 0.0, 0.0, 0.5, 0.5]),
            
            # Sports cluster
            'sports': np.array([0.0, 0.1, 0.0, 0.1, 0.8]),
            'football': np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
            'player': np.array([0.0, 0.0, 0.1, 0.0, 0.8]),
            'game': np.array([0.1, 0.0, 0.0, 0.1, 0.7]),
            'team': np.array([0.0, 0.0, 0.0, 0.2, 0.8])
        }
        
        self.embedding_model = vocab
        print(f"Created embeddings for {len(vocab)} words")
    
    def get_word_embedding(self, word):
        """Get embedding for a single word"""
        
        word_lower = word.lower()
        if word_lower in self.embedding_model:
            return self.embedding_model[word_lower]
        else:
            # Return zero vector for unknown words
            return np.zeros(5)  # Assuming 5-dimensional embeddings
    
    def create_document_embedding(self, document):
        """Create document embedding by averaging word embeddings"""
        
        words = document.lower().split()
        word_embeddings = []
        
        for word in words:
            embedding = self.get_word_embedding(word)
            if np.any(embedding):  # Skip zero vectors
                word_embeddings.append(embedding)
        
        if word_embeddings:
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros(5)
    
    def index_documents(self, documents, metadata=None):
        """Index documents for semantic search"""
        
        print("Indexing documents...")
        self.documents = documents
        self.document_metadata = metadata or [{'id': i} for i in range(len(documents))]
        
        # Create semantic embeddings
        for i, doc in enumerate(documents):
            self.document_embeddings[i] = self.create_document_embedding(doc)
        
        # Create TF-IDF index for hybrid search
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        print(f"Indexed {len(documents)} documents")
    
    def semantic_search(self, query, top_k=5):
        """Perform semantic search"""
        
        # Create query embedding
        query_embedding = self.create_document_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def lexical_search(self, query, top_k=5):
        """Perform traditional TF-IDF search"""
        
        # Transform query
        query_tfidf = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [(idx, similarities[idx]) for idx in top_indices]
        return results
    
    def hybrid_search(self, query, top_k=5):
        """Combine semantic and lexical search"""
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=len(self.documents))
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}
        
        # Get lexical results  
        lexical_results = self.lexical_search(query, top_k=len(self.documents))
        lexical_scores = {doc_id: score for doc_id, score in lexical_results}
        
        # Combine scores
        combined_scores = []
        for doc_id in range(len(self.documents)):
            semantic_score = semantic_scores.get(doc_id, 0)
            lexical_score = lexical_scores.get(doc_id, 0)
            
            combined_score = (
                self.hybrid_weights['semantic'] * semantic_score +
                self.hybrid_weights['lexical'] * lexical_score
            )
            
            combined_scores.append((doc_id, combined_score))
        
        # Sort and return top results
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]
    
    def search_with_explanation(self, query, method='hybrid'):
        """Search with detailed explanation"""
        
        print(f"\nSearching for: '{query}'")
        print(f"Method: {method}")
        print("-" * 50)
        
        # Perform search
        if method == 'semantic':
            results = self.semantic_search(query)
        elif method == 'lexical':
            results = self.lexical_search(query)
        else:
            results = self.hybrid_search(query)
        
        # Display results with explanation
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = self.documents[doc_id]
            metadata = self.document_metadata[doc_id]
            
            print(f"{rank}. Score: {score:.3f}")
            print(f"   Document: {doc[:80]}...")
            
            if 'category' in metadata:
                print(f"   Category: {metadata['category']}")
            
            # Show word overlap for explanation
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = query_words & doc_words
            
            if overlap:
                print(f"   Word overlap: {overlap}")
            
            print()
    
    def compare_search_methods(self, query):
        """Compare different search methods side by side"""
        
        print(f"\nComparing search methods for: '{query}'")
        print("=" * 70)
        
        methods = ['semantic', 'lexical', 'hybrid']
        
        for method in methods:
            print(f"\n{method.upper()} SEARCH:")
            print("-" * 30)
            
            if method == 'semantic':
                results = self.semantic_search(query, top_k=3)
            elif method == 'lexical':
                results = self.lexical_search(query, top_k=3)
            else:
                results = self.hybrid_search(query, top_k=3)
            
            for rank, (doc_id, score) in enumerate(results, 1):
                doc = self.documents[doc_id]
                print(f"{rank}. ({score:.3f}) {doc[:60]}...")

# Create sample document corpus
def create_document_corpus():
    """Create a diverse document corpus for search testing"""
    
    documents = [
        # Technology documents
        "Latest advances in computer programming and software development",
        "New laptop models with improved performance for developers", 
        "Machine learning algorithms revolutionize technology industry",
        "Programming languages comparison for software engineering",
        "Computer science education and technology careers",
        
        # Health documents
        "Medical treatment advances in modern healthcare systems",
        "Doctor consultation and hospital patient care improvement",
        "Health insurance coverage for medical procedures",
        "Medicine research and pharmaceutical development",
        "Healthcare technology and digital health solutions",
        
        # Education documents
        "Student learning outcomes and education quality metrics",
        "Teacher training programs and school curriculum development",
        "Educational technology integration in modern classrooms",
        "Learning management systems for online education",
        "Student assessment and academic performance evaluation",
        
        # Business documents
        "Company financial performance and market analysis",
        "Business investment strategies and finance management",
        "Market trends and business development opportunities",
        "Corporate finance and investment portfolio management",
        "Business analytics and company performance metrics",
        
        # Sports documents
        "Football team performance and player statistics",
        "Sports game analysis and team strategy",
        "Player training and athletic performance improvement",
        "Sports marketing and team sponsorship deals",
        "Game highlights and sports entertainment"
    ]
    
    # Add metadata
    categories = ['tech'] * 5 + ['health'] * 5 + ['education'] * 5 + ['business'] * 5 + ['sports'] * 5
    metadata = [{'id': i, 'category': cat} for i, cat in enumerate(categories)]
    
    return documents, metadata

# Demonstrate semantic search
print("=" * 60)
print("SEMANTIC SEARCH ENGINE DEMONSTRATION")
print("=" * 60)

# Create search engine
search_engine = SemanticSearchEngine()
search_engine.load_embeddings()

# Create and index documents
docs, meta = create_document_corpus()
search_engine.index_documents(docs, meta)

# Test different types of queries
test_queries = [
    "artificial intelligence programming",  # Tech query
    "medical doctor patient care",          # Health query  
    "student classroom learning",           # Education query
    "company financial investment",         # Business query
    "football player team game"            # Sports query
]

# Test each query with different methods
for query in test_queries:
    search_engine.compare_search_methods(query)
    print("\n" + "="*70)
```

## 🎭 Recommendation Systems Using Embeddings

Embeddings can power sophisticated recommendation systems by understanding user preferences and item similarities:

```python
class EmbeddingRecommendationSystem:
    """Recommendation system using embedding-based similarities"""
    
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_item_interactions = {}
        self.item_features = {}
        
    def create_content_embeddings(self, items_data):
        """Create item embeddings from content features"""
        
        print("Creating content-based embeddings...")
        
        for item_id, features in items_data.items():
            # Combine text features
            text_content = ' '.join([
                features.get('title', ''),
                features.get('description', ''),
                features.get('genre', ''),
                features.get('category', '')
            ])
            
            # Simple embedding: TF-IDF + dimensionality reduction
            embedding = self.text_to_embedding(text_content)
            self.item_embeddings[item_id] = embedding
            self.item_features[item_id] = features
    
    def text_to_embedding(self, text):
        """Convert text to embedding (simplified)"""
        
        # In practice, use proper word embeddings
        # This is a simplified hash-based approach
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        
        for word in words:
            # Simple hash to embedding
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 1
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def learn_user_embeddings(self, user_interactions):
        """Learn user embeddings from interaction history"""
        
        print("Learning user embeddings from interactions...")
        
        for user_id, interactions in user_interactions.items():
            user_embedding = np.zeros(self.embedding_dim)
            total_weight = 0
            
            for item_id, rating in interactions.items():
                if item_id in self.item_embeddings:
                    # Weight by rating (higher rating = more influence)
                    weight = rating / 5.0  # Assuming 5-star scale
                    user_embedding += weight * self.item_embeddings[item_id]
                    total_weight += weight
            
            # Average the embeddings
            if total_weight > 0:
                user_embedding = user_embedding / total_weight
            
            self.user_embeddings[user_id] = user_embedding
            self.user_item_interactions[user_id] = interactions
    
    def recommend_items(self, user_id, num_recommendations=5, method='collaborative'):
        """Generate recommendations for a user"""
        
        if user_id not in self.user_embeddings:
            print(f"User {user_id} not found")
            return []
        
        user_embedding = self.user_embeddings[user_id]
        user_interactions = self.user_item_interactions.get(user_id, {})
        
        # Calculate similarities to all items
        recommendations = []
        
        for item_id, item_embedding in self.item_embeddings.items():
            # Skip items user has already interacted with
            if item_id in user_interactions:
                continue
            
            if method == 'collaborative':
                # User-item similarity
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    item_embedding.reshape(1, -1)
                )[0, 0]
            
            elif method == 'content':
                # Content-based similarity to user's preferred items
                similarity = self.content_based_similarity(user_id, item_id)
            
            else:  # hybrid
                collab_sim = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    item_embedding.reshape(1, -1)
                )[0, 0]
                content_sim = self.content_based_similarity(user_id, item_id)
                similarity = 0.7 * collab_sim + 0.3 * content_sim
            
            recommendations.append((item_id, similarity))
        
        # Sort by similarity and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def content_based_similarity(self, user_id, item_id):
        """Calculate content-based similarity"""
        
        user_interactions = self.user_item_interactions[user_id]
        target_item = self.item_embeddings[item_id]
        
        # Find similarity to user's highly-rated items
        similarities = []
        
        for interacted_item, rating in user_interactions.items():
            if rating >= 4 and interacted_item in self.item_embeddings:  # High rating threshold
                similarity = cosine_similarity(
                    target_item.reshape(1, -1),
                    self.item_embeddings[interacted_item].reshape(1, -1)
                )[0, 0]
                similarities.append(similarity * (rating / 5.0))
        
        return np.mean(similarities) if similarities else 0
    
    def explain_recommendation(self, user_id, item_id):
        """Explain why an item was recommended"""
        
        user_interactions = self.user_item_interactions[user_id]
        target_features = self.item_features[item_id]
        
        print(f"\nWhy we recommend '{target_features.get('title', item_id)}':")
        
        # Find similar items user liked
        similar_items = []
        target_embedding = self.item_embeddings[item_id]
        
        for interacted_item, rating in user_interactions.items():
            if rating >= 4 and interacted_item in self.item_embeddings:
                similarity = cosine_similarity(
                    target_embedding.reshape(1, -1),
                    self.item_embeddings[interacted_item].reshape(1, -1)
                )[0, 0]
                
                if similarity > 0.5:  # Threshold for similarity
                    similar_features = self.item_features[interacted_item]
                    similar_items.append((
                        similar_features.get('title', interacted_item),
                        similarity,
                        rating
                    ))
        
        if similar_items:
            print("  Based on your preferences for:")
            for title, sim, rating in sorted(similar_items, key=lambda x: x[1], reverse=True)[:3]:
                print(f"    • {title} (similarity: {sim:.2f}, your rating: {rating}/5)")
        
        # Show content similarities
        print(f"  Genre: {target_features.get('genre', 'Unknown')}")
        print(f"  Category: {target_features.get('category', 'Unknown')}")
    
    def evaluate_recommendations(self, test_interactions):
        """Evaluate recommendation quality"""
        
        print("\nEvaluating recommendation system...")
        
        total_precision = 0
        total_recall = 0
        num_users = 0
        
        for user_id, test_items in test_interactions.items():
            if user_id not in self.user_embeddings:
                continue
            
            # Get recommendations
            recommendations = self.recommend_items(user_id, num_recommendations=10)
            recommended_items = {item_id for item_id, _ in recommendations}
            
            # Get ground truth (items user actually liked)
            liked_items = {item_id for item_id, rating in test_items.items() if rating >= 4}
            
            # Calculate precision and recall
            if recommended_items and liked_items:
                intersection = recommended_items & liked_items
                precision = len(intersection) / len(recommended_items)
                recall = len(intersection) / len(liked_items)
                
                total_precision += precision
                total_recall += recall
                num_users += 1
        
        if num_users > 0:
            avg_precision = total_precision / num_users
            avg_recall = total_recall / num_users
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            print(f"Average Precision: {avg_precision:.3f}")
            print(f"Average Recall: {avg_recall:.3f}")
            print(f"F1 Score: {f1_score:.3f}")

# Create sample recommendation data
def create_recommendation_data():
    """Create sample data for recommendation system"""
    
    # Sample items (movies)
    items = {
        'movie_1': {
            'title': 'Action Hero Adventures',
            'description': 'Exciting action movie with explosions and fights',
            'genre': 'action',
            'category': 'entertainment'
        },
        'movie_2': {
            'title': 'Romantic Comedy Delight',
            'description': 'Funny romantic story with happy ending',
            'genre': 'comedy romance',
            'category': 'entertainment'
        },
        'movie_3': {
            'title': 'Sci-Fi Space Odyssey',
            'description': 'Future technology and space exploration adventure',
            'genre': 'science fiction',
            'category': 'entertainment'
        },
        'movie_4': {
            'title': 'Horror Night Terrors',
            'description': 'Scary movie with supernatural elements',
            'genre': 'horror',
            'category': 'entertainment'
        },
        'movie_5': {
            'title': 'Documentary Nature',
            'description': 'Educational film about wildlife and environment',
            'genre': 'documentary',
            'category': 'educational'
        },
        'movie_6': {
            'title': 'Action Thriller Chase',
            'description': 'Fast-paced action with car chases and suspense',
            'genre': 'action thriller',
            'category': 'entertainment'
        },
        'movie_7': {
            'title': 'Comedy Family Fun',
            'description': 'Family-friendly comedy for all ages',
            'genre': 'comedy family',
            'category': 'entertainment'
        },
        'movie_8': {
            'title': 'Space Adventure Epic',
            'description': 'Epic space battles and alien encounters',
            'genre': 'science fiction action',
            'category': 'entertainment'
        }
    }
    
    # Sample user interactions (user_id: {item_id: rating})
    user_interactions = {
        'user_1': {  # Likes action movies
            'movie_1': 5,
            'movie_6': 4,
            'movie_3': 3,
            'movie_2': 2
        },
        'user_2': {  # Likes comedy
            'movie_2': 5,
            'movie_7': 4,
            'movie_1': 2,
            'movie_4': 1
        },
        'user_3': {  # Likes sci-fi
            'movie_3': 5,
            'movie_8': 5,
            'movie_5': 4,
            'movie_1': 3
        },
        'user_4': {  # Diverse tastes
            'movie_1': 4,
            'movie_2': 4,
            'movie_3': 3,
            'movie_5': 5
        }
    }
    
    return items, user_interactions

# Demonstrate recommendation system
print("\n" + "=" * 60)
print("EMBEDDING-BASED RECOMMENDATION SYSTEM")
print("=" * 60)

# Create recommendation system
rec_system = EmbeddingRecommendationSystem(embedding_dim=20)

# Load data
items_data, user_data = create_recommendation_data()

# Train the system
rec_system.create_content_embeddings(items_data)
rec_system.learn_user_embeddings(user_data)

# Test recommendations for each user
for user_id in user_data.keys():
    print(f"\nRecommendations for {user_id}:")
    print("-" * 30)
    
    recommendations = rec_system.recommend_items(user_id, num_recommendations=3)
    
    for rank, (item_id, score) in enumerate(recommendations, 1):
        item_title = items_data[item_id]['title']
        genre = items_data[item_id]['genre']
        print(f"{rank}. {item_title} (Score: {score:.3f}, Genre: {genre})")
    
    # Explain first recommendation
    if recommendations:
        top_item = recommendations[0][0]
        rec_system.explain_recommendation(user_id, top_item)
```

## 📊 Document Clustering with Embeddings

Use embeddings to automatically group similar documents:

```python
class DocumentClusteringSystem:
    """Document clustering using embedding representations"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.documents = []
        self.document_embeddings = []
        self.clusters = {}
        self.cluster_centers = {}
        
    def create_sample_embeddings(self):
        """Create sample word embeddings for clustering"""
        
        # Technology words
        tech_words = ['computer', 'software', 'programming', 'technology', 'algorithm']
        # Science words
        science_words = ['research', 'experiment', 'hypothesis', 'data', 'analysis']
        # Business words
        business_words = ['company', 'profit', 'market', 'investment', 'strategy']
        # Health words
        health_words = ['medicine', 'doctor', 'patient', 'treatment', 'hospital']
        
        embeddings = {}
        
        # Create clustered embeddings
        for i, word in enumerate(tech_words):
            embeddings[word] = np.array([0.8, 0.1, 0.1, 0.0]) + np.random.normal(0, 0.05, 4)
        
        for i, word in enumerate(science_words):
            embeddings[word] = np.array([0.1, 0.8, 0.1, 0.0]) + np.random.normal(0, 0.05, 4)
        
        for i, word in enumerate(business_words):
            embeddings[word] = np.array([0.1, 0.1, 0.8, 0.0]) + np.random.normal(0, 0.05, 4)
        
        for i, word in enumerate(health_words):
            embeddings[word] = np.array([0.0, 0.1, 0.1, 0.8]) + np.random.normal(0, 0.05, 4)
        
        self.embedding_model = embeddings
    
    def document_to_embedding(self, document):
        """Convert document to embedding"""
        
        words = document.lower().split()
        word_vectors = []
        
        for word in words:
            if word in self.embedding_model:
                word_vectors.append(self.embedding_model[word])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(4)  # Assuming 4-dimensional embeddings
    
    def cluster_documents(self, documents, n_clusters=4, method='kmeans'):
        """Cluster documents using embeddings"""
        
        self.documents = documents
        print(f"Clustering {len(documents)} documents into {n_clusters} clusters...")
        
        # Create document embeddings
        self.document_embeddings = []
        for doc in documents:
            embedding = self.document_to_embedding(doc)
            self.document_embeddings.append(embedding)
        
        self.document_embeddings = np.array(self.document_embeddings)
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(self.document_embeddings)
        
        # Store results
        self.clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            self.clusters[label].append(i)
        
        # Calculate cluster centers
        for cluster_id, doc_indices in self.clusters.items():
            cluster_embeddings = self.document_embeddings[doc_indices]
            self.cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        print(f"Clustering completed. Found {len(self.clusters)} clusters.")
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze and describe the clusters"""
        
        print("\n" + "=" * 50)
        print("CLUSTER ANALYSIS")
        print("=" * 50)
        
        for cluster_id, doc_indices in self.clusters.items():
            print(f"\nCluster {cluster_id} ({len(doc_indices)} documents):")
            print("-" * 30)
            
            # Show sample documents
            for i, doc_idx in enumerate(doc_indices[:3]):  # Show first 3
                doc = self.documents[doc_idx]
                print(f"  {i+1}. {doc[:60]}...")
            
            if len(doc_indices) > 3:
                print(f"  ... and {len(doc_indices) - 3} more documents")
            
            # Analyze common words
            self.analyze_cluster_keywords(cluster_id)
    
    def analyze_cluster_keywords(self, cluster_id):
        """Find characteristic keywords for a cluster"""
        
        doc_indices = self.clusters[cluster_id]
        
        # Collect all words in cluster
        cluster_words = defaultdict(int)
        for doc_idx in doc_indices:
            words = self.documents[doc_idx].lower().split()
            for word in words:
                cluster_words[word] += 1
        
        # Find most common words
        sorted_words = sorted(cluster_words.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in sorted_words[:5]]
        
        print(f"  Top keywords: {', '.join(top_words)}")
    
    def visualize_clusters(self):
        """Visualize document clusters"""
        
        if len(self.document_embeddings) == 0:
            print("No embeddings to visualize")
            return
        
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Reduce dimensionality for visualization
        if self.document_embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(self.document_embeddings)
        else:
            embeddings_2d = self.document_embeddings
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for cluster_id, doc_indices in self.clusters.items():
            cluster_points = embeddings_2d[doc_indices]
            color = colors[cluster_id % len(colors)]
            
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                c=color, 
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=50
            )
            
            # Add cluster center
            if cluster_id in self.cluster_centers:
                center_2d = tsne.transform(self.cluster_centers[cluster_id].reshape(1, -1))
                plt.scatter(
                    center_2d[0, 0], 
                    center_2d[0, 1],
                    c=color,
                    marker='x',
                    s=200,
                    linewidths=3
                )
        
        plt.title('Document Clusters (t-SNE visualization)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def find_similar_documents(self, query_doc, top_k=5):
        """Find documents similar to a query document"""
        
        query_embedding = self.document_to_embedding(query_doc)
        
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nDocuments similar to: '{query_doc[:50]}...'")
        print("-" * 50)
        
        for rank, (doc_idx, similarity) in enumerate(similarities[:top_k], 1):
            doc = self.documents[doc_idx]
            print(f"{rank}. ({similarity:.3f}) {doc[:60]}...")

# Create sample documents for clustering
def create_clustering_documents():
    """Create documents from different domains for clustering"""
    
    documents = [
        # Technology cluster
        "Advanced computer algorithms improve software performance significantly",
        "Programming languages evolution and software development trends",
        "Technology innovation drives computer science research forward",
        "Software engineering principles for algorithm design",
        
        # Science cluster
        "Scientific research methodology and experimental data analysis",
        "Hypothesis testing and research data interpretation",
        "Experimental design for scientific research projects",
        "Data analysis techniques in scientific experiments",
        
        # Business cluster
        "Market analysis and investment strategy development",
        "Company profit optimization and business strategy",
        "Investment portfolio management and market trends",
        "Business strategy and market competition analysis",
        
        # Health cluster
        "Medical treatment advances in hospital patient care",
        "Doctor consultation and medicine prescription guidelines",
        "Patient treatment protocols in hospital settings",
        "Medicine research and medical treatment innovations"
    ]
    
    return documents

# Demonstrate document clustering
print("\n" + "=" * 60)
print("DOCUMENT CLUSTERING WITH EMBEDDINGS")
print("=" * 60)

# Create clustering system
clustering_system = DocumentClusteringSystem()
clustering_system.create_sample_embeddings()

# Create and cluster documents
clustering_docs = create_clustering_documents()
cluster_labels = clustering_system.cluster_documents(clustering_docs, n_clusters=4)

# Analyze results
clustering_system.analyze_clusters()

# Visualize clusters
clustering_system.visualize_clusters()

# Test similarity search
test_query = "new computer programming technology"
clustering_system.find_similar_documents(test_query)
```

## ⚖️ Bias Detection and Mitigation in Embeddings

Word embeddings can inherit and amplify societal biases. Let's learn to detect and mitigate them:

```python
class BiasDetectionSystem:
    """System for detecting and mitigating bias in word embeddings"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.bias_tests = {}
        self.debiased_embeddings = {}
        
    def create_biased_embeddings(self):
        """Create sample embeddings that demonstrate common biases"""
        
        print("Creating sample embeddings with inherent biases...")
        
        # Gender-biased embeddings
        embeddings = {
            # Professions (biased towards gender stereotypes)
            'doctor': np.array([0.7, 0.3, 0.5, 0.8, 0.2]),      # Male-leaning
            'nurse': np.array([0.2, 0.8, 0.6, 0.3, 0.7]),       # Female-leaning
            'engineer': np.array([0.9, 0.1, 0.4, 0.9, 0.1]),    # Male-leaning
            'teacher': np.array([0.3, 0.7, 0.8, 0.2, 0.8]),     # Female-leaning
            
            # Gender words
            'man': np.array([1.0, 0.0, 0.5, 0.8, 0.2]),
            'woman': np.array([0.0, 1.0, 0.6, 0.2, 0.8]),
            'he': np.array([0.9, 0.1, 0.5, 0.8, 0.2]),
            'she': np.array([0.1, 0.9, 0.6, 0.2, 0.8]),
            
            # Adjectives (showing bias)
            'strong': np.array([0.8, 0.2, 0.3, 0.9, 0.1]),      # Male-associated
            'gentle': np.array([0.1, 0.9, 0.7, 0.1, 0.9]),      # Female-associated
            'leader': np.array([0.8, 0.2, 0.4, 0.9, 0.1]),      # Male-associated
            'caring': np.array([0.2, 0.8, 0.8, 0.1, 0.9]),      # Female-associated
            
            # Race-related words (simplified for demonstration)
            'european': np.array([0.5, 0.5, 0.8, 0.6, 0.4]),
            'african': np.array([0.5, 0.5, 0.2, 0.4, 0.6]),
            'asian': np.array([0.5, 0.5, 0.6, 0.7, 0.3]),
            
            # Neutral words
            'person': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            'human': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            'individual': np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        }
        
        self.embedding_model = embeddings
        print(f"Created {len(embeddings)} biased embeddings")
    
    def test_gender_bias(self):
        """Test for gender bias in profession-related words"""
        
        print("\n" + "=" * 50)
        print("GENDER BIAS DETECTION")
        print("=" * 50)
        
        # Define gender direction
        male_vector = self.embedding_model['man']
        female_vector = self.embedding_model['woman']
        gender_direction = male_vector - female_vector
        
        # Test professions
        professions = ['doctor', 'nurse', 'engineer', 'teacher']
        
        print("Gender bias in professions:")
        print("(Positive = Male-leaning, Negative = Female-leaning)")
        
        bias_scores = {}
        
        for profession in professions:
            if profession in self.embedding_model:
                prof_vector = self.embedding_model[profession]
                
                # Project profession onto gender direction
                bias_score = np.dot(prof_vector, gender_direction) / np.linalg.norm(gender_direction)
                bias_scores[profession] = bias_score
                
                bias_direction = "Male-leaning" if bias_score > 0.1 else "Female-leaning" if bias_score < -0.1 else "Neutral"
                
                print(f"  {profession}: {bias_score:.3f} ({bias_direction})")
        
        return bias_scores
    
    def test_analogy_bias(self):
        """Test bias using word analogies"""
        
        print("\n" + "=" * 50)
        print("ANALOGY BIAS DETECTION")
        print("=" * 50)
        
        # Test analogies like: man:doctor :: woman:?
        analogies = [
            ('man', 'doctor', 'woman'),
            ('man', 'engineer', 'woman'),
            ('he', 'leader', 'she'),
            ('man', 'strong', 'woman')
        ]
        
        print("Testing gender analogies:")
        
        for word1, word2, word3 in analogies:
            if all(w in self.embedding_model for w in [word1, word2, word3]):
                # Compute analogy: word1:word2 :: word3:?
                analogy_vector = (self.embedding_model[word2] - 
                                self.embedding_model[word1] + 
                                self.embedding_model[word3])
                
                # Find closest word
                best_word = None
                best_similarity = -1
                
                for word, embedding in self.embedding_model.items():
                    if word not in [word1, word2, word3]:
                        similarity = cosine_similarity(
                            analogy_vector.reshape(1, -1),
                            embedding.reshape(1, -1)
                        )[0, 0]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_word = word
                
                print(f"  {word1}:{word2} :: {word3}:{best_word} (similarity: {best_similarity:.3f})")
    
    def test_association_bias(self):
        """Test word association biases"""
        
        print("\n" + "=" * 50)
        print("WORD ASSOCIATION BIAS")
        print("=" * 50)
        
        # Define word sets
        male_words = ['man', 'he']
        female_words = ['woman', 'she']
        career_words = ['doctor', 'engineer']
        family_words = ['caring', 'gentle']
        
        print("Testing stereotype associations:")
        
        # Test male-career vs female-career associations
        male_career_sim = []
        female_career_sim = []
        
        for male_word in male_words:
            for career_word in career_words:
                if male_word in self.embedding_model and career_word in self.embedding_model:
                    sim = cosine_similarity(
                        self.embedding_model[male_word].reshape(1, -1),
                        self.embedding_model[career_word].reshape(1, -1)
                    )[0, 0]
                    male_career_sim.append(sim)
        
        for female_word in female_words:
            for career_word in career_words:
                if female_word in self.embedding_model and career_word in self.embedding_model:
                    sim = cosine_similarity(
                        self.embedding_model[female_word].reshape(1, -1),
                        self.embedding_model[career_word].reshape(1, -1)
                    )[0, 0]
                    female_career_sim.append(sim)
        
        avg_male_career = np.mean(male_career_sim) if male_career_sim else 0
        avg_female_career = np.mean(female_career_sim) if female_career_sim else 0
        
        print(f"  Male-Career association: {avg_male_career:.3f}")
        print(f"  Female-Career association: {avg_female_career:.3f}")
        print(f"  Bias (Male-Female): {avg_male_career - avg_female_career:.3f}")
        
        if avg_male_career > avg_female_career + 0.1:
            print("  ⚠️  Career bias toward males detected")
        elif avg_female_career > avg_male_career + 0.1:
            print("  ⚠️  Career bias toward females detected")
        else:
            print("  ✓ No significant career bias detected")
    
    def debias_embeddings(self, method='linear_projection'):
        """Remove bias from embeddings"""
        
        print("\n" + "=" * 50)
        print("BIAS MITIGATION")
        print("=" * 50)
        
        if method == 'linear_projection':
            self.linear_projection_debiasing()
        elif method == 'neutralization':
            self.neutralization_debiasing()
        else:
            print(f"Unknown debiasing method: {method}")
    
    def linear_projection_debiasing(self):
        """Remove gender component using linear projection"""
        
        print("Applying linear projection debiasing...")
        
        # Define gender direction
        male_vector = self.embedding_model['man']
        female_vector = self.embedding_model['woman']
        gender_direction = male_vector - female_vector
        gender_direction = gender_direction / np.linalg.norm(gender_direction)
        
        # Create debiased embeddings
        self.debiased_embeddings = {}
        
        for word, embedding in self.embedding_model.items():
            if word not in ['man', 'woman', 'he', 'she']:  # Keep gender words as is
                # Remove gender component
                gender_component = np.dot(embedding, gender_direction) * gender_direction
                debiased_embedding = embedding - gender_component
                self.debiased_embeddings[word] = debiased_embedding
            else:
                self.debiased_embeddings[word] = embedding
        
        print("Linear projection debiasing completed")
    
    def neutralization_debiasing(self):
        """Neutralize specific words to remove bias"""
        
        print("Applying neutralization debiasing...")
        
        # Words to neutralize (professions, adjectives)
        words_to_neutralize = ['doctor', 'nurse', 'engineer', 'teacher', 'strong', 'gentle', 'leader', 'caring']
        
        self.debiased_embeddings = self.embedding_model.copy()
        
        # Calculate neutral direction (average of male and female)
        male_vector = self.embedding_model['man']
        female_vector = self.embedding_model['woman']
        neutral_vector = (male_vector + female_vector) / 2
        
        for word in words_to_neutralize:
            if word in self.embedding_model:
                # Move word toward neutral position
                original = self.embedding_model[word]
                neutralized = 0.7 * original + 0.3 * neutral_vector
                self.debiased_embeddings[word] = neutralized
        
        print("Neutralization debiasing completed")
    
    def evaluate_debiasing(self):
        """Evaluate the effectiveness of debiasing"""
        
        print("\n" + "=" * 50)
        print("DEBIASING EVALUATION")
        print("=" * 50)
        
        if not self.debiased_embeddings:
            print("No debiased embeddings available")
            return
        
        print("Comparing bias before and after debiasing:")
        
        # Test gender bias in professions
        professions = ['doctor', 'nurse', 'engineer', 'teacher']
        
        male_vector = self.embedding_model['man']
        female_vector = self.embedding_model['woman']
        gender_direction = male_vector - female_vector
        
        debiased_male = self.debiased_embeddings['man']
        debiased_female = self.debiased_embeddings['woman']
        debiased_gender_direction = debiased_male - debiased_female
        
        print("\nGender bias scores (original vs debiased):")
        
        for profession in professions:
            if profession in self.embedding_model and profession in self.debiased_embeddings:
                # Original bias
                original_bias = np.dot(self.embedding_model[profession], gender_direction) / np.linalg.norm(gender_direction)
                
                # Debiased bias
                debiased_bias = np.dot(self.debiased_embeddings[profession], debiased_gender_direction) / np.linalg.norm(debiased_gender_direction)
                
                bias_reduction = abs(original_bias) - abs(debiased_bias)
                
                print(f"  {profession}:")
                print(f"    Original: {original_bias:.3f}")
                print(f"    Debiased: {debiased_bias:.3f}")
                print(f"    Reduction: {bias_reduction:.3f}")
    
    def create_bias_report(self):
        """Create comprehensive bias analysis report"""
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE BIAS ANALYSIS REPORT")
        print("=" * 60)
        
        # Test all bias types
        gender_bias = self.test_gender_bias()
        self.test_analogy_bias()
        self.test_association_bias()
        
        # Apply debiasing
        self.debias_embeddings()
        
        # Evaluate debiasing
        self.evaluate_debiasing()
        
        # Summary
        print("\n" + "=" * 60)
        print("BIAS MITIGATION SUMMARY")
        print("=" * 60)
        
        print("✓ Detected gender bias in profession-related words")
        print("✓ Applied linear projection debiasing method")
        print("✓ Reduced bias while preserving semantic relationships")
        print("\nRecommendations:")
        print("• Use debiased embeddings for fair applications")
        print("• Regularly audit embeddings for emerging biases")
        print("• Consider multiple debiasing techniques")
        print("• Evaluate downstream task fairness")

# Demonstrate bias detection and mitigation
print("\n" + "=" * 60)
print("BIAS DETECTION AND MITIGATION")
print("=" * 60)

# Create bias detection system
bias_detector = BiasDetectionSystem(None)
bias_detector.create_biased_embeddings()

# Run comprehensive bias analysis
bias_detector.create_bias_report()
```

## 🏋️ Final Project: Multi-Modal Content Discovery Platform

Build a comprehensive content discovery platform that combines all the techniques we've learned:

```python
def build_content_discovery_platform():
    """
    Build a comprehensive content discovery platform
    
    Requirements:
    1. Semantic search across multiple content types
    2. Personalized recommendations using embeddings
    3. Automatic content clustering and categorization
    4. Bias-aware ranking and recommendation
    5. Real-time similarity updates
    6. Multi-modal content support (text, metadata, tags)
    
    Features to implement:
    • Hybrid search (semantic + lexical + collaborative)
    • Cold start problem handling for new users/items
    • Explainable recommendations
    • A/B testing framework for recommendation algorithms
    • Content drift detection and model updates
    • Fair ranking across different demographic groups
    
    Evaluation metrics:
    • Search relevance (precision@k, recall@k, nDCG)
    • Recommendation quality (diversity, novelty, serendipity)
    • Fairness metrics (demographic parity, equality of opportunity)
    • System performance (latency, throughput)
    """
    
    # Your implementation here
    pass

# Integration test scenarios
test_scenarios = [
    {
        'name': 'E-commerce Product Discovery',
        'content_types': ['product descriptions', 'reviews', 'categories'],
        'user_actions': ['search', 'view', 'purchase', 'rate'],
        'challenges': ['cold start', 'seasonality', 'price sensitivity']
    },
    {
        'name': 'Academic Paper Recommendation',
        'content_types': ['abstracts', 'full text', 'citations', 'authors'],
        'user_actions': ['search', 'download', 'cite', 'bookmark'],
        'challenges': ['domain expertise', 'recency bias', 'filter bubbles']
    },
    {
        'name': 'News Content Curation',
        'content_types': ['articles', 'headlines', 'categories', 'sources'],
        'user_actions': ['read', 'share', 'comment', 'subscribe'],
        'challenges': ['real-time updates', 'bias mitigation', 'diversity']
    }
]
```

## 💡 Key Takeaways

1. **Semantic search understands meaning** - Beyond keyword matching to concept similarity
2. **Embeddings power modern recommendation systems** - Content-based and collaborative filtering
3. **Document clustering reveals hidden patterns** - Automatic content organization
4. **Bias detection is crucial** - Embeddings can amplify societal biases
5. **Multiple techniques work better together** - Hybrid approaches outperform single methods
6. **Evaluation must be comprehensive** - Technical metrics + fairness + user experience

## 🚀 What's Next?

You've mastered word embedding applications! You're now ready to explore [Information Extraction](../04_information_extraction/01_ner_fundamentals.md) to learn how to extract structured information from unstructured text.

**Coming up:**

- Named Entity Recognition (NER)
- Relation extraction
- Knowledge graph construction
- Event extraction and temporal reasoning

Ready to extract structured knowledge from text? Let's continue!
