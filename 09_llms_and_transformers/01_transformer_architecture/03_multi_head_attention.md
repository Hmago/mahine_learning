# Multi-Head Attention 👥

Imagine you're trying to understand a complex movie. You might want to pay attention to multiple things simultaneously:
- **Director 1 (Plot Expert):** Focuses on story progression and character development
- **Director 2 (Visual Expert):** Focuses on cinematography and visual elements  
- **Director 3 (Audio Expert):** Focuses on dialogue and sound effects
- **Director 4 (Emotion Expert):** Focuses on character emotions and relationships

Multi-head attention works exactly like this - it's like having multiple specialized "directors" each focusing on different aspects of the text!

## 🧠 The Core Concept

### Single Attention Head Limitation

With single attention, we get one perspective:

**Sentence:** "The cat sat on the mat"

**Single attention might focus on:** Subject-verb relationships
- "cat" ↔ "sat" (grammatical connection)

**But we miss:** 
- Spatial relationships ("sat" ↔ "on" ↔ "mat")
- Article relationships ("the" ↔ "cat", "the" ↔ "mat")
- Action-location relationships ("sat" ↔ "mat")

### Multi-Head Attention Solution

With multiple heads, we get multiple perspectives simultaneously:

**Head 1 (Grammar Expert):** "cat" ↔ "sat" (subject-verb)
**Head 2 (Location Expert):** "sat" ↔ "on" ↔ "mat" (spatial)
**Head 3 (Reference Expert):** "the" ↔ "cat", "the" ↔ "mat" (determiners)
**Head 4 (Action Expert):** "cat" ↔ "sat" ↔ "mat" (action-context)

## 🎯 Why Multiple Heads Matter

### Real-World Example: Understanding Complex Sentences

**Sentence:** "The CEO of the company that developed the revolutionary AI system announced remarkable quarterly profits."

**What different heads might focus on:**

**Head 1 - Grammatical Relationships:**
- "CEO" ↔ "announced" (main subject-verb)
- "company" ↔ "developed" (subordinate clause)

**Head 2 - Entity Relationships:**
- "CEO" ↔ "company" (organizational hierarchy)
- "company" ↔ "AI system" (ownership/creation)

**Head 3 - Descriptive Relationships:**
- "revolutionary" ↔ "AI system" (adjective-noun)
- "remarkable" ↔ "quarterly profits" (adjective-noun)

**Head 4 - Logical Flow:**
- "AI system" ↔ "profits" (cause-effect reasoning)
- "developed" ↔ "announced" (temporal sequence)

## 🏗️ How Multi-Head Attention Works

### Step 1: Create Multiple Query-Key-Value Sets

Instead of one set of Q, K, V matrices, we create multiple sets:

```python
# Single head (simplified)
Q = input × W_Q    # One query matrix
K = input × W_K    # One key matrix  
V = input × W_V    # One value matrix

# Multi-head (8 heads)
Q1 = input × W_Q1, Q2 = input × W_Q2, ..., Q8 = input × W_Q8
K1 = input × W_K1, K2 = input × W_K2, ..., K8 = input × W_K8  
V1 = input × W_V1, V2 = input × W_V2, ..., V8 = input × W_V8
```

### Step 2: Parallel Attention Computation

Each head computes attention independently:

```python
# Each head computes its own attention
head1_output = attention(Q1, K1, V1)
head2_output = attention(Q2, K2, V2)
# ... up to head8_output
```

### Step 3: Concatenate and Project

Combine all head outputs:

```python
# Concatenate all heads
multi_head_output = concatenate([head1_output, head2_output, ..., head8_output])

# Final linear projection
final_output = multi_head_output × W_O
```

## 🎨 Visual Analogy: The Expert Panel

Think of multi-head attention like a panel of experts analyzing the same document:

### The Document: "Scientists discovered water on Mars using advanced telescopes."

**Expert 1 - Scientific Relationships:**
- "Scientists" ↔ "discovered" (agent-action)
- "telescopes" ↔ "discovered" (tool-action)

**Expert 2 - Object Relationships:**
- "water" ↔ "Mars" (location relationship)
- "advanced" ↔ "telescopes" (descriptor relationship)

**Expert 3 - Causal Relationships:**
- "using" ↔ "telescopes" ↔ "discovered" (method-tool-result)

**Expert 4 - Semantic Relationships:**
- "Scientists" ↔ "advanced telescopes" (expertise connection)
- "Mars" ↔ "water" ↔ "discovered" (discovery context)

### The Magic: Combining Perspectives

Each expert provides their analysis, then a "synthesis expert" combines all perspectives into a comprehensive understanding.

## 🧮 The Mathematics (Intuitive)

### Dimension Management

**Key insight:** Each head works on a smaller dimension!

If our model has dimension 512 and 8 heads:
- Each head works with dimension 512/8 = 64
- This keeps computation manageable
- Each head specializes in a smaller space

```python
# Original dimensions
input_dim = 512
num_heads = 8
head_dim = input_dim // num_heads  # = 64

# Each head processes smaller chunks
for head in range(num_heads):
    q_head = input[:, head*head_dim:(head+1)*head_dim]
    k_head = input[:, head*head_dim:(head+1)*head_dim]
    v_head = input[:, head*head_dim:(head+1)*head_dim]
    
    head_output = attention(q_head, k_head, v_head)
```

## 🌟 What Each Head Learns

### Empirical Discoveries

Researchers have found that different heads spontaneously learn to focus on:

**Head Types in BERT:**

**Syntactic Heads:**
- Subject-verb relationships
- Noun-adjective relationships
- Prepositional phrases

**Semantic Heads:**
- Word similarity and synonyms
- Antonym relationships
- Conceptual associations

**Positional Heads:**
- Beginning/end of sentences
- Relative positioning
- Sequence ordering

**Reference Heads:**
- Pronoun resolution ("it" → "cat")
- Coreference chains
- Entity tracking

### Example: GPT Head Specialization

**In GPT models, different heads learn:**

**Head 1:** "When I say 'The cat', this head focuses on 'cat' when generating the next word"

**Head 2:** "This head tracks whether we're in a question context or statement context"

**Head 3:** "This head focuses on maintaining tense consistency throughout the text"

**Head 4:** "This head tracks emotional tone and maintains consistent sentiment"

## 🚀 Why Multi-Head Attention Is So Powerful

### 1. Parallel Specialization

**Traditional approach:** One model tries to learn everything
**Multi-head approach:** Multiple specialists work in parallel

**Analogy:** Like having a medical team instead of one general doctor:
- Cardiologist focuses on heart
- Neurologist focuses on brain  
- Orthopedist focuses on bones
- General practitioner coordinates overall care

### 2. Robust Understanding

If one head makes an error, other heads can compensate:

**Example:** Head 1 incorrectly thinks "bank" means "river bank"
**Compensation:** Head 2 sees "deposit" and "money" and corrects the interpretation

### 3. Emergent Capabilities

Multiple heads working together create capabilities that no single head has:

**Example:** Understanding sarcasm requires:
- Head 1: Understanding literal meaning
- Head 2: Detecting tone indicators
- Head 3: Understanding context
- Head 4: Detecting contradiction between tone and content

## 🛠️ Code Example: Simple Multi-Head Implementation

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create weight matrices for each head
        self.W_q = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.W_k = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.W_v = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        
        # Output projection
        self.W_o = np.random.randn(embed_dim, embed_dim)
    
    def attention(self, q, k, v):
        """Single head attention"""
        scores = np.dot(q, k.T) / np.sqrt(self.head_dim)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return np.dot(weights, v)
    
    def forward(self, x):
        head_outputs = []
        
        # Compute attention for each head
        for i in range(self.num_heads):
            q = np.dot(x, self.W_q[i])
            k = np.dot(x, self.W_k[i]) 
            v = np.dot(x, self.W_v[i])
            
            head_output = self.attention(q, k, v)
            head_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head_output = np.concatenate(head_outputs, axis=-1)
        
        # Final projection
        output = np.dot(multi_head_output, self.W_o)
        return output

# Example usage
embed_dim = 128
num_heads = 8
seq_length = 10

# Create multi-head attention layer
mha = MultiHeadAttention(embed_dim, num_heads)

# Input sequence (seq_length × embed_dim)
x = np.random.randn(seq_length, embed_dim)

# Forward pass
output = mha.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## 🔍 Attention Pattern Visualization

### What Attention Patterns Look Like

When we visualize attention patterns, we see fascinating behaviors:

**Sentence:** "The cat that I saw yesterday is sleeping"

**Head 1 Pattern (Grammar):**
```
         The  cat  that  I   saw  yesterday  is  sleeping
The      0.1  0.0  0.0   0.0 0.0  0.0       0.0 0.0
cat      0.1  0.2  0.0   0.0 0.0  0.0       0.1 0.6  ← attends to "sleeping"
that     0.0  0.8  0.1   0.0 0.0  0.0       0.0 0.1  ← attends to "cat"
I        0.0  0.0  0.0   0.3 0.6  0.0       0.0 0.1  ← attends to "saw"
saw      0.0  0.2  0.0   0.7 0.1  0.0       0.0 0.0  ← attends to "I"
yesterday 0.0 0.0  0.0   0.0 0.9  0.1       0.0 0.0  ← attends to "saw"
is       0.0  0.1  0.0   0.0 0.0  0.0       0.2 0.7  ← attends to "sleeping"
sleeping 0.0  0.8  0.0   0.0 0.0  0.0       0.1 0.1  ← attends to "cat"
```

**Head 2 Pattern (Time/Reference):**
```
         The  cat  that  I   saw  yesterday  is  sleeping
...
yesterday 0.0 0.0  0.0   0.1 0.8  0.1       0.0 0.0  ← relates to "saw"
...
```

## 🤯 Surprising Discoveries

### 1. Spontaneous Linguistic Structure Discovery

Multi-head attention spontaneously discovers:
- **Syntax trees:** Heads learn grammatical parse structures
- **Semantic roles:** Agent, patient, instrument relationships
- **Discourse structure:** Topic flow and coherence

### 2. Cross-Lingual Transfer

Heads trained on English spontaneously work for other languages:
- Grammar heads transfer to similar language families
- Semantic heads capture universal concepts
- Position heads adapt to different word orders

### 3. Hierarchical Representations

Different layers learn different abstractions:
- **Early layers:** Focus on local patterns (bigrams, trigrams)
- **Middle layers:** Focus on syntactic relationships
- **Late layers:** Focus on semantic and logical relationships

## 💡 Design Choices and Trade-offs

### Number of Heads

**Fewer heads (2-4):**
- ✅ Faster computation
- ✅ Less overfitting risk
- ❌ Limited perspective diversity

**More heads (8-16):**
- ✅ Rich diverse perspectives
- ✅ Better complex pattern capture
- ❌ Higher computational cost
- ❌ Potential redundancy

### Head Dimension Size

**Larger head dimensions:**
- ✅ More expressive power per head
- ❌ Fewer heads possible (fixed total dimension)

**Smaller head dimensions:**
- ✅ More heads possible
- ✅ Better specialization
- ❌ Less expressive power per head

## 🎓 What You've Mastered

You now understand:

- ✅ **Why multi-head:** Different perspectives on the same data
- ✅ **How it works:** Parallel attention computation + concatenation
- ✅ **What heads learn:** Syntax, semantics, position, reference
- ✅ **The power:** Robust, specialized, emergent understanding
- ✅ **Implementation:** The mathematical and coding foundations

## 🔮 Coming Next: Positional Encoding

Great question from earlier: "How does a transformer know word order if it processes everything in parallel?"

The answer is **positional encoding** - a clever way to inject sequence information into parallel processing. We'll explore:

- Why position matters in language
- Different types of positional encoding
- How to handle sequences longer than training data
- The math behind sinusoidal encoding

## 🧠 Think About This

**Reflection Question:** If you were designing attention heads for understanding code instead of natural language, what different types of relationships would you want different heads to specialize in?

*Think about: function calls, variable scope, data flow, type relationships, import dependencies...*

Ready to learn how transformers understand sequence order? Let's dive into positional encoding! 🚀
