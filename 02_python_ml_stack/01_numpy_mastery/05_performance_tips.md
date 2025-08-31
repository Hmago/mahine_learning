# Performance Optimization: Making NumPy Lightning Fast

## 🤔 Why Worry About Performance?

Imagine processing customer data:
- **Small dataset**: 1,000 customers → finishes in seconds
- **Real dataset**: 10,000,000 customers → could take hours or crash

The difference between a data scientist who can handle small toy datasets and one who can tackle real-world problems often comes down to **performance optimization**.

## 🎯 The Performance Hierarchy

### Level 1: Basic NumPy (Already Fast)
```python
import numpy as np

data = np.random.rand(1000000)
result = data * 2 + 1  # Already 50x faster than pure Python
```

### Level 2: Optimized NumPy (Faster)
```python
# Use in-place operations
data *= 2
data += 1  # No intermediate arrays created
```

### Level 3: Advanced Techniques (Fastest)
```python
# Vectorized operations with optimal memory access
np.add(data, 1, out=data)  # Explicitly use output parameter
```

## 🚀 Memory Optimization: The Hidden Performance Killer

### Understanding Memory Layout

```python
# Memory-efficient array creation
def analyze_memory_usage():
    # Bad: Creates multiple intermediate arrays
    data = np.random.rand(1000, 1000)
    result = ((data * 2) + 1) ** 2  # Creates 3 intermediate arrays!
    
    # Good: Chain operations efficiently
    data = np.random.rand(1000, 1000)
    np.square(data * 2 + 1, out=data)  # Reuse the same memory
    
    return data

# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

print(f"Memory before: {get_memory_usage():.1f} MB")
result = analyze_memory_usage()
print(f"Memory after: {get_memory_usage():.1f} MB")
```

### Choosing the Right Data Type

```python
# Data type choice dramatically affects memory
def compare_data_types():
    size = (10000, 1000)
    
    # Different precisions for different use cases
    int8_array = np.random.randint(0, 256, size, dtype=np.int8)      # 1 byte per element
    int32_array = np.random.randint(0, 2**31, size, dtype=np.int32)  # 4 bytes per element
    float32_array = np.random.rand(*size).astype(np.float32)         # 4 bytes per element
    float64_array = np.random.rand(*size).astype(np.float64)         # 8 bytes per element
    
    print("Memory usage comparison:")
    print(f"int8:    {int8_array.nbytes / 1024**2:.1f} MB")
    print(f"int32:   {int32_array.nbytes / 1024**2:.1f} MB")
    print(f"float32: {float32_array.nbytes / 1024**2:.1f} MB")
    print(f"float64: {float64_array.nbytes / 1024**2:.1f} MB")

compare_data_types()
```

**Rule of thumb**:
- **Images**: Use `uint8` (0-255 range)
- **Financial data**: Use `float64` (high precision needed)
- **ML features**: Often `float32` is sufficient

## ⚡ Vectorization: Eliminating Loops

### The Anti-Pattern: Python Loops

```python
# Slow: Calculate customer value scores using loops
def slow_customer_scoring(customers):
    scores = []
    for customer in customers:
        age, income, spending = customer
        # Complex scoring formula
        score = (income * 0.3) + (spending * 0.5) + ((40 - abs(age - 40)) * 1000)
        scores.append(score)
    return np.array(scores)
```

### The Fast Way: Pure Vectorization

```python
# Fast: Same calculation using vectorization
def fast_customer_scoring(customers):
    ages = customers[:, 0]
    incomes = customers[:, 1]
    spending = customers[:, 2]
    
    # Vectorized calculation (no loops!)
    scores = (incomes * 0.3) + (spending * 0.5) + ((40 - np.abs(ages - 40)) * 1000)
    return scores

# Performance test
customers = np.random.rand(100000, 3) * [60, 100000, 5000]

import time

# Time the slow version
start = time.time()
slow_scores = slow_customer_scoring(customers)
slow_time = time.time() - start

# Time the fast version
start = time.time()
fast_scores = fast_customer_scoring(customers)
fast_time = time.time() - start

print(f"Slow method: {slow_time:.4f} seconds")
print(f"Fast method: {fast_time:.4f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x faster!")
```

## 🔥 Advanced Performance Techniques

### 1. Using NumPy's Built-in Functions

```python
# Leverage optimized implementations
data = np.random.rand(1000000)

# Instead of manual calculations
manual_mean = data.sum() / len(data)

# Use optimized built-ins
optimized_mean = np.mean(data)  # Faster and more accurate

# For complex statistics
stats = {
    'mean': np.mean(data),
    'median': np.median(data),
    'std': np.std(data),
    'percentiles': np.percentile(data, [25, 50, 75])
}
```

### 2. Efficient Aggregations

```python
# Sales analysis across multiple dimensions
# Shape: (stores, products, months)
sales_cube = np.random.poisson(100, (50, 20, 12))

# Efficient multi-dimensional aggregations
total_by_store = sales_cube.sum(axis=(1, 2))      # Sum across products and months
total_by_product = sales_cube.sum(axis=(0, 2))    # Sum across stores and months
total_by_month = sales_cube.sum(axis=(0, 1))      # Sum across stores and products

# Combined analysis
store_performance = total_by_store / total_by_store.mean()  # Relative performance
top_stores = np.argsort(store_performance)[-5:]  # Top 5 stores

print(f"Top performing stores: {top_stores}")
print(f"Their performance ratios: {store_performance[top_stores]:.2f}")
```

### 3. Chunked Processing for Large Datasets

```python
def process_large_dataset(data, chunk_size=10000):
    """Process large datasets in chunks to manage memory"""
    n_samples = data.shape[0]
    results = []
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = data[start:end]
        
        # Process chunk (example: complex calculation)
        chunk_result = np.sqrt(chunk ** 2 + np.sin(chunk))
        results.append(chunk_result)
    
    return np.concatenate(results)

# Example with 10 million data points
large_data = np.random.rand(10000000)
result = process_large_dataset(large_data)
print(f"Processed {len(large_data)} data points in chunks")
```

## 📊 Profiling Your Code: Finding Bottlenecks

### Simple Timing

```python
import time

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} took {end - start:.4f} seconds")
    return result

# Example usage
data = np.random.rand(1000000)
result = time_function(np.sqrt, data)
```

### Memory Profiling

```python
def profile_memory_usage(func, *args, **kwargs):
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    
    result = func(*args, **kwargs)
    
    mem_after = process.memory_info().rss / 1024**2
    mem_used = mem_after - mem_before
    
    print(f"{func.__name__} used {mem_used:.1f} MB of memory")
    return result
```

## 🎮 Real-World Performance Challenge

Let's optimize a customer segmentation algorithm:

```python
import numpy as np
import time

def slow_customer_segmentation(customers):
    """Slow version using Python loops"""
    segments = []
    for customer in customers:
        age, income, spending = customer
        
        if age < 30 and income > 50000:
            segment = "Young High Earner"
        elif age >= 30 and age < 50 and spending > 3000:
            segment = "Prime Spender"
        elif age >= 50 and income > 75000:
            segment = "Senior Wealthy"
        else:
            segment = "Standard"
        
        segments.append(segment)
    
    return segments

def fast_customer_segmentation(customers):
    """Fast version using vectorized operations"""
    ages = customers[:, 0]
    incomes = customers[:, 1]
    spending = customers[:, 2]
    
    # Vectorized conditions
    young_high_earner = (ages < 30) & (incomes > 50000)
    prime_spender = (ages >= 30) & (ages < 50) & (spending > 3000)
    senior_wealthy = (ages >= 50) & (incomes > 75000)
    
    # Create segments array
    segments = np.full(len(customers), "Standard", dtype='U20')
    segments[young_high_earner] = "Young High Earner"
    segments[prime_spender] = "Prime Spender"
    segments[senior_wealthy] = "Senior Wealthy"
    
    return segments

# Performance test
customers = np.random.rand(100000, 3) * [50, 100000, 5000]
customers[:, 0] += 20  # Ages 20-70

# Time both approaches
start = time.time()
slow_segments = slow_customer_segmentation(customers)
slow_time = time.time() - start

start = time.time()
fast_segments = fast_customer_segmentation(customers)
fast_time = time.time() - start

print(f"Slow segmentation: {slow_time:.4f} seconds")
print(f"Fast segmentation: {fast_time:.4f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x faster!")

# Verify results are the same
print(f"Results match: {np.array(slow_segments == fast_segments).all()}")
```

## 🎯 Key Performance Principles

1. **Vectorize everything**: Replace loops with array operations
2. **Minimize memory allocations**: Use in-place operations when possible
3. **Choose appropriate data types**: Don't use float64 when float32 will do
4. **Profile before optimizing**: Measure to find real bottlenecks
5. **Consider memory vs speed tradeoffs**: Sometimes copying is faster than in-place

## 🚀 Advanced Tips for Production Code

### 1. Parallel Processing

```python
# For CPU-intensive tasks, consider parallel processing
from multiprocessing import Pool

def parallel_processing_example():
    def process_chunk(chunk):
        # Complex calculation on data chunk
        return np.sqrt(chunk ** 3 + np.sin(chunk) * np.cos(chunk))
    
    # Large dataset
    large_data = np.random.rand(10000000)
    
    # Split into chunks for parallel processing
    chunks = np.array_split(large_data, 4)  # 4 processes
    
    # Process in parallel
    with Pool(4) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    final_result = np.concatenate(results)
    return final_result
```

### 2. Memory Mapping for Huge Datasets

```python
# For datasets too large for memory
def memory_mapped_processing():
    # Create a large file-backed array
    large_array = np.memmap('large_dataset.dat', dtype='float32', 
                           mode='w+', shape=(100000, 1000))
    
    # Fill with data
    large_array[:] = np.random.rand(100000, 1000)
    
    # Process in chunks without loading everything into memory
    chunk_means = []
    for i in range(0, 100000, 10000):
        chunk = large_array[i:i+10000]
        chunk_means.append(chunk.mean())
    
    return np.array(chunk_means)
```

## 🎯 Key Takeaways

1. **Performance matters at scale**: Optimizations become critical with large datasets
2. **Memory is often the bottleneck**: Watch memory usage as much as speed
3. **Profile first**: Don't optimize blindly - measure where time is spent
4. **Vectorization wins**: Almost always faster than loops
5. **Know your tools**: NumPy has optimized functions for most operations

## 🚀 What's Next?

Congratulations! You've mastered NumPy fundamentals. Now you're ready to tackle **Pandas** - the tool that makes working with real-world, messy data as smooth as working with clean arrays!
