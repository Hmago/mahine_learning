# Deployment Strategies: From Lab to Production

Learn how to deploy your deep learning models in real-world environments that serve millions of users. Master different deployment patterns, frameworks, and best practices for reliable AI systems.

## 🎯 What You'll Master

- **Deployment Patterns**: Choose the right strategy for your use case
- **Production Frameworks**: TensorFlow Serving, Kubernetes, cloud platforms
- **Scalability**: Handle growing traffic and data volumes
- **Reliability**: Build fault-tolerant ML systems

## 📚 The Deployment Landscape

### Understanding Deployment Requirements

**Real-World Scenarios:**
```
E-commerce: 10,000 recommendations/sec during Black Friday
Autonomous Vehicles: <10ms inference for safety-critical decisions
Medical Diagnosis: 99.9% uptime for hospital systems
Mobile Apps: Offline capability with <50MB models
```

**The Deployment Spectrum:**
```
Research → Prototype → MVP → Production → Scale
```

Think of deployment like opening a restaurant - you need the right kitchen, skilled staff, and systems to serve customers reliably!

## 1. Deployment Patterns

### Online Serving: Real-Time Predictions

**Concept:** Serve predictions immediately when users request them.

**Analogy:** Like a fast-food restaurant - take orders and serve immediately.

```python
import flask
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelServer:
    """Production-ready model serving class"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        self.prediction_count = 0
        self.start_time = time.time()
        
    def load_model(self, model_path):
        """Load and prepare model for inference"""
        try:
            model = torch.load(model_path, map_location=self.device)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, input_data):
        """Preprocess input data"""
        try:
            # Convert to tensor
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            
            # Normalize (example for image data)
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            
            # Add batch dimension if needed
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
                
            return input_tensor
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Invalid input format: {e}")
    
    def predict(self, input_tensor):
        """Make prediction with error handling"""
        try:
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            inference_time = time.time() - start_time
            self.prediction_count += 1
            
            logger.info(f"Prediction completed in {inference_time:.4f}s")
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'prediction_id': self.prediction_count
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# Flask API wrapper
app = Flask(__name__)
model_server = None

def initialize_server():
    """Initialize model server"""
    global model_server
    try:
        model_server = ModelServer(
            model_path='models/production_model.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("Model server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_server is None:
        return jsonify({'status': 'unhealthy', 'error': 'Model not loaded'}), 503
    
    uptime = time.time() - model_server.start_time
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'predictions_served': model_server.prediction_count,
        'model_device': str(model_server.device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        if 'input' not in data:
            return jsonify({'error': 'Missing input field'}), 400
        
        # Preprocess and predict
        input_tensor = model_server.preprocess(data['input'])
        result = model_server.predict(input_tensor)
        
        # Add metadata
        result['timestamp'] = datetime.utcnow().isoformat()
        result['model_version'] = 'v1.0'
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple inputs"""
    try:
        data = request.get_json()
        if 'inputs' not in data:
            return jsonify({'error': 'Missing inputs field'}), 400
        
        results = []
        for i, input_data in enumerate(data['inputs']):
            try:
                input_tensor = model_server.preprocess(input_data)
                result = model_server.predict(input_tensor)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for index {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'total_processed': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Advanced: Model versioning and A/B testing
class ModelVersionManager:
    """Manage multiple model versions for A/B testing"""
    
    def __init__(self):
        self.models = {}
        self.traffic_split = {}
        self.default_version = None
    
    def add_model(self, version, model_path, traffic_percentage=0):
        """Add a new model version"""
        self.models[version] = ModelServer(model_path)
        self.traffic_split[version] = traffic_percentage
        
        if self.default_version is None:
            self.default_version = version
    
    def get_model_for_request(self, user_id=None):
        """Select model version based on traffic split"""
        import random
        
        # Simple hash-based assignment for consistent user experience
        if user_id:
            random.seed(hash(user_id))
        
        rand_val = random.random() * 100
        cumulative = 0
        
        for version, percentage in self.traffic_split.items():
            cumulative += percentage
            if rand_val <= cumulative:
                return self.models[version], version
        
        # Fallback to default
        return self.models[self.default_version], self.default_version

# Production deployment with Gunicorn
def create_production_app():
    """Create production-ready Flask app"""
    
    # Initialize model server
    initialize_server()
    
    # Add production configurations
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Add request logging
    @app.before_request
    def log_request_info():
        logger.info(f"Request: {request.method} {request.url}")
    
    @app.after_request
    def log_response_info(response):
        logger.info(f"Response: {response.status_code}")
        return response
    
    return app

if __name__ == '__main__':
    # Development server
    initialize_server()
    app.run(host='0.0.0.0', port=5000, debug=True)

# Production deployment script
"""
# Install dependencies
pip install flask gunicorn torch torchvision

# Run with Gunicorn (production WSGI server)
gunicorn --workers 4 --bind 0.0.0.0:5000 --timeout 30 app:app

# With auto-reload for updates
gunicorn --workers 4 --bind 0.0.0.0:5000 --reload app:app

# For high-performance deployments
gunicorn --workers 8 --worker-class gevent --bind 0.0.0.0:5000 app:app
"""
```

### Batch Processing: High-Throughput Predictions

**Concept:** Process large datasets offline in batches.

**Analogy:** Like a factory assembly line - process many items efficiently in bulk.

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import logging
from pathlib import Path

class BatchProcessor:
    """High-performance batch processing for ML inference"""
    
    def __init__(self, model_path, batch_size=64, num_workers=4, device='cuda'):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
    def process_csv(self, input_path, output_path, feature_columns):
        """Process CSV file with predictions"""
        
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Create dataset
        dataset = CSVDataset(df, feature_columns)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"Processing {len(dataset)} samples...")
        start_time = time.time()
        
        predictions = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                
                # Get predictions
                outputs = self.model(batch_data)
                batch_predictions = torch.softmax(outputs, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                
                if batch_idx % 100 == 0:
                    processed = (batch_idx + 1) * self.batch_size
                    print(f"Processed {processed} samples...")
        
        # Save results
        prediction_df = pd.DataFrame(predictions, columns=[f'class_{i}' for i in range(predictions[0].shape[0])])
        result_df = pd.concat([df, prediction_df], axis=1)
        result_df.to_csv(output_path, index=False)
        
        elapsed = time.time() - start_time
        throughput = len(dataset) / elapsed
        print(f"Completed! Processed {len(dataset)} samples in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} samples/second")

class CSVDataset(Dataset):
    """Dataset for CSV data"""
    
    def __init__(self, dataframe, feature_columns):
        self.data = dataframe[feature_columns].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# Distributed batch processing
class DistributedBatchProcessor:
    """Process large datasets across multiple machines"""
    
    def __init__(self, model_path, num_processes=4):
        self.model_path = model_path
        self.num_processes = num_processes
    
    def process_large_dataset(self, data_dir, output_dir, chunk_size=10000):
        """Process large datasets by splitting into chunks"""
        
        data_files = list(Path(data_dir).glob('*.csv'))
        print(f"Found {len(data_files)} files to process")
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            
            for file_path in data_files:
                future = executor.submit(
                    self._process_single_file, 
                    file_path, 
                    output_dir
                )
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                result = future.result()
                print(f"Completed processing: {result}")
    
    def _process_single_file(self, input_path, output_dir):
        """Process a single file"""
        processor = BatchProcessor(self.model_path)
        
        output_path = Path(output_dir) / f"predictions_{input_path.name}"
        processor.process_csv(
            input_path, 
            output_path, 
            feature_columns=['feature1', 'feature2', 'feature3']  # Configure as needed
        )
        
        return str(output_path)

# Stream processing for continuous data
class StreamProcessor:
    """Process streaming data in real-time"""
    
    def __init__(self, model_path, buffer_size=1000):
        self.model = torch.load(model_path)
        self.model.eval()
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add_sample(self, sample):
        """Add sample to processing buffer"""
        self.buffer.append(sample)
        
        if len(self.buffer) >= self.buffer_size:
            return self.process_buffer()
        return None
    
    def process_buffer(self):
        """Process accumulated samples"""
        if not self.buffer:
            return []
        
        # Convert to tensor
        batch_data = torch.tensor(self.buffer)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch_data)
            predictions = torch.softmax(outputs, dim=1)
        
        # Clear buffer
        results = predictions.cpu().numpy().tolist()
        self.buffer = []
        
        return results

# Example usage
if __name__ == "__main__":
    # Batch processing
    processor = BatchProcessor(
        model_path='models/production_model.pth',
        batch_size=128,
        num_workers=8
    )
    
    # Process CSV file
    processor.process_csv(
        input_path='data/input_data.csv',
        output_path='data/predictions.csv',
        feature_columns=['feature1', 'feature2', 'feature3', 'feature4']
    )
    
    # Distributed processing
    distributed_processor = DistributedBatchProcessor(
        model_path='models/production_model.pth',
        num_processes=8
    )
    
    distributed_processor.process_large_dataset(
        data_dir='data/input_chunks/',
        output_dir='data/output_chunks/'
    )
```

### Edge Deployment: On-Device Intelligence

**Concept:** Deploy models directly on user devices for offline capability.

**Analogy:** Like having a personal chef in your kitchen instead of ordering delivery.

```python
# Mobile deployment with ONNX Runtime
import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple
import time

class EdgeModelRunner:
    """Optimized model runner for edge devices"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize edge model runner
        
        Args:
            model_path: Path to ONNX model
            providers: Execution providers (CPU, GPU, etc.)
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get model input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if len(input_shape) == 4 else 224
        self.input_width = input_shape[3] if len(input_shape) == 4 else 224
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Available providers: {providers}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        
        # Resize image
        image = cv2.resize(image, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: np.ndarray) -> Tuple[int, float, float]:
        """
        Make prediction on image
        
        Returns:
            predicted_class, confidence, inference_time
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Post-process
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        inference_time = time.time() - start_time
        
        return predicted_class, confidence, inference_time
    
    def benchmark(self, num_runs: int = 100) -> dict:
        """Benchmark model performance"""
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, self.input_height, self.input_width).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.session.run([self.output_name], {self.input_name: dummy_input})
            times.append(time.time() - start_time)
        
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }

# Mobile app integration (conceptual)
class MobileMLPipeline:
    """Complete ML pipeline for mobile deployment"""
    
    def __init__(self, model_path: str, classes: List[str]):
        self.model = EdgeModelRunner(model_path)
        self.classes = classes
        self.prediction_history = []
    
    def process_camera_frame(self, frame: np.ndarray) -> dict:
        """Process single camera frame"""
        
        # Make prediction
        pred_class, confidence, inf_time = self.model.predict(frame)
        
        # Convert to human-readable result
        result = {
            'class_name': self.classes[pred_class],
            'class_id': int(pred_class),
            'confidence': float(confidence),
            'inference_time_ms': inf_time * 1000,
            'timestamp': time.time()
        }
        
        # Store in history
        self.prediction_history.append(result)
        
        # Keep only recent predictions
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        return result
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.prediction_history:
            return {}
        
        inference_times = [p['inference_time_ms'] for p in self.prediction_history[-50:]]
        
        return {
            'avg_inference_ms': np.mean(inference_times),
            'avg_fps': 1000.0 / np.mean(inference_times),
            'predictions_made': len(self.prediction_history)
        }

# Example usage for mobile deployment
def deploy_to_mobile_demo():
    """Demo of mobile deployment workflow"""
    
    # Initialize mobile pipeline
    classes = ['cat', 'dog', 'bird', 'fish']
    pipeline = MobileMLPipeline(
        model_path='models/mobile_model.onnx',
        classes=classes
    )
    
    # Simulate camera input
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process frame
    result = pipeline.process_camera_frame(dummy_frame)
    print(f"Prediction: {result}")
    
    # Benchmark performance
    benchmark_results = pipeline.model.benchmark()
    print(f"Benchmark: {benchmark_results}")
    
    return pipeline

# Edge optimization utilities
class EdgeOptimizer:
    """Utilities for edge deployment optimization"""
    
    @staticmethod
    def profile_model(model_path: str) -> dict:
        """Profile model for edge deployment"""
        
        session = ort.InferenceSession(model_path)
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        # Calculate model size
        import os
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return {
            'model_size_mb': model_size_mb,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_shape': output_info.shape,
            'output_type': output_info.type,
            'providers': session.get_providers()
        }
    
    @staticmethod
    def test_different_providers(model_path: str) -> dict:
        """Test model with different execution providers"""
        
        providers_to_test = [
            ['CPUExecutionProvider'],
            ['CoreMLExecutionProvider', 'CPUExecutionProvider'],  # iOS
            ['OpenVINOExecutionProvider', 'CPUExecutionProvider'],  # Intel
        ]
        
        results = {}
        
        for providers in providers_to_test:
            try:
                runner = EdgeModelRunner(model_path, providers)
                benchmark = runner.benchmark(num_runs=50)
                
                provider_name = providers[0].replace('ExecutionProvider', '')
                results[provider_name] = benchmark
                
            except Exception as e:
                provider_name = providers[0].replace('ExecutionProvider', '')
                results[provider_name] = {'error': str(e)}
        
        return results

if __name__ == "__main__":
    # Demo edge deployment
    # pipeline = deploy_to_mobile_demo()
    
    # Profile model for edge deployment
    # profile_info = EdgeOptimizer.profile_model('models/mobile_model.onnx')
    # print(f"Model profile: {profile_info}")
    
    # Test different execution providers
    # provider_results = EdgeOptimizer.test_different_providers('models/mobile_model.onnx')
    # print(f"Provider benchmark: {provider_results}")
    
    pass
```

## 2. Production Frameworks

### TensorFlow Serving: Enterprise-Grade Model Serving

```python
# TensorFlow Serving setup and configuration
import tensorflow as tf
import requests
import json
import numpy as np

class TensorFlowServingClient:
    """Client for TensorFlow Serving"""
    
    def __init__(self, server_url='http://localhost:8501'):
        self.server_url = server_url
        self.model_name = None
        self.model_version = None
    
    def set_model(self, model_name, version='latest'):
        """Set model name and version"""
        self.model_name = model_name
        self.model_version = version
    
    def predict(self, input_data):
        """Make prediction via REST API"""
        
        url = f"{self.server_url}/v1/models/{self.model_name}"
        if self.model_version != 'latest':
            url += f"/versions/{self.model_version}"
        url += ":predict"
        
        # Prepare request
        data = {
            "instances": input_data.tolist() if isinstance(input_data, np.ndarray) else input_data
        }
        
        # Make request
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()['predictions']
        else:
            raise Exception(f"Prediction failed: {response.text}")
    
    def get_model_metadata(self):
        """Get model metadata"""
        
        url = f"{self.server_url}/v1/models/{self.model_name}/metadata"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get metadata: {response.text}")
    
    def get_model_status(self):
        """Get model serving status"""
        
        url = f"{self.server_url}/v1/models/{self.model_name}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get status: {response.text}")

# TensorFlow Serving Docker deployment
tf_serving_config = """
# Dockerfile for TensorFlow Serving
FROM tensorflow/serving:latest

# Copy model
COPY ./models /models

# Set environment variables
ENV MODEL_NAME=my_model
ENV MODEL_BASE_PATH=/models

# Expose port
EXPOSE 8501

# Start serving
CMD ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=${MODEL_NAME}", "--model_base_path=${MODEL_BASE_PATH}"]

# Build and run:
# docker build -t my-tf-serving .
# docker run -p 8501:8501 my-tf-serving
"""

# Model versioning with TensorFlow Serving
class ModelVersionManager:
    """Manage model versions in TensorFlow Serving"""
    
    def __init__(self, model_base_path):
        self.model_base_path = model_base_path
    
    def deploy_new_version(self, model, version_number):
        """Deploy new model version"""
        
        version_path = f"{self.model_base_path}/{version_number}"
        
        # Save model in SavedModel format
        tf.saved_model.save(model, version_path)
        
        print(f"Model version {version_number} deployed to {version_path}")
    
    def get_serving_config(self):
        """Generate serving configuration"""
        
        config = {
            "model_config_list": {
                "config": [
                    {
                        "name": "my_model",
                        "base_path": self.model_base_path,
                        "model_platform": "tensorflow",
                        "model_version_policy": {
                            "latest": {
                                "num_versions": 2  # Keep 2 latest versions
                            }
                        }
                    }
                ]
            }
        }
        
        return json.dumps(config, indent=2)

# Example usage
def test_tensorflow_serving():
    """Test TensorFlow Serving deployment"""
    
    # Initialize client
    client = TensorFlowServingClient('http://localhost:8501')
    client.set_model('my_model', 'latest')
    
    # Test prediction
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    try:
        predictions = client.predict(dummy_input)
        print(f"Predictions: {predictions}")
        
        # Get model info
        metadata = client.get_model_metadata()
        print(f"Model metadata: {metadata}")
        
        status = client.get_model_status()
        print(f"Model status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")

# Load balancing and auto-scaling
load_balancer_config = """
# nginx.conf for load balancing TensorFlow Serving
upstream tf_serving {
    server tf-serving-1:8501;
    server tf-serving-2:8501;
    server tf-serving-3:8501;
}

server {
    listen 80;
    
    location /v1/models/ {
        proxy_pass http://tf_serving;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
"""
```

### Kubernetes: Container Orchestration

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: my-ml-model:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/models/production_model.pth"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```python
# Kubernetes management with Python
from kubernetes import client, config
import yaml

class KubernetesMLDeployer:
    """Manage ML model deployments on Kubernetes"""
    
    def __init__(self, kubeconfig_path=None):
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_incluster_config()  # For in-cluster usage
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.autoscaling_v2 = client.AutoscalingV2Api()
    
    def deploy_model(self, deployment_config, namespace='default'):
        """Deploy ML model to Kubernetes"""
        
        try:
            # Create deployment
            response = self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment_config
            )
            
            print(f"Deployment created: {response.metadata.name}")
            return response
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            raise
    
    def update_model(self, deployment_name, new_image, namespace='default'):
        """Update model deployment with new image"""
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update image
            deployment.spec.template.spec.containers[0].image = new_image
            
            # Update deployment
            response = self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            print(f"Deployment updated: {response.metadata.name}")
            return response
            
        except Exception as e:
            print(f"Update failed: {e}")
            raise
    
    def scale_deployment(self, deployment_name, replicas, namespace='default'):
        """Scale deployment to specified number of replicas"""
        
        try:
            # Create scale object
            scale = client.V1Scale(
                spec=client.V1ScaleSpec(replicas=replicas)
            )
            
            # Scale deployment
            response = self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=scale
            )
            
            print(f"Deployment scaled to {replicas} replicas")
            return response
            
        except Exception as e:
            print(f"Scaling failed: {e}")
            raise
    
    def get_deployment_status(self, deployment_name, namespace='default'):
        """Get deployment status"""
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            return {
                'name': deployment.metadata.name,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'available_replicas': deployment.status.available_replicas or 0,
                'conditions': [
                    {
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
            
        except Exception as e:
            print(f"Failed to get status: {e}")
            return None
    
    def setup_autoscaling(self, deployment_name, min_replicas=2, max_replicas=10, 
                         cpu_target=70, namespace='default'):
        """Setup horizontal pod autoscaling"""
        
        hpa_config = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{deployment_name}-hpa",
                'namespace': namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': cpu_target
                            }
                        }
                    }
                ]
            }
        }
        
        try:
            response = self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=namespace,
                body=hpa_config
            )
            
            print(f"HPA created: {response.metadata.name}")
            return response
            
        except Exception as e:
            print(f"HPA creation failed: {e}")
            raise

# Example usage
def deploy_ml_model_to_k8s():
    """Example of deploying ML model to Kubernetes"""
    
    deployer = KubernetesMLDeployer()
    
    # Define deployment configuration
    deployment_config = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'ml-model-deployment',
            'labels': {'app': 'ml-model'}
        },
        'spec': {
            'replicas': 3,
            'selector': {
                'matchLabels': {'app': 'ml-model'}
            },
            'template': {
                'metadata': {
                    'labels': {'app': 'ml-model'}
                },
                'spec': {
                    'containers': [{
                        'name': 'ml-model',
                        'image': 'my-ml-model:v1.0',
                        'ports': [{'containerPort': 5000}],
                        'resources': {
                            'requests': {
                                'memory': '2Gi',
                                'cpu': '1000m'
                            },
                            'limits': {
                                'memory': '4Gi',
                                'cpu': '2000m'
                            }
                        }
                    }]
                }
            }
        }
    }
    
    # Deploy model
    deployment = deployer.deploy_model(deployment_config)
    
    # Setup autoscaling
    deployer.setup_autoscaling('ml-model-deployment')
    
    # Monitor deployment
    status = deployer.get_deployment_status('ml-model-deployment')
    print(f"Deployment status: {status}")
```

## 3. Cloud Platform Deployment

### AWS SageMaker Deployment

```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
import json

class SageMakerDeployer:
    """Deploy models to AWS SageMaker"""
    
    def __init__(self, region='us-east-1'):
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sagemaker_session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
    
    def deploy_pytorch_model(self, model_data_url, entry_point_script, 
                           endpoint_name, instance_type='ml.m5.large'):
        """Deploy PyTorch model to SageMaker endpoint"""
        
        # Create PyTorch model
        pytorch_model = PyTorchModel(
            model_data=model_data_url,
            role=self.role,
            entry_point=entry_point_script,
            framework_version='1.12',
            py_version='py38'
        )
        
        # Deploy to endpoint
        predictor = pytorch_model.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=1
        )
        
        print(f"Model deployed to endpoint: {endpoint_name}")
        return predictor
    
    def create_auto_scaling_policy(self, endpoint_name, min_capacity=1, max_capacity=10):
        """Setup auto-scaling for SageMaker endpoint"""
        
        application_autoscaling = boto3.client('application-autoscaling')
        
        # Register scalable target
        response = application_autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        
        # Create scaling policy
        policy_response = application_autoscaling.put_scaling_policy(
            PolicyName=f'{endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 70.0,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                }
            }
        )
        
        print(f"Auto-scaling policy created for {endpoint_name}")
        return policy_response

# SageMaker inference script (inference.py)
sagemaker_inference_script = """
import torch
import torch.nn.functional as F
import json
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    \"\"\"Load model for inference\"\"\"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_dir}/model.pth", map_location=device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    \"\"\"Parse input data\"\"\"
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return torch.tensor(data['inputs'], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    \"\"\"Make prediction\"\"\"
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
        predictions = F.softmax(outputs, dim=1)
    
    return predictions.cpu().numpy()

def output_fn(prediction, content_type):
    \"\"\"Format output\"\"\"
    if content_type == 'application/json':
        return json.dumps({
            'predictions': prediction.tolist()
        })
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
"""

# Example usage
def deploy_to_sagemaker():
    """Example SageMaker deployment"""
    
    deployer = SageMakerDeployer()
    
    # Deploy model
    predictor = deployer.deploy_pytorch_model(
        model_data_url='s3://my-bucket/models/model.tar.gz',
        entry_point_script='inference.py',
        endpoint_name='ml-model-endpoint-v1'
    )
    
    # Setup auto-scaling
    deployer.create_auto_scaling_policy('ml-model-endpoint-v1')
    
    # Test prediction
    test_data = {'inputs': [[1.0, 2.0, 3.0, 4.0]]}
    result = predictor.predict(test_data)
    print(f"Prediction result: {result}")
```

## 4. Serverless Deployment

```python
# AWS Lambda deployment
import json
import boto3
import base64
import torch
import numpy as np

def lambda_handler(event, context):
    """AWS Lambda function for ML inference"""
    
    try:
        # Parse input
        body = json.loads(event['body'])
        input_data = np.array(body['input'])
        
        # Load model (cached after first invocation)
        model = load_model()
        
        # Make prediction
        prediction = predict(model, input_data)
        
        # Return result
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': prediction.tolist(),
                'model_version': 'v1.0'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def load_model():
    """Load model (with caching)"""
    global cached_model
    
    if 'cached_model' not in globals():
        # Download model from S3 or use included model
        s3 = boto3.client('s3')
        s3.download_file('my-bucket', 'models/lambda_model.pth', '/tmp/model.pth')
        
        cached_model = torch.load('/tmp/model.pth', map_location='cpu')
        cached_model.eval()
    
    return cached_model

def predict(model, input_data):
    """Make prediction"""
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.softmax(output, dim=-1)
    
    return prediction.numpy()

# Serverless deployment configuration (serverless.yml)
serverless_config = """
service: ml-inference-api

provider:
  name: aws
  runtime: python3.8
  region: us-east-1
  memorySize: 1024
  timeout: 30
  environment:
    MODEL_BUCKET: my-ml-models-bucket

functions:
  predict:
    handler: handler.lambda_handler
    events:
      - http:
          path: predict
          method: post
          cors: true
    layers:
      - arn:aws:lambda:us-east-1:XXXXXXXXXX:layer:pytorch:1

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
"""

# Google Cloud Functions deployment
def cloud_function_handler(request):
    """Google Cloud Function for ML inference"""
    
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    if request.method == 'POST':
        try:
            request_json = request.get_json()
            input_data = np.array(request_json['input'])
            
            # Load model
            model = load_cloud_model()
            
            # Predict
            prediction = predict(model, input_data)
            
            response = {
                'prediction': prediction.tolist(),
                'status': 'success'
            }
            
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            }
            
            return (json.dumps(response), 200, headers)
            
        except Exception as e:
            error_response = {
                'error': str(e),
                'status': 'error'
            }
            
            return (json.dumps(error_response), 500)

def load_cloud_model():
    """Load model in Cloud Function"""
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket('my-ml-models')
    blob = bucket.blob('models/cloud_model.pth')
    
    blob.download_to_filename('/tmp/model.pth')
    model = torch.load('/tmp/model.pth', map_location='cpu')
    model.eval()
    
    return model
```

## 🎯 Deployment Strategy Decision Matrix

| Use Case | Pattern | Framework | Pros | Cons |
|----------|---------|-----------|------|------|
| Real-time Web App | Online Serving | Flask + Gunicorn | Simple, flexible | Manual scaling |
| High Traffic API | Online Serving | TensorFlow Serving | Auto-scaling, robust | Complex setup |
| Data Processing | Batch Processing | Kubernetes Jobs | High throughput | Not real-time |
| Mobile App | Edge Deployment | ONNX Runtime | Offline capable | Limited models |
| Microservices | Serverless | AWS Lambda | Auto-scaling, cost-effective | Cold starts |
| Enterprise | Kubernetes | K8s + Istio | Full control, scalable | Complex management |

## 📊 Performance Monitoring

```python
import time
import logging
from functools import wraps
import psutil
import GPUtil

class DeploymentMonitor:
    """Monitor deployment performance and health"""
    
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_failed': 0,
            'response_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': []
        }
    
    def monitor_request(self, func):
        """Decorator to monitor request performance"""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                self.metrics['requests_total'] += 1
                
                response_time = time.time() - start_time
                self.metrics['response_times'].append(response_time)
                
                # Keep only recent response times
                if len(self.metrics['response_times']) > 1000:
                    self.metrics['response_times'] = self.metrics['response_times'][-1000:]
                
                return result
                
            except Exception as e:
                self.metrics['requests_failed'] += 1
                raise
        
        return wrapper
    
    def log_system_metrics(self):
        """Log system resource usage"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory_percent)
        
        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
                self.metrics['gpu_usage'].append(gpu_usage)
        except:
            pass
        
        # Keep only recent metrics
        for key in ['cpu_usage', 'memory_usage', 'gpu_usage']:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def get_performance_summary(self):
        """Get performance summary"""
        
        if not self.metrics['response_times']:
            return {'error': 'No requests recorded'}
        
        response_times = self.metrics['response_times']
        
        summary = {
            'requests_total': self.metrics['requests_total'],
            'requests_failed': self.metrics['requests_failed'],
            'success_rate': (self.metrics['requests_total'] - self.metrics['requests_failed']) / self.metrics['requests_total'],
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'qps': len(response_times) / (max(response_times) - min(response_times)) if len(response_times) > 1 else 0
        }
        
        if self.metrics['cpu_usage']:
            summary['avg_cpu_usage'] = np.mean(self.metrics['cpu_usage'])
            summary['avg_memory_usage'] = np.mean(self.metrics['memory_usage'])
        
        if self.metrics['gpu_usage']:
            summary['avg_gpu_usage'] = np.mean(self.metrics['gpu_usage'])
        
        return summary

# Usage with Flask app
monitor = DeploymentMonitor()

@app.route('/predict', methods=['POST'])
@monitor.monitor_request
def predict():
    # Your prediction logic here
    pass

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Metrics endpoint for monitoring"""
    monitor.log_system_metrics()
    summary = monitor.get_performance_summary()
    return jsonify(summary)
```

## 🚀 Production Readiness Checklist

### Infrastructure
- [ ] Load balancing configured
- [ ] Auto-scaling policies set
- [ ] Health checks implemented
- [ ] Monitoring and alerting active
- [ ] Backup and recovery procedures

### Security
- [ ] Authentication and authorization
- [ ] Input validation and sanitization
- [ ] Rate limiting implemented
- [ ] SSL/TLS encryption
- [ ] Security vulnerability scanning

### Performance
- [ ] Latency requirements met
- [ ] Throughput targets achieved
- [ ] Resource utilization optimized
- [ ] Caching strategies implemented
- [ ] Database performance tuned

### Reliability
- [ ] Error handling and graceful degradation
- [ ] Circuit breaker patterns implemented
- [ ] Timeout configurations set
- [ ] Retry mechanisms with backoff
- [ ] Chaos engineering tests passed

### Observability
- [ ] Comprehensive logging
- [ ] Metrics collection and dashboards
- [ ] Distributed tracing
- [ ] Alerting rules configured
- [ ] Runbook documentation

## 💡 Key Takeaways

1. **Choose the Right Pattern**: Match deployment pattern to your use case requirements
2. **Start Simple, Scale Smart**: Begin with simple deployments and add complexity as needed
3. **Monitor Everything**: Comprehensive monitoring is crucial for production success
4. **Plan for Failure**: Build resilient systems that gracefully handle failures
5. **Automate Operations**: Reduce manual work through automation and infrastructure as code

Remember: The best deployment is the one that reliably serves your users while meeting your business requirements!

Ready to deploy your AI to the world? Start with a simple REST API and grow from there! 🚀
