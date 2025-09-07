# Model Optimization: Making AI Fast and Efficient

Transform your research models into production-ready systems that serve millions of users. Learn to optimize for speed, memory, and accuracy while maintaining the performance your business needs.

## 🎯 What You'll Master

- **Model Compression**: Reduce size by 10x without losing accuracy
- **Hardware Acceleration**: Leverage GPUs, TPUs, and specialized chips
- **Memory Optimization**: Run large models on limited hardware
- **Inference Optimization**: Achieve millisecond response times

## 📚 The Optimization Journey

### Understanding the Problem

**Real-World Challenge:**
```
Research Model: 150MB, 500ms inference, 99.2% accuracy
Production Need: <10MB, <50ms inference, >99% accuracy
```

**The Optimization Tradeoff:**
```
Performance ↔ Accuracy ↔ Resource Usage
```

Think of optimization like packing for a trip - you want to bring everything important while fitting in your suitcase!

## 1. Model Compression Techniques

### Pruning: Removing Unnecessary Connections

**Concept:** Remove neurons and connections that contribute little to the final prediction.

**Analogy:** Like trimming a tree - remove dead branches while keeping the healthy structure.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Magnitude-based pruning
def magnitude_pruning(model, pruning_ratio=0.3):
    """Remove weights with smallest magnitudes"""
    
    # Apply pruning to each layer
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
    
    return model

# Structured pruning (remove entire channels/filters)
def structured_pruning(model, pruning_ratio=0.3):
    """Remove entire filters/channels"""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune entire filters (structured)
            prune.ln_structured(
                module, 
                name='weight', 
                amount=pruning_ratio, 
                n=2, 
                dim=0  # Prune output channels
            )
    
    return model

# Gradual pruning during training
class GradualPruner:
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.9, 
                 pruning_frequency=100):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.step = 0
    
    def prune_step(self):
        """Apply gradual pruning during training"""
        if self.step % self.pruning_frequency == 0:
            # Calculate current sparsity
            current_sparsity = self.initial_sparsity + \
                (self.final_sparsity - self.initial_sparsity) * \
                (self.step / 10000)  # Over 10k steps
            
            # Apply pruning
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(
                        module, 
                        name='weight', 
                        amount=current_sparsity
                    )
        
        self.step += 1

# Example usage
model = OptimizedCNN()

# Method 1: One-shot pruning
pruned_model = magnitude_pruning(model.copy(), pruning_ratio=0.5)

# Method 2: Structured pruning
structured_model = structured_pruning(model.copy(), pruning_ratio=0.3)

# Method 3: Gradual pruning
pruner = GradualPruner(model, final_sparsity=0.8)

# During training loop:
def train_with_pruning(model, dataloader, optimizer, pruner):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Apply gradual pruning
        pruner.prune_step()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

# Measure model size reduction
def calculate_sparsity(model):
    """Calculate the sparsity of a pruned model"""
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity

print(f"Model sparsity: {calculate_sparsity(pruned_model):.2%}")
```

### Quantization: Reducing Precision

**Concept:** Use fewer bits to represent weights and activations (e.g., 8-bit instead of 32-bit).

**Analogy:** Like using approximate measurements in cooking - "about a cup" instead of "1.0347 cups."

```python
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub

class QuantizedModel(nn.Module):
    def __init__(self, original_model):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()
        self.model = original_model
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# Post-training quantization (PTQ)
def post_training_quantization(model, calibration_loader):
    """Quantize model after training"""
    
    # Prepare model for quantization
    model.eval()
    quantized_model = QuantizedModel(model)
    
    # Set quantization config
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for data, _ in calibration_loader:
            quantized_model(data)
    
    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    
    return quantized_model

# Quantization-aware training (QAT)
def quantization_aware_training(model, train_loader, num_epochs=5):
    """Train model with quantization simulation"""
    
    # Prepare model for QAT
    model.train()
    quantized_model = QuantizedModel(model)
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare for QAT
    torch.quantization.prepare_qat(quantized_model, inplace=True)
    
    # Training loop with quantization
    optimizer = torch.optim.Adam(quantized_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = quantized_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Convert to quantized model
    quantized_model.eval()
    torch.quantization.convert(quantized_model, inplace=True)
    
    return quantized_model

# Dynamic quantization (simple and effective)
def dynamic_quantization(model):
    """Apply dynamic quantization to linear layers"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Layers to quantize
        dtype=torch.qint8  # Target dtype
    )
    return quantized_model

# Compare model sizes
def compare_model_sizes(original_model, quantized_model):
    """Compare file sizes of models"""
    
    # Save models
    torch.save(original_model.state_dict(), 'original_model.pth')
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')
    
    # Check file sizes
    import os
    original_size = os.path.getsize('original_model.pth') / (1024**2)  # MB
    quantized_size = os.path.getsize('quantized_model.pth') / (1024**2)  # MB
    
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Clean up
    os.remove('original_model.pth')
    os.remove('quantized_model.pth')

# Example usage
original_model = OptimizedCNN()

# Method 1: Dynamic quantization (easiest)
dynamic_quantized = dynamic_quantization(original_model)

# Method 2: Post-training quantization
# ptq_model = post_training_quantization(original_model, calibration_loader)

# Method 3: Quantization-aware training
# qat_model = quantization_aware_training(original_model, train_loader)

compare_model_sizes(original_model, dynamic_quantized)
```

### Knowledge Distillation: Learning from Teacher Models

**Concept:** Train a smaller "student" model to mimic a larger "teacher" model.

**Analogy:** Like learning from an expert teacher who guides you to the right answers.

```python
class StudentModel(nn.Module):
    """Smaller, faster student model"""
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    def __init__(self, temperature=4, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss (ground truth)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss (teacher knowledge)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss

def distill_knowledge(teacher_model, student_model, train_loader, 
                     num_epochs=10, device='cuda'):
    """Train student model using teacher's knowledge"""
    
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distill_loss = DistillationLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Get student predictions
            student_logits = student_model(data)
            
            # Calculate distillation loss
            loss = distill_loss(student_logits, teacher_logits, target)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} average loss: {total_loss/len(train_loader):.4f}')
    
    return student_model

# Advanced: Feature-based distillation
class FeatureDistillationLoss(nn.Module):
    """Distill intermediate features as well as final outputs"""
    def __init__(self, feature_weight=0.3, output_weight=0.7):
        super(FeatureDistillationLoss, self).__init__()
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.mse_loss = nn.MSELoss()
        self.distill_loss = DistillationLoss()
    
    def forward(self, student_features, teacher_features, 
                student_logits, teacher_logits, labels):
        # Feature matching loss
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Output distillation loss
        output_loss = self.distill_loss(student_logits, teacher_logits, labels)
        
        total_loss = self.feature_weight * feature_loss + \
                    self.output_weight * output_loss
        return total_loss

# Model comparison
def compare_models(teacher, student, test_loader, device='cuda'):
    """Compare teacher and student model performance"""
    
    def evaluate_model(model, loader):
        model.eval()
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_inference_time = np.mean(inference_times)
        return accuracy, avg_inference_time
    
    teacher_acc, teacher_time = evaluate_model(teacher, test_loader)
    student_acc, student_time = evaluate_model(student, test_loader)
    
    print(f"Teacher - Accuracy: {teacher_acc:.4f}, Inference: {teacher_time:.4f}s")
    print(f"Student - Accuracy: {student_acc:.4f}, Inference: {student_time:.4f}s")
    print(f"Speedup: {teacher_time/student_time:.2f}x")
    print(f"Accuracy drop: {teacher_acc - student_acc:.4f}")

# Example usage
teacher_model = OptimizedCNN()  # Pre-trained large model
student_model = StudentModel()

# Train student using teacher's knowledge
# trained_student = distill_knowledge(teacher_model, student_model, train_loader)

# Compare performance
# compare_models(teacher_model, trained_student, test_loader)
```

## 2. Hardware Acceleration

### GPU Optimization

```python
import torch.backends.cudnn as cudnn

def optimize_for_gpu(model):
    """Optimize model for GPU inference"""
    
    # Enable cudnn benchmark for consistent input sizes
    cudnn.benchmark = True
    
    # Use half precision (FP16) for faster inference
    model = model.half()
    
    # Compile model for optimal GPU kernels (PyTorch 2.0+)
    model = torch.compile(model)
    
    return model

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, num_epochs=5):
    """Train with automatic mixed precision (AMP)"""
    
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Batch processing for throughput
def batch_inference(model, data_loader, batch_size=32):
    """Optimize inference with larger batches"""
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Process larger batches for better GPU utilization
            predictions = model(batch)
            all_predictions.append(predictions.cpu())
    
    return torch.cat(all_predictions, dim=0)
```

### Model Format Conversion

```python
# Convert to ONNX for cross-platform deployment
def convert_to_onnx(model, input_shape=(1, 3, 224, 224), output_path='model.onnx'):
    """Convert PyTorch model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted to ONNX: {output_path}")

# TensorRT optimization (NVIDIA GPUs)
def optimize_with_tensorrt(onnx_path, output_path='model.trt'):
    """Optimize ONNX model with TensorRT"""
    
    import tensorrt as trt
    
    # Create TensorRT logger and builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Parse ONNX model
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Enable FP16 precision
    config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved: {output_path}")

# TensorFlow Lite conversion (for mobile/edge)
def convert_to_tflite(model, input_shape=(1, 224, 224, 3)):
    """Convert to TensorFlow Lite for mobile deployment"""
    
    # This is a conceptual example - actual conversion depends on framework
    print("Converting to TensorFlow Lite...")
    
    # Typical TFLite optimizations:
    # - Quantization to int8
    # - Pruning for sparse models
    # - Clustering for weight sharing
    
    optimizations = [
        'OPTIMIZE_FOR_SIZE',      # Reduce model size
        'OPTIMIZE_FOR_LATENCY',   # Reduce inference time
    ]
    
    print(f"Applied optimizations: {optimizations}")
    print("Model ready for mobile deployment!")
```

## 3. Memory Optimization

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    """Model with gradient checkpointing to save memory"""
    
    def __init__(self, num_layers=50):
        super(MemoryEfficientModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(num_layers)
        ])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Use checkpointing to save memory during training
        for layer in self.layers:
            # Trade computation for memory
            x = checkpoint(self._layer_forward, layer, x)
        return x
    
    def _layer_forward(self, layer, x):
        return self.activation(layer(x))

# Memory monitoring
def monitor_memory_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Efficient data loading
class MemoryEfficientDataLoader:
    """DataLoader optimized for memory usage"""
    
    def __init__(self, dataset, batch_size=32, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,          # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2,        # Prefetch batches
        )
```

## 4. Real-World Optimization Examples

### Optimization Pipeline

```python
class ModelOptimizer:
    """Complete optimization pipeline"""
    
    def __init__(self, model, target_size_mb=10, target_latency_ms=50):
        self.model = model
        self.target_size_mb = target_size_mb
        self.target_latency_ms = target_latency_ms
        self.optimization_history = []
    
    def optimize(self, train_loader, val_loader):
        """Run complete optimization pipeline"""
        
        print("Starting model optimization...")
        original_metrics = self._evaluate_model(self.model, val_loader)
        print(f"Original model: {original_metrics}")
        
        # Step 1: Pruning
        print("\n1. Applying pruning...")
        pruned_model = self._apply_pruning(self.model, sparsity=0.5)
        pruned_metrics = self._evaluate_model(pruned_model, val_loader)
        print(f"After pruning: {pruned_metrics}")
        
        # Step 2: Quantization
        print("\n2. Applying quantization...")
        quantized_model = dynamic_quantization(pruned_model)
        quantized_metrics = self._evaluate_model(quantized_model, val_loader)
        print(f"After quantization: {quantized_metrics}")
        
        # Step 3: Knowledge distillation (if still too large)
        if quantized_metrics['size_mb'] > self.target_size_mb:
            print("\n3. Applying knowledge distillation...")
            student_model = StudentModel()
            distilled_model = distill_knowledge(
                quantized_model, student_model, train_loader
            )
            final_metrics = self._evaluate_model(distilled_model, val_loader)
            print(f"After distillation: {final_metrics}")
            return distilled_model
        
        return quantized_model
    
    def _evaluate_model(self, model, val_loader):
        """Evaluate model performance and efficiency"""
        
        # Accuracy
        accuracy = self._calculate_accuracy(model, val_loader)
        
        # Model size
        size_mb = self._calculate_model_size(model)
        
        # Inference time
        latency_ms = self._measure_latency(model)
        
        return {
            'accuracy': accuracy,
            'size_mb': size_mb,
            'latency_ms': latency_ms
        }
    
    def _calculate_accuracy(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def _calculate_model_size(self, model):
        # Save model and check file size
        temp_path = 'temp_model.pth'
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024**2)
        os.remove(temp_path)
        return size_mb
    
    def _measure_latency(self, model, num_runs=100):
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            times.append((time.time() - start) * 1000)  # ms
        
        return np.mean(times)
    
    def _apply_pruning(self, model, sparsity=0.3):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        return model

# Usage example
optimizer = ModelOptimizer(
    model=OptimizedCNN(),
    target_size_mb=5,
    target_latency_ms=30
)

# optimized_model = optimizer.optimize(train_loader, val_loader)
```

## 🎯 Optimization Strategies by Use Case

### Mobile Deployment
- **Priority**: Size and battery efficiency
- **Techniques**: Quantization, pruning, MobileNet architectures
- **Target**: <10MB, <100ms on mobile CPU

### Real-time Applications
- **Priority**: Ultra-low latency
- **Techniques**: Model distillation, hardware acceleration
- **Target**: <10ms inference time

### High-throughput Services
- **Priority**: Maximum requests per second
- **Techniques**: Batch processing, GPU optimization
- **Target**: >1000 requests/second

### Edge Computing
- **Priority**: Minimal resource usage
- **Techniques**: Extreme quantization, specialized hardware
- **Target**: Run on microcontrollers

## 📊 Optimization Results Tracking

```python
class OptimizationTracker:
    """Track optimization progress and results"""
    
    def __init__(self):
        self.results = []
    
    def log_result(self, technique, metrics, notes=""):
        """Log optimization results"""
        self.results.append({
            'technique': technique,
            'accuracy': metrics['accuracy'],
            'size_mb': metrics['size_mb'],
            'latency_ms': metrics['latency_ms'],
            'compression_ratio': self._calculate_compression_ratio(metrics),
            'speedup': self._calculate_speedup(metrics),
            'notes': notes
        })
    
    def _calculate_compression_ratio(self, metrics):
        if len(self.results) == 0:
            return 1.0
        original_size = self.results[0]['size_mb']
        return original_size / metrics['size_mb']
    
    def _calculate_speedup(self, metrics):
        if len(self.results) == 0:
            return 1.0
        original_latency = self.results[0]['latency_ms']
        return original_latency / metrics['latency_ms']
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n🎯 Optimization Summary")
        print("=" * 50)
        
        for i, result in enumerate(self.results):
            print(f"\n{i+1}. {result['technique']}")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Size: {result['size_mb']:.2f}MB")
            print(f"   Latency: {result['latency_ms']:.2f}ms")
            print(f"   Compression: {result['compression_ratio']:.2f}x")
            print(f"   Speedup: {result['speedup']:.2f}x")
            if result['notes']:
                print(f"   Notes: {result['notes']}")
```

## 🚀 Production Optimization Checklist

### Before Optimization
- [ ] Baseline performance established
- [ ] Target metrics defined
- [ ] Test data prepared
- [ ] Evaluation pipeline ready

### During Optimization
- [ ] Track accuracy at each step
- [ ] Measure actual inference time
- [ ] Test on target hardware
- [ ] Validate with real data

### After Optimization
- [ ] A/B test against original model
- [ ] Monitor production performance
- [ ] Document optimization process
- [ ] Plan for model updates

## 💡 Key Takeaways

1. **Start Simple**: Begin with the easiest optimizations (quantization, pruning)
2. **Measure Everything**: Track accuracy, size, and speed at each step
3. **Know Your Constraints**: Understand hardware and latency requirements
4. **Test on Target Hardware**: Optimization results vary by platform
5. **Maintain Quality**: Don't sacrifice accuracy for speed without justification

Remember: The best optimization is the one that meets your specific requirements while maintaining the quality your users need!

Ready to make your models lightning fast? Start with dynamic quantization - it's the easiest win! 🚀
