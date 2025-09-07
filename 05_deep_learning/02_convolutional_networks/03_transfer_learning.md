# Transfer Learning: Standing on the Shoulders of Giants

Learn to leverage pre-trained models and achieve state-of-the-art results with minimal training time and data. Transfer learning is your shortcut to building powerful computer vision applications.

## 🎯 What You'll Learn

- Understanding transfer learning and why it works
- How to use pre-trained models effectively
- Fine-tuning strategies for different scenarios
- Building custom applications with transfer learning

## 🤔 What is Transfer Learning?

### The Human Learning Analogy

Imagine you're a skilled basketball player learning to play volleyball. You don't start from scratch - you transfer your knowledge of:

- **Hand-eye coordination**: Still useful for volleyball
- **Team strategy**: Similar concepts apply
- **Athletic conditioning**: Directly transferable

But you need to learn volleyball-specific skills:
- **Different ball handling techniques**: New motor skills
- **Net height and court size**: Adapted spatial awareness
- **Scoring rules**: Sport-specific knowledge

Transfer learning works the same way with neural networks!

### Transfer Learning in CNNs

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

class TransferLearningExample:
    """
    Complete example of transfer learning with different strategies
    """
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pretrained_model(self, model_name='resnet18', strategy='feature_extraction'):
        """
        Load a pre-trained model with different transfer learning strategies
        
        Args:
            model_name: Name of the pre-trained model
            strategy: 'feature_extraction' or 'fine_tuning'
        """
        print(f"Loading {model_name} with {strategy} strategy...")
        
        # Load pre-trained model
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[6].in_features
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_features = model.classifier.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if strategy == 'feature_extraction':
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the final layer
            if model_name == 'resnet18':
                model.fc = nn.Linear(num_features, self.num_classes)
            elif model_name == 'vgg16':
                model.classifier[6] = nn.Linear(num_features, self.num_classes)
            elif model_name == 'densenet121':
                model.classifier = nn.Linear(num_features, self.num_classes)
                
        elif strategy == 'fine_tuning':
            # Replace the final layer (all parameters remain trainable)
            if model_name == 'resnet18':
                model.fc = nn.Linear(num_features, self.num_classes)
            elif model_name == 'vgg16':
                model.classifier[6] = nn.Linear(num_features, self.num_classes)
            elif model_name == 'densenet121':
                model.classifier = nn.Linear(num_features, self.num_classes)
        
        return model.to(self.device)
    
    def get_trainable_params(self, model):
        """Count trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        return total_params, trainable_params

# Example usage
transfer_demo = TransferLearningExample(num_classes=10)

# Compare different strategies
print("=== Feature Extraction Strategy ===")
feature_extractor = transfer_demo.load_pretrained_model('resnet18', 'feature_extraction')
transfer_demo.get_trainable_params(feature_extractor)

print("\n=== Fine-tuning Strategy ===")
fine_tuned = transfer_demo.load_pretrained_model('resnet18', 'fine_tuning')
transfer_demo.get_trainable_params(fine_tuned)
```

## 🎯 Transfer Learning Strategies

### 1. Feature Extraction

Use the pre-trained network as a fixed feature extractor. Only train the final classifier.

**When to use**: Small dataset, similar to pre-training data

```python
class FeatureExtractor:
    """
    Feature extraction approach to transfer learning
    """
    
    def __init__(self, base_model='resnet18', num_classes=10):
        self.model = self._build_feature_extractor(base_model, num_classes)
        
    def _build_feature_extractor(self, base_model, num_classes):
        # Load pre-trained model
        if base_model == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            
            # Freeze all layers
            for param in backbone.parameters():
                param.requires_grad = False
            
            # Replace classifier
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, num_classes)
            
        return backbone
    
    def train_classifier_only(self, train_loader, num_epochs=10):
        """
        Train only the classifier layer
        """
        criterion = nn.CrossEntropyLoss()
        # Only optimize classifier parameters
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.001)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

print("Feature Extraction Benefits:")
print("✅ Fast training (fewer parameters to update)")
print("✅ Less likely to overfit (frozen features are robust)")
print("✅ Requires less computational resources")
print("❌ Limited adaptation to new domain")
```

### 2. Fine-tuning

Start with pre-trained weights and continue training the entire network with a small learning rate.

**When to use**: Larger dataset, domain different from pre-training data

```python
class FineTuner:
    """
    Fine-tuning approach to transfer learning
    """
    
    def __init__(self, base_model='resnet18', num_classes=10):
        self.model = self._build_fine_tuning_model(base_model, num_classes)
        
    def _build_fine_tuning_model(self, base_model, num_classes):
        # Load pre-trained model
        if base_model == 'resnet18':
            model = models.resnet18(pretrained=True)
            
            # Replace classifier
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            
        return model
    
    def train_with_different_lr(self, train_loader, num_epochs=10):
        """
        Fine-tune with different learning rates for different layers
        """
        criterion = nn.CrossEntropyLoss()
        
        # Different learning rates for different parts
        optimizer = torch.optim.Adam([
            {'params': self.model.conv1.parameters(), 'lr': 0.0001},  # Very small LR for early layers
            {'params': self.model.layer1.parameters(), 'lr': 0.0001},
            {'params': self.model.layer2.parameters(), 'lr': 0.0005},
            {'params': self.model.layer3.parameters(), 'lr': 0.0005},
            {'params': self.model.layer4.parameters(), 'lr': 0.001},
            {'params': self.model.fc.parameters(), 'lr': 0.01}  # Largest LR for new classifier
        ])
        
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

print("Fine-tuning Benefits:")
print("✅ Better adaptation to new domain")
print("✅ Can achieve higher accuracy")
print("✅ Leverages all pre-trained knowledge")
print("❌ Slower training (more parameters)")
print("❌ Risk of overfitting with small datasets")
```

### 3. Progressive Unfreezing

Gradually unfreeze layers during training, starting from the top.

```python
class ProgressiveUnfreezer:
    """
    Progressive unfreezing strategy
    """
    
    def __init__(self, model):
        self.model = model
        self.layer_groups = self._create_layer_groups()
    
    def _create_layer_groups(self):
        """Group layers for progressive unfreezing"""
        if isinstance(self.model, models.ResNet):
            return [
                [self.model.fc],                    # Classifier
                [self.model.layer4],               # High-level features
                [self.model.layer3],               # Mid-level features
                [self.model.layer2],               # Low-level features
                [self.model.layer1, self.model.conv1]  # Very low-level features
            ]
        else:
            # Generic approach: reverse order of named children
            children = list(self.model.children())
            return [[child] for child in reversed(children)]
    
    def freeze_all(self):
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_group(self, group_idx):
        """Unfreeze a specific group of layers"""
        if group_idx < len(self.layer_groups):
            for layer in self.layer_groups[group_idx]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze layer group {group_idx}")
    
    def train_progressive(self, train_loader, epochs_per_stage=5):
        """
        Train with progressive unfreezing
        """
        criterion = nn.CrossEntropyLoss()
        
        # Start with everything frozen
        self.freeze_all()
        
        for stage in range(len(self.layer_groups)):
            print(f"\n=== Stage {stage + 1}: Unfreezing group {stage} ===")
            
            # Unfreeze next group
            self.unfreeze_group(stage)
            
            # Create optimizer for currently trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=0.001 * (0.1 ** stage))
            
            # Train for specified epochs
            for epoch in range(epochs_per_stage):
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                print(f'  Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

print("Progressive Unfreezing Benefits:")
print("✅ Stable training (gradual adaptation)")
print("✅ Prevents catastrophic forgetting")
print("✅ Good balance between speed and performance")
```

## 🎯 Choosing the Right Strategy

### Decision Framework

```python
def choose_transfer_strategy(dataset_size, domain_similarity):
    """
    Help choose the right transfer learning strategy
    
    Args:
        dataset_size: 'small' (<1000), 'medium' (1000-10000), 'large' (>10000)
        domain_similarity: 'similar', 'different', 'very_different'
    
    Returns:
        Recommended strategy and explanation
    """
    
    strategies = {
        ('small', 'similar'): {
            'strategy': 'feature_extraction',
            'reason': 'Small dataset + similar domain = risk of overfitting if fine-tuning',
            'tips': ['Use pre-trained features as-is', 'Only train classifier', 'Strong data augmentation']
        },
        
        ('small', 'different'): {
            'strategy': 'feature_extraction',
            'reason': 'Small dataset = high overfitting risk, but features still useful',
            'tips': ['Extract features from multiple layers', 'Use ensemble of classifiers', 'Heavy regularization']
        },
        
        ('medium', 'similar'): {
            'strategy': 'fine_tuning',
            'reason': 'Enough data to fine-tune without overfitting',
            'tips': ['Small learning rate', 'Fine-tune top layers only initially', 'Monitor validation carefully']
        },
        
        ('medium', 'different'): {
            'strategy': 'progressive_unfreezing',
            'reason': 'Need adaptation but want to preserve low-level features',
            'tips': ['Start with feature extraction', 'Gradually unfreeze layers', 'Different LR for different layers']
        },
        
        ('large', 'similar'): {
            'strategy': 'fine_tuning',
            'reason': 'Large dataset allows full fine-tuning',
            'tips': ['Can use higher learning rates', 'Fine-tune entire network', 'May even train from scratch']
        },
        
        ('large', 'different'): {
            'strategy': 'fine_tuning',
            'reason': 'Large dataset + different domain = need full adaptation',
            'tips': ['Higher learning rates for top layers', 'May need to replace more than just classifier', 'Consider training some layers from scratch']
        }
    }
    
    key = (dataset_size, domain_similarity)
    if key in strategies:
        return strategies[key]
    else:
        return {
            'strategy': 'feature_extraction',
            'reason': 'Default safe choice',
            'tips': ['Start simple', 'Experiment with different approaches']
        }

# Examples
scenarios = [
    ('small', 'similar'),
    ('medium', 'different'),
    ('large', 'very_different')
]

for size, similarity in scenarios:
    recommendation = choose_transfer_strategy(size, similarity)
    print(f"\nScenario: {size} dataset, {similarity} domain")
    print(f"Strategy: {recommendation['strategy']}")
    print(f"Reason: {recommendation['reason']}")
    print("Tips:", ', '.join(recommendation['tips']))
```

## 🛠️ Practical Implementation

### Complete Transfer Learning Pipeline

```python
class TransferLearningPipeline:
    """
    Complete pipeline for transfer learning projects
    """
    
    def __init__(self, num_classes, input_size=(224, 224)):
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_data_transforms(self, phase='train'):
        """
        Get appropriate data transformations
        """
        if phase == 'train':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def build_model(self, architecture='resnet18', strategy='fine_tuning'):
        """
        Build transfer learning model
        """
        # Load pre-trained model
        if architecture == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif architecture == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif architecture == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        # Apply strategy
        if strategy == 'feature_extraction':
            for param in model.parameters():
                param.requires_grad = False
            
            # Make final layer trainable
            if architecture.startswith('resnet'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif architecture == 'vgg16':
                for param in model.classifier[6].parameters():
                    param.requires_grad = True
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, num_epochs=25):
        """
        Training loop with validation
        """
        criterion = nn.CrossEntropyLoss()
        
        # Different optimizers based on strategy
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_acc = 0.0
        train_history = {'loss': [], 'acc': []}
        val_history = {'loss': [], 'acc': []}
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            train_history['loss'].append(epoch_loss)
            train_history['acc'].append(epoch_acc)
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
            
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
            
            val_history['loss'].append(val_epoch_loss)
            val_history['acc'].append(val_epoch_acc)
            
            print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Save best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
            
            scheduler.step()
            print()
        
        print(f'Best val Acc: {best_acc:4f}')
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        
        return model, train_history, val_history

# Example usage
print("Transfer Learning Pipeline Example:")
print("1. Define your dataset and number of classes")
print("2. Choose architecture and strategy based on your data")
print("3. Apply appropriate data transformations")
print("4. Train with validation monitoring")
print("5. Evaluate and deploy")
```

## 🎯 Real-World Applications

### Medical Image Classification

```python
def medical_imaging_transfer():
    """
    Example: Transfer learning for medical image classification
    """
    # Scenario: Classify chest X-rays (pneumonia vs normal)
    # Dataset: Small (few thousand images)
    # Domain: Different from ImageNet but medical images have similar structure
    
    strategy = "feature_extraction"  # Safe choice for small medical dataset
    
    # Use DenseNet (good for medical images)
    model = models.densenet121(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # pneumonia vs normal
    )
    
    print("Medical Imaging Setup:")
    print("✅ Feature extraction (safe for small dataset)")
    print("✅ DenseNet backbone (good for fine details)")
    print("✅ Additional dropout (prevent overfitting)")
    print("✅ Data augmentation essential")

def wildlife_classification_transfer():
    """
    Example: Wildlife species classification
    """
    # Scenario: Classify 50 animal species
    # Dataset: Medium (10,000 images)
    # Domain: Similar to ImageNet (natural images)
    
    strategy = "fine_tuning"  # Good balance for medium dataset
    
    # Use ResNet50 (good general-purpose model)
    model = models.resnet50(pretrained=True)
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 50)  # 50 species
    
    # Different learning rates
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.001},
        {'params': model.fc.parameters(), 'lr': 0.01}
    ])
    
    print("Wildlife Classification Setup:")
    print("✅ Fine-tuning (sufficient data)")
    print("✅ ResNet50 (robust architecture)")
    print("✅ Different LR for classifier vs features")
    print("✅ Strong data augmentation recommended")

# Run examples
medical_imaging_transfer()
print()
wildlife_classification_transfer()
```

## 📊 Model Comparison and Selection

```python
def compare_pretrained_models():
    """
    Compare different pre-trained models for transfer learning
    """
    models_info = {
        'resnet18': {
            'parameters': '11.7M',
            'top1_accuracy': '69.8%',
            'pros': ['Fast training', 'Good for small datasets', 'Skip connections'],
            'cons': ['Lower accuracy', 'Less feature diversity']
        },
        'resnet50': {
            'parameters': '25.6M',
            'top1_accuracy': '76.1%',
            'pros': ['Good balance', 'Robust features', 'Wide adoption'],
            'cons': ['More parameters', 'Slower than ResNet18']
        },
        'vgg16': {
            'parameters': '138.4M',
            'top1_accuracy': '71.6%',
            'pros': ['Simple architecture', 'Good for visualization'],
            'cons': ['Many parameters', 'Memory intensive']
        },
        'densenet121': {
            'parameters': '8.0M',
            'top1_accuracy': '74.4%',
            'pros': ['Parameter efficient', 'Good for medical images', 'Feature reuse'],
            'cons': ['Memory intensive during training', 'Complex architecture']
        },
        'efficientnet_b0': {
            'parameters': '5.3M',
            'top1_accuracy': '77.7%',
            'pros': ['Highly efficient', 'Good accuracy', 'Mobile-friendly'],
            'cons': ['Complex training', 'Newer (less tested)']
        }
    }
    
    print("Pre-trained Model Comparison:")
    print("=" * 80)
    
    for model_name, info in models_info.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Parameters: {info['parameters']}")
        print(f"  ImageNet Top-1: {info['top1_accuracy']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
    
    print("\nRecommendations:")
    print("🚀 Start with: ResNet18 or ResNet50")
    print("🎯 For efficiency: EfficientNet or DenseNet")
    print("🔬 For medical images: DenseNet121")
    print("📱 For mobile deployment: EfficientNet-B0")

compare_pretrained_models()
```

## 🚀 Next Steps and Best Practices

### Best Practices Checklist

```python
transfer_learning_checklist = {
    "Data Preparation": [
        "✅ Use ImageNet normalization for pre-trained models",
        "✅ Apply appropriate data augmentation",
        "✅ Ensure consistent input size (224x224 for most models)",
        "✅ Balance your dataset or use weighted loss"
    ],
    
    "Model Selection": [
        "✅ Start with ResNet18/50 for general tasks",
        "✅ Consider EfficientNet for efficiency",
        "✅ Use DenseNet for medical/detailed images",
        "✅ Match model complexity to dataset size"
    ],
    
    "Training Strategy": [
        "✅ Use smaller learning rates for pre-trained layers",
        "✅ Monitor validation loss to prevent overfitting",
        "✅ Use learning rate scheduling",
        "✅ Save best model based on validation performance"
    ],
    
    "Evaluation": [
        "✅ Test on completely separate test set",
        "✅ Analyze per-class performance",
        "✅ Check for bias in predictions",
        "✅ Visualize learned features if possible"
    ]
}

for category, items in transfer_learning_checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")
```

## 💡 Key Takeaways

1. **Transfer learning is not magic** - it works because deep networks learn hierarchical features
2. **Choose strategy based on data size and domain similarity**
3. **Always use appropriate data transformations**
4. **Monitor validation performance carefully**
5. **Start simple, then increase complexity**

Transfer learning has democratized computer vision - now you can build powerful applications with minimal data and training time. The key is understanding when and how to apply each strategy effectively.

## 📝 Quick Check: Test Your Understanding

1. When would you choose feature extraction vs. fine-tuning?
2. Why do we use smaller learning rates for pre-trained layers?
3. How does domain similarity affect your transfer learning strategy?
4. What are the risks of fine-tuning with a small dataset?

Ready to explore advanced CNN techniques and modern innovations? Let's dive into the cutting edge!
