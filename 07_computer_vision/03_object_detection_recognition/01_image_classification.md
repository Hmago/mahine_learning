# Image Classification with Deep Learning 🖼️🧠

## Overview

Image classification is the fundamental task of assigning a label or category to an entire image. It's the foundation for many computer vision applications and the starting point for understanding deep learning in vision.

## Understanding Image Classification

### What is Image Classification?

**Definition:** Given an image, predict which class or category it belongs to from a predefined set of classes.

**Examples:**
- **Medical**: Classify X-rays as normal vs pneumonia
- **Autonomous Driving**: Classify traffic signs (stop, yield, speed limit)
- **E-commerce**: Classify products (clothing, electronics, books)
- **Wildlife**: Classify animal species in camera trap images

### Single-Label vs Multi-Label Classification

#### Single-Label Classification
- **One label per image**: Each image belongs to exactly one class
- **Mutually exclusive**: Classes don't overlap
- **Output**: Probability distribution over classes (softmax)
- **Loss function**: Cross-entropy loss

#### Multi-Label Classification
- **Multiple labels per image**: Image can belong to multiple classes
- **Non-exclusive**: Classes can co-occur
- **Output**: Independent probability for each class (sigmoid)
- **Loss function**: Binary cross-entropy loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single-label classification
class SingleLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        logits = self.backbone(x)
        probabilities = F.softmax(logits, dim=1)  # Sum to 1
        return logits, probabilities

# Multi-label classification
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        logits = self.backbone(x)
        probabilities = torch.sigmoid(logits)  # Independent probabilities
        return logits, probabilities

# Loss functions
def single_label_loss(logits, targets):
    return F.cross_entropy(logits, targets)

def multi_label_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets.float())
```

## Evolution of CNN Architectures

### 1. LeNet-5 (1998) - The Pioneer

**Key Innovations:**
- First successful CNN for image recognition
- Alternating convolution and pooling layers
- Local connectivity and weight sharing

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**Historical Impact:**
- Proved CNNs could work for real applications
- Introduced key concepts still used today
- Limited by computational power of the era

### 2. AlexNet (2012) - The Breakthrough

**Key Innovations:**
- Deep network (8 layers)
- ReLU activation functions
- Dropout regularization
- GPU acceleration
- Data augmentation

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second block
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third block
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth block
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```

**Revolution Impact:**
- Reduced ImageNet error from 26% to 15%
- Sparked the deep learning revolution
- Showed the power of big data + big compute

### 3. VGG (2014) - Deeper and Uniform

**Key Innovations:**
- Much deeper networks (16-19 layers)
- Small 3x3 filters throughout
- Uniform architecture design
- Demonstrated depth importance

```python
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)

# VGG-16 configuration
vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
             512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg16(num_classes=1000, batch_norm=False):
    features = make_vgg_layers(vgg16_cfg, batch_norm=batch_norm)
    model = VGG(features, num_classes=num_classes)
    return model
```

**Key Insights:**
- Small filters can be as effective as large ones
- Depth matters more than filter size
- Uniform design simplifies understanding

### 4. ResNet (2015) - Solving the Depth Problem

**The Vanishing Gradient Problem:**
- Very deep networks became harder to train
- Gradients vanish in backpropagation
- Deeper networks performed worse than shallow ones

**Residual Learning Solution:**
- Instead of learning H(x), learn F(x) = H(x) - x
- Add skip connections: H(x) = F(x) + x
- Makes optimization much easier

```python
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # Both conv1 and downsample layers downsample when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
```

**Revolutionary Impact:**
- Enabled training of 100+ layer networks
- Consistently improved with depth
- Foundation for many subsequent architectures

### 5. EfficientNet (2019) - Compound Scaling

**Key Innovation:** Scale depth, width, and resolution together using a compound coefficient.

**Scaling Rules:**
- **Depth**: φ^α (number of layers)
- **Width**: φ^β (number of channels)
- **Resolution**: φ^γ (input image size)
- **Constraint**: α·β²·γ² ≈ 2, α ≥ 1, β ≥ 1, γ ≥ 1

```python
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                     kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, se_channels)
        else:
            self.se = nn.Identity()
        
        # Point-wise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        result = self.expand_conv(x)
        result = self.depthwise_conv(result)
        result = self.se(result)
        result = self.pointwise_conv(result)
        
        if self.use_residual:
            result = result + x
        
        return result

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)
```

## Transfer Learning Strategies

### 1. Feature Extraction (Frozen Backbone)

**When to Use:**
- Small dataset (< 10k images)
- Similar domain to pre-training data
- Limited computational resources

```python
def create_feature_extractor(backbone_name, num_classes, freeze_backbone=True):
    """
    Create feature extraction model with frozen backbone
    """
    if backbone_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        
        if freeze_backbone:
            # Freeze all backbone parameters
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    return model

# Training setup for feature extraction
def setup_feature_extraction_training(model, learning_rate=0.001):
    """
    Setup optimizer for feature extraction (only train classifier)
    """
    # Only parameters that require gradients
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    
    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    
    return optimizer
```

### 2. Fine-tuning (Unfrozen Layers)

**When to Use:**
- Medium to large dataset (> 10k images)
- Different domain from pre-training
- Need to adapt low-level features

```python
def create_fine_tuning_model(backbone_name, num_classes, unfreeze_layers=2):
    """
    Create fine-tuning model with selective layer unfreezing
    """
    if backbone_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        modules = list(model.children())
        for module in modules[:-unfreeze_layers]:
            for param in module.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    return model

def setup_fine_tuning_optimizer(model, backbone_lr=1e-4, classifier_lr=1e-3):
    """
    Different learning rates for backbone and classifier
    """
    backbone_params = []
    classifier_params = []
    
    # Separate parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ])
    
    return optimizer
```

### 3. Progressive Unfreezing

**Strategy:** Gradually unfreeze layers during training.

```python
class ProgressiveUnfreezing:
    def __init__(self, model, unfreeze_schedule):
        """
        Progressive unfreezing strategy
        
        Args:
            model: The model to unfreeze
            unfreeze_schedule: Dict mapping epoch to layers to unfreeze
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.frozen_layers = list(model.children())[:-1]  # All except classifier
    
    def unfreeze_layers(self, epoch):
        """Unfreeze layers based on schedule"""
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            
            for layer_idx in layers_to_unfreeze:
                if layer_idx < len(self.frozen_layers):
                    for param in self.frozen_layers[layer_idx].parameters():
                        param.requires_grad = True
                    
                    print(f"Unfroze layer {layer_idx} at epoch {epoch}")

# Usage example
def train_with_progressive_unfreezing(model, train_loader, val_loader, num_epochs):
    """
    Training with progressive unfreezing
    """
    # Define unfreezing schedule
    unfreeze_schedule = {
        5: [3],    # Unfreeze layer 3 at epoch 5
        10: [2],   # Unfreeze layer 2 at epoch 10
        15: [1],   # Unfreeze layer 1 at epoch 15
    }
    
    unfreezer = ProgressiveUnfreezing(model, unfreeze_schedule)
    
    for epoch in range(num_epochs):
        # Check for unfreezing
        unfreezer.unfreeze_layers(epoch)
        
        # Adjust optimizer if needed
        if epoch in unfreeze_schedule:
            optimizer = setup_fine_tuning_optimizer(model)
        
        # Regular training loop
        train_epoch(model, train_loader, optimizer, criterion)
        validate_epoch(model, val_loader, criterion)
```

## Data Augmentation Strategies

### 1. Basic Augmentations

```python
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment

def get_basic_augmentations(image_size=224, is_training=True):
    """
    Basic data augmentation pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.2)),  # Slightly larger
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform
```

### 2. Advanced Augmentations

```python
def get_advanced_augmentations(image_size=224):
    """
    Advanced augmentation techniques
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.2)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        
        # AutoAugment
        autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
        
        # Advanced transforms
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Mixup augmentation
def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss calculation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix augmentation
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation
    """
    lam = np.random.beta(alpha, alpha)
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Generate random bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam
```

## Training Best Practices

### 1. Learning Rate Scheduling

```python
def get_scheduler(optimizer, scheduler_type='cosine', num_epochs=100):
    """
    Learning rate schedulers
    """
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    elif scheduler_type == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=5, 
                                                   num_training_steps=num_epochs)
    
    return scheduler

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Cosine schedule with warmup
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 2. Model Training Loop

```python
def train_classification_model(model, train_loader, val_loader, num_epochs, device):
    """
    Complete training loop for image classification
    """
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_scheduler(optimizer, 'cosine', num_epochs)
    
    # Metrics tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Apply mixup with probability 0.5
            if np.random.random() > 0.5:
                data, targets_a, targets_b, lam = mixup_data(data, targets)
                
                # Forward pass
                outputs = model(data)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Regular forward pass
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_acc += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_acc += predicted.eq(targets).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Logging
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 60)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
```

## Model Evaluation

### 1. Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

def evaluate_classification_model(model, test_loader, class_names, device):
    """
    Comprehensive model evaluation
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC AUC for multi-class
    if len(class_names) > 2:
        # One-vs-rest AUC
        auc_scores = []
        for i in range(len(class_names)):
            binary_targets = (all_targets == i).astype(int)
            binary_probs = all_probabilities[:, i]
            auc = roc_auc_score(binary_targets, binary_probs)
            auc_scores.append(auc)
            print(f"AUC for {class_names[i]}: {auc:.4f}")
        
        print(f"Macro-averaged AUC: {np.mean(auc_scores):.4f}")
    
    # Accuracy
    accuracy = np.mean(all_predictions == all_targets)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy
    }
```

### 2. Error Analysis

```python
def analyze_classification_errors(model, test_loader, class_names, device, num_examples=10):
    """
    Analyze misclassified examples
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            # Find misclassified examples
            incorrect = (predictions != targets)
            
            for i in range(data.size(0)):
                if incorrect[i]:
                    errors.append({
                        'image': data[i].cpu(),
                        'true_label': class_names[targets[i].item()],
                        'predicted_label': class_names[predictions[i].item()],
                        'confidence': probabilities[i][predictions[i]].item(),
                        'true_confidence': probabilities[i][targets[i]].item()
                    })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Visualize top errors
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(min(num_examples, len(errors))):
        error = errors[i]
        
        # Denormalize image for display
        img = error['image'].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {error['true_label']}\n"
                         f"Pred: {error['predicted_label']}\n"
                         f"Conf: {error['confidence']:.3f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return errors[:num_examples]
```

## Real-World Applications

### 1. Medical Image Classification

```python
class MedicalImageClassifier(nn.Module):
    """
    Specialized classifier for medical images
    """
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        
        # Use pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Medical-specific layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Attention mechanism for interpretability
        self.attention = nn.Sequential(
            nn.Conv2d(num_features, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention=False):
        # Extract features
        features = self.backbone(x)  # This will be feature maps before global pooling
        
        # Generate attention map
        attention_map = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_map
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(attended_features, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        if return_attention:
            return logits, attention_map
        return logits

def train_medical_classifier(train_data, val_data, num_classes):
    """
    Training pipeline specifically for medical images
    """
    model = MedicalImageClassifier(num_classes)
    
    # Medical-specific augmentations (conservative)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Conservative flipping
        transforms.RandomRotation(5),  # Small rotations
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class-weighted loss for imbalanced medical data
    class_weights = compute_class_weights(train_data)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Conservative learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    return train_classification_model(model, train_data, val_data, 50, device)
```

### 2. Product Classification for E-commerce

```python
class ProductClassifier(nn.Module):
    """
    Multi-level product classification (category, subcategory, product type)
    """
    def __init__(self, num_categories, num_subcategories, num_products):
        super().__init__()
        
        # Shared backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Multiple classification heads
        self.category_classifier = nn.Linear(num_features, num_categories)
        self.subcategory_classifier = nn.Linear(num_features, num_subcategories)
        self.product_classifier = nn.Linear(num_features, num_products)
    
    def forward(self, x):
        features = self.backbone(x)
        
        category_logits = self.category_classifier(features)
        subcategory_logits = self.subcategory_classifier(features)
        product_logits = self.product_classifier(features)
        
        return {
            'category': category_logits,
            'subcategory': subcategory_logits,
            'product': product_logits
        }

def hierarchical_loss(outputs, targets, weights=[1.0, 0.8, 0.6]):
    """
    Hierarchical loss for multi-level classification
    """
    total_loss = 0
    
    for i, (level, weight) in enumerate(zip(['category', 'subcategory', 'product'], weights)):
        if level in outputs and level in targets:
            loss = F.cross_entropy(outputs[level], targets[level])
            total_loss += weight * loss
    
    return total_loss
```

## Next Steps

This foundation in image classification prepares you for:

1. **Object Detection**: Localizing and classifying multiple objects
2. **Semantic Segmentation**: Pixel-level classification
3. **Instance Segmentation**: Separating individual object instances
4. **Advanced Architectures**: Vision transformers, neural architecture search
5. **Production Deployment**: Model optimization and serving

Image classification skills transfer directly to these more complex tasks, making it an essential foundation for computer vision expertise.
