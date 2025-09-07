# Advanced CNN Techniques: Modern Innovations

Explore cutting-edge CNN techniques that power today's state-of-the-art computer vision systems. From attention mechanisms to neural architecture search, learn the innovations that are shaping the future of AI.

## 🎯 What You'll Learn

- Attention mechanisms in computer vision
- Modern architectural innovations
- Optimization techniques for better training
- Mobile and efficient CNN designs
- Latest research developments

## 🔍 Attention Mechanisms in CNNs

### Understanding Attention

Think of attention like a spotlight on a stage. While the entire stage is lit, the spotlight draws your focus to the most important performer. Similarly, attention mechanisms help CNNs focus on the most relevant parts of an image.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM)
    Focuses on 'what' is meaningful in the feature maps
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        # Global pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling operations
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply global pooling
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM)
    Focuses on 'where' is meaningful in the feature maps
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        return x * attention

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines channel and spatial attention
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x

# Example usage
def demonstrate_attention():
    """
    Demonstrate how attention mechanisms work
    """
    # Create sample feature maps
    batch_size, channels, height, width = 1, 64, 32, 32
    feature_maps = torch.randn(batch_size, channels, height, width)
    
    # Apply different attention mechanisms
    cbam = CBAM(channels)
    
    print("Input shape:", feature_maps.shape)
    
    # Original features
    original_energy = torch.sum(feature_maps**2)
    print(f"Original feature energy: {original_energy:.2f}")
    
    # After attention
    attended_features = cbam(feature_maps)
    attended_energy = torch.sum(attended_features**2)
    print(f"Attended feature energy: {attended_energy:.2f}")
    
    # Attention weights
    with torch.no_grad():
        channel_weights = cbam.channel_attention(feature_maps)
        spatial_weights = cbam.spatial_attention(channel_weights)
    
    print(f"Channel attention range: [{channel_weights.min():.3f}, {channel_weights.max():.3f}]")
    print(f"Spatial attention range: [{spatial_weights.min():.3f}, {spatial_weights.max():.3f}]")

demonstrate_attention()
```

### Self-Attention for Vision

```python
class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature maps
    Allows each position to attend to all other positions
    """
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # Calculate attention scores
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        out = self.gamma * self.out(out) + x
        
        return out, attention

# Example: ResNet with Self-Attention
class AttentionResBlock(nn.Module):
    """
    ResNet block enhanced with self-attention
    """
    
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(AttentionResBlock, self).__init__()
        
        # Standard ResNet block
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)
    
    def forward(self, x):
        identity = x
        
        # Standard convolution path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.skip(identity)
        out = F.relu(out)
        
        # Apply attention if enabled
        if self.use_attention:
            out, attention_map = self.attention(out)
            return out, attention_map
        
        return out

print("Self-Attention Benefits:")
print("✅ Long-range dependencies")
print("✅ Adaptive receptive field")
print("✅ Interpretable attention maps")
print("❌ Quadratic complexity in spatial dimensions")
```

## 🏗️ Modern Architectural Innovations

### EfficientNet: Scaling CNNs Systematically

```python
class EfficientNetBlock(nn.Module):
    """
    MobileNetV3-style inverted residual block used in EfficientNet
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(EfficientNetBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            ])
        
        # Point-wise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

print("EfficientNet Innovations:")
print("1. Compound scaling: width × depth × resolution")
print("2. Neural architecture search for optimal structure")
print("3. Squeeze-and-excitation for channel attention")
print("4. Swish/SiLU activation function")
```

### Vision Transformer Integration

```python
class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings for Vision Transformer
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch extraction using convolution
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        return x

class HybridCNNTransformer(nn.Module):
    """
    Hybrid model combining CNN feature extraction with Transformer processing
    """
    
    def __init__(self, num_classes=1000, embed_dim=768):
        super(HybridCNNTransformer, self).__init__()
        
        # CNN backbone for feature extraction
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            # ResNet-style blocks
            self._make_layer(64, 128, 2, stride=1),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Project CNN features to transformer dimension
        self.cnn_projection = nn.Conv2d(512, embed_dim, 1)
        
        # Transformer layers
        self.patch_embedding = PatchEmbedding(img_size=28, patch_size=4, 
                                           in_channels=embed_dim, embed_dim=embed_dim)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(AttentionResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(AttentionResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Project to transformer dimension
        projected_features = self.cnn_projection(cnn_features)
        
        # Convert to patches for transformer
        patches = self.patch_embedding(projected_features)
        
        # Global average pooling
        global_features = patches.mean(dim=1)
        
        # Classification
        output = self.classifier(global_features)
        
        return output

print("Hybrid CNN-Transformer Benefits:")
print("✅ CNN inductive biases for local patterns")
print("✅ Transformer global attention")
print("✅ Best of both architectures")
print("✅ Good for complex visual reasoning")
```

## ⚡ Optimization Techniques

### Advanced Training Strategies

```python
class MixUp:
    """
    MixUp augmentation: mix two images and their labels
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        # Mix labels
        y_a, y_b = labels, labels[index]
        
        return mixed_images, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CutMix:
    """
    CutMix augmentation: cut and paste patches between images
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = labels, labels[index]
        
        # Generate random bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply cutmix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, y_a, y_b, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# Advanced learning rate scheduling
class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                               (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2

print("Advanced Training Techniques:")
print("🎯 MixUp: Reduces overfitting, improves generalization")
print("✂️ CutMix: Localization and classification jointly")
print("📈 Cosine Annealing: Smooth learning rate cycles")
print("🔄 Warm Restarts: Escapes local minima")
```

### Knowledge Distillation

```python
class KnowledgeDistillation:
    """
    Transfer knowledge from a large teacher model to a smaller student
    """
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        """
        Compute combined distillation and classification loss
        """
        # Hard loss (classification)
        hard_loss = F.cross_entropy(student_outputs, labels)
        
        # Soft loss (distillation)
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + \
                    self.alpha * (self.temperature ** 2) * soft_loss
        
        return total_loss, hard_loss, soft_loss
    
    def train_student(self, student, train_loader, num_epochs=10):
        """
        Train student model with knowledge distillation
        """
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            student.train()
            total_loss = 0
            
            for inputs, labels in train_loader:
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # Get student predictions
                student_outputs = student(inputs)
                
                # Compute distillation loss
                loss, hard_loss, soft_loss = self.distillation_loss(
                    student_outputs, teacher_outputs, labels
                )
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

print("Knowledge Distillation Benefits:")
print("🎓 Transfer knowledge from large to small models")
print("📱 Enable deployment on resource-constrained devices")
print("⚡ Maintain performance with fewer parameters")
print("🔄 Can be combined with other compression techniques")
```

## 📱 Mobile and Efficient CNNs

### MobileNet Architecture

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution used in MobileNet
    Reduces computation by factoring standard convolution
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution (one filter per input channel)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1 conv to combine features)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV2Block(nn.Module):
    """
    MobileNetV2 inverted residual block
    """
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MobileNetV2Block, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Linear bottleneck
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

def calculate_efficiency_metrics(model, input_size=(1, 3, 224, 224)):
    """
    Calculate model efficiency metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate FLOPs (simplified)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        
        # Use hooks to count operations (simplified approach)
        flops = 0
        def count_flops(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                # Convolution FLOPs
                batch_size = input[0].size(0)
                output_dims = output.size()[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
                
                active_elements_count = batch_size * int(np.prod(output_dims))
                overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
                
                flops += overall_conv_flops
        
        # Register hooks
        handles = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(count_flops))
        
        # Forward pass
        _ = model(dummy_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    return {
        'parameters': total_params,
        'flops': flops,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }

# Compare different architectures
print("Mobile CNN Efficiency Comparison:")
print("=" * 50)

architectures = {
    'MobileNetV2': lambda: models.mobilenet_v2(pretrained=False),
    'EfficientNet-B0': lambda: models.efficientnet_b0(pretrained=False),
    'ResNet18': lambda: models.resnet18(pretrained=False),
}

for name, create_model in architectures.items():
    model = create_model()
    metrics = calculate_efficiency_metrics(model)
    
    print(f"\n{name}:")
    print(f"  Parameters: {metrics['parameters']:,}")
    print(f"  FLOPs: {metrics['flops']:,}")
    print(f"  Model Size: {metrics['model_size_mb']:.1f} MB")
```

## 🔬 Latest Research Developments

### Neural Architecture Search (NAS)

```python
class NASSearchSpace:
    """
    Simplified Neural Architecture Search space
    """
    
    def __init__(self):
        self.operations = [
            'conv_3x3',
            'conv_5x5',
            'dwise_conv_3x3',
            'dwise_conv_5x5',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect'
        ]
        
        self.channels = [16, 24, 32, 64, 96, 128, 160, 320]
    
    def sample_architecture(self, num_blocks=6):
        """
        Sample a random architecture from the search space
        """
        architecture = []
        
        for i in range(num_blocks):
            block = {
                'operation': np.random.choice(self.operations),
                'channels': np.random.choice(self.channels),
                'kernel_size': np.random.choice([3, 5]),
                'stride': np.random.choice([1, 2]) if i % 2 == 0 else 1
            }
            architecture.append(block)
        
        return architecture
    
    def build_model(self, architecture, num_classes=10):
        """
        Build a model from architecture description
        """
        layers = []
        in_channels = 3
        
        for block in architecture:
            if block['operation'] == 'conv_3x3':
                layers.append(nn.Conv2d(in_channels, block['channels'], 3, 
                                      block['stride'], 1))
                layers.append(nn.BatchNorm2d(block['channels']))
                layers.append(nn.ReLU(inplace=True))
                in_channels = block['channels']
            
            elif block['operation'] == 'dwise_conv_3x3':
                layers.append(DepthwiseSeparableConv(in_channels, block['channels'], 
                                                   3, block['stride']))
                in_channels = block['channels']
            
            # Add more operation types as needed
        
        # Global pooling and classifier
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        ])
        
        return nn.Sequential(*layers)

# Example NAS workflow
nas = NASSearchSpace()
for i in range(3):
    arch = nas.sample_architecture()
    model = nas.build_model(arch)
    metrics = calculate_efficiency_metrics(model)
    
    print(f"\nArchitecture {i+1}:")
    print(f"  Operations: {[block['operation'] for block in arch]}")
    print(f"  Parameters: {metrics['parameters']:,}")
    print(f"  Model Size: {metrics['model_size_mb']:.1f} MB")
```

### Self-Supervised Learning

```python
class SimCLR:
    """
    Simplified SimCLR for self-supervised representation learning
    """
    
    def __init__(self, backbone, projection_dim=128):
        self.backbone = backbone
        
        # Remove final classification layer
        if hasattr(backbone, 'fc'):
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
    
    def contrastive_loss(self, features, temperature=0.1):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        """
        batch_size = features.size(0) // 2
        
        # Split into two views
        z1 = features[:batch_size]
        z2 = features[batch_size:]
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temperature
        
        # Create labels (positive pairs)
        labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                           torch.arange(0, batch_size)], dim=0)
        
        # Mask to remove self-similarity
        mask = torch.eye(2*batch_size, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

print("Self-Supervised Learning Benefits:")
print("🎯 Learn representations without labeled data")
print("📊 Better transfer learning performance")
print("🔄 Leverages large amounts of unlabeled data")
print("🎨 Learns robust visual representations")
```

## 🚀 Putting It All Together

### Modern CNN Best Practices

```python
class ModernCNN(nn.Module):
    """
    State-of-the-art CNN incorporating modern techniques
    """
    
    def __init__(self, num_classes=1000, use_attention=True, use_se=True):
        super(ModernCNN, self).__init__()
        
        # Efficient backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Replace classifier
        feature_dim = self.backbone.classifier[1].in_features
        
        if use_attention:
            # Add attention module
            self.attention = CBAM(feature_dim)
        
        # Modern classifier with dropout and label smoothing-friendly design
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.use_attention = use_attention
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone.features(x)
        
        # Apply attention
        if self.use_attention:
            features = self.attention(features)
        
        # Classification
        output = self.classifier(features)
        
        return output

def train_modern_cnn(model, train_loader, val_loader, num_epochs=100):
    """
    Training loop with modern techniques
    """
    # Modern optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Data augmentation techniques
    mixup = MixUp(alpha=0.2)
    cutmix = CutMix(alpha=1.0)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Random choice of augmentation
            aug_choice = np.random.choice(['normal', 'mixup', 'cutmix'], p=[0.5, 0.25, 0.25])
            
            if aug_choice == 'mixup':
                inputs, y_a, y_b, lam = mixup((inputs, labels))
                outputs = model(inputs)
                loss = mixup.mixup_criterion(criterion, outputs, y_a, y_b, lam)
            elif aug_choice == 'cutmix':
                inputs, y_a, y_b, lam = cutmix((inputs, labels))
                outputs = model(inputs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_acc = 0.0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_acc += (predicted == labels).sum().item()
        
        val_acc = val_acc / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_modern_cnn.pth')
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Acc: {val_acc:.4f}, Best: {best_acc:.4f}')

print("Modern CNN Training Checklist:")
print("✅ EfficientNet or ResNet backbone")
print("✅ Attention mechanisms (CBAM, SE)")
print("✅ AdamW optimizer with weight decay")
print("✅ Cosine annealing with warm restarts")
print("✅ Label smoothing")
print("✅ MixUp and CutMix augmentation")
print("✅ Proper validation and early stopping")
```

## 🎯 Real-World Implementation Tips

### Production Deployment Considerations

```python
def optimize_for_deployment(model):
    """
    Optimize model for production deployment
    """
    # 1. Model Pruning
    def prune_model(model, amount=0.2):
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        return model
    
    # 2. Quantization
    def quantize_model(model):
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    # 3. ONNX Export
    def export_to_onnx(model, dummy_input, filename='model.onnx'):
        model.eval()
        torch.onnx.export(
            model, dummy_input, filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
    
    print("Deployment Optimization Steps:")
    print("1. Pruning: Remove 20% of least important weights")
    print("2. Quantization: Convert to 8-bit integers")
    print("3. ONNX Export: Cross-platform compatibility")
    print("4. TensorRT/TensorFlow Lite: Hardware-specific optimization")
    
    return model

print("\nAdvanced CNN Techniques Summary:")
print("=" * 50)
print("🔍 Attention: CBAM, Self-Attention, Squeeze-and-Excitation")
print("🏗️ Architecture: EfficientNet, Vision Transformers, NAS")
print("⚡ Training: MixUp, CutMix, Knowledge Distillation")
print("📱 Efficiency: MobileNet, Pruning, Quantization")
print("🔬 Research: Self-Supervised Learning, Multi-Scale Features")
```

## 📚 Key Takeaways

1. **Attention mechanisms** help models focus on important features
2. **Efficient architectures** balance accuracy and computational cost
3. **Advanced training techniques** improve generalization and robustness
4. **Mobile optimizations** enable edge deployment
5. **Self-supervised learning** reduces dependence on labeled data

The field of computer vision is rapidly evolving. These advanced techniques represent the current state-of-the-art, but new innovations emerge constantly. The key is understanding the principles behind these techniques so you can adapt to future developments.

## 📝 Quick Check: Test Your Understanding

1. How does channel attention differ from spatial attention?
2. What makes EfficientNet more efficient than traditional CNNs?
3. When would you use MixUp vs CutMix augmentation?
4. How does knowledge distillation help with model deployment?

Ready to explore Recurrent Neural Networks and sequence modeling? Let's continue our deep learning journey!
