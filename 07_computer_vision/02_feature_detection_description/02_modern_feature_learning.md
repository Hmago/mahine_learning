# Modern Feature Learning with Deep Networks 🧠

## Overview

While classical features are manually designed, modern computer vision leverages deep learning to automatically learn the most relevant features for specific tasks. This represents a paradigm shift from hand-crafted to data-driven feature extraction.

## Evolution from Classical to Modern Features

### The Limitation of Hand-crafted Features

**Classical Approach Problems:**
- **Manual Design**: Features are designed by humans, may miss optimal patterns
- **Task-Specific**: Different tasks require different feature engineering
- **Limited Complexity**: Cannot capture very complex patterns
- **Scale Issues**: May not generalize across different scales or domains

**Deep Learning Revolution:**
- **Automatic Learning**: Networks discover features from data
- **Hierarchical Features**: Learn from simple to complex patterns
- **Task-Adaptive**: Features adapt to specific objectives
- **End-to-End**: Feature learning integrated with task solving

## How CNNs Learn Visual Features

### Hierarchical Feature Learning

**Layer-by-Layer Feature Evolution:**

**Layer 1 (Low-level features):**
- Edge detectors
- Color blobs
- Simple textures
- Gabor-like filters

**Layer 2-3 (Mid-level features):**
- Corners and junctions
- Simple shapes
- Texture patterns
- Color combinations

**Layer 4-5 (High-level features):**
- Object parts
- Complex textures
- Semantic patterns
- Spatial relationships

**Final Layers (Task-specific features):**
- Complete object representations
- Scene understanding
- Task-specific patterns

### Feature Visualization Techniques

#### 1. Activation Maximization

**Concept:** Find input patterns that maximally activate specific neurons.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def visualize_filter_activation(model, layer_name, filter_idx, size=(224, 224)):
    """
    Visualize what activates a specific filter maximally
    """
    # Hook to get intermediate activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations[layer_name] = output
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
    
    # Create random input
    input_image = torch.randn(1, 3, size[0], size[1], requires_grad=True)
    
    # Optimizer for input
    optimizer = optim.Adam([input_image], lr=0.1)
    
    # Optimization loop
    for i in range(200):
        optimizer.zero_grad()
        
        # Forward pass
        _ = model(input_image)
        
        # Get activation for specific filter
        if layer_name in activations:
            activation = activations[layer_name][0, filter_idx].mean()
            
            # Backward pass to update input
            loss = -activation  # Negative to maximize
            loss.backward()
            optimizer.step()
    
    # Convert to displayable format
    result = input_image.detach().squeeze().permute(1, 2, 0)
    result = (result - result.min()) / (result.max() - result.min())
    
    return result.numpy()
```

#### 2. Gradient-based Visualization

**Saliency Maps:** Show which pixels contribute most to predictions.

```python
def generate_saliency_map(model, image, target_class):
    """
    Generate saliency map showing important pixels
    """
    # Set model to evaluation mode
    model.eval()
    
    # Ensure image requires gradients
    image.requires_grad_()
    
    # Forward pass
    output = model(image)
    
    # Get gradient w.r.t. target class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    gradients = image.grad.data.abs()
    
    # Take maximum across color channels
    saliency = torch.max(gradients, dim=1)[0]
    
    return saliency.squeeze().cpu().numpy()

def integrated_gradients(model, image, target_class, steps=50):
    """
    More robust attribution method using integrated gradients
    """
    model.eval()
    
    # Baseline (black image)
    baseline = torch.zeros_like(image)
    
    # Generate interpolated images
    alphas = torch.linspace(0, 1, steps)
    integrated_grad = torch.zeros_like(image)
    
    for alpha in alphas:
        # Interpolated image
        interpolated = baseline + alpha * (image - baseline)
        interpolated.requires_grad_()
        
        # Forward pass
        output = model(interpolated)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Accumulate gradients
        integrated_grad += interpolated.grad / steps
    
    # Scale by input difference
    attribution = (image - baseline) * integrated_grad
    
    return attribution.detach()
```

#### 3. Feature Map Visualization

**Concept:** Visualize intermediate feature maps to understand what networks detect.

```python
def visualize_feature_maps(model, image, layer_names):
    """
    Visualize feature maps from multiple layers
    """
    # Dictionary to store activations
    feature_maps = {}
    
    # Hook function
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Visualize feature maps
    fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, 5 * len(layer_names)))
    
    for i, layer_name in enumerate(layer_names):
        if layer_name in feature_maps:
            features = feature_maps[layer_name].squeeze()
            
            # Show first 8 feature maps
            for j in range(min(8, features.shape[0])):
                ax = axes[i, j] if len(layer_names) > 1 else axes[j]
                ax.imshow(features[j].cpu().numpy(), cmap='viridis')
                ax.set_title(f'{layer_name}_filter_{j}')
                ax.axis('off')
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    plt.tight_layout()
    plt.show()
    
    return feature_maps
```

## Transfer Learning for Feature Extraction

### Pre-trained Networks as Feature Extractors

**Why Transfer Learning Works:**
- Networks trained on ImageNet learn universal visual features
- Lower layers capture general patterns (edges, textures)
- Higher layers can be fine-tuned for specific tasks
- Significantly reduces training time and data requirements

#### Feature Extraction Pipeline

```python
import torchvision.models as models
import torch.nn as nn

class FeatureExtractor:
    def __init__(self, model_name='resnet50', layer_name='avgpool'):
        """
        Initialize feature extractor with pre-trained model
        """
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Remove classification head
        if model_name == 'resnet50':
            self.features = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'vgg16':
            self.features = self.model.features
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features.to(self.device)
    
    def extract_features(self, images):
        """
        Extract features from batch of images
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.features(images)
            
            # Flatten features
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy()
    
    def extract_layer_features(self, images, layer_name):
        """
        Extract features from specific layer
        """
        features = {}
        
        def hook_fn(module, input, output):
            features['target'] = output.detach()
        
        # Register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(images.to(self.device))
        
        # Remove hook
        hook.remove()
        
        return features['target'].cpu()

# Usage example
def build_feature_database(image_loader, extractor):
    """
    Build feature database for image retrieval
    """
    all_features = []
    all_labels = []
    
    for images, labels in image_loader:
        # Extract features
        features = extractor.extract_features(images)
        
        all_features.append(features)
        all_labels.extend(labels.numpy())
    
    # Concatenate all features
    feature_matrix = np.vstack(all_features)
    
    return feature_matrix, np.array(all_labels)
```

### Fine-tuning Strategies

#### 1. Feature Extraction (Frozen Backbone)

```python
def create_feature_extractor_model(num_classes, backbone='resnet50'):
    """
    Create model with frozen backbone for feature extraction
    """
    # Load pre-trained model
    if backbone == 'resnet50':
        model = models.resnet50(pretrained=True)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    return model
```

#### 2. Fine-tuning (Unfrozen Layers)

```python
def create_fine_tuning_model(num_classes, backbone='resnet50', 
                            unfreeze_layers=1):
    """
    Create model for fine-tuning with selective unfreezing
    """
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    layers = list(model.children())
    for layer in layers[:-unfreeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def get_fine_tuning_optimizer(model, backbone_lr=1e-4, classifier_lr=1e-3):
    """
    Different learning rates for different parts of the network
    """
    # Separate parameters
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:  # classifier parameters
            classifier_params.append(param)
        else:  # backbone parameters
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ])
    
    return optimizer
```

## Attention Mechanisms in Vision

### Spatial Attention

**Concept:** Learn to focus on relevant spatial locations in images.

```python
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        # Attention mechanism
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        attention = self.conv1(x)
        attention = torch.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        attended = x * attention
        
        return attended, attention

class AttentionCNN(nn.Module):
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
        
        # Backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Attention module
        self.attention = SpatialAttention(2048)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attended_features, attention_map = self.attention(features)
        
        # Global pooling and classification
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)
        
        return output, attention_map
```

### Channel Attention (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.global_pool(x).view(batch_size, channels)
        
        # Excitation: FC layers with sigmoid
        excitation = torch.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))
        
        # Scale original features
        excitation = excitation.view(batch_size, channels, 1, 1)
        scaled = x * excitation
        
        return scaled
```

## Modern Architecture Features

### Vision Transformer (ViT) Features

**Key Innovation:** Treat images as sequences of patches, apply transformer attention.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        out = self.projection(out)
        
        return out, attention
```

### EfficientNet Features

**Key Innovation:** Compound scaling (depth, width, resolution) with neural architecture search.

```python
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze-and-excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze-and-excitation
        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight
        
        # Output projection
        x = self.project(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x
```

## Feature Quality Assessment

### Feature Discriminability

```python
def assess_feature_discriminability(features, labels):
    """
    Assess how well features separate different classes
    """
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    # Dimensionality reduction for visualization
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features
    
    # Silhouette score (higher is better)
    silhouette = silhouette_score(features_reduced, labels)
    
    # Intra-class vs inter-class distances
    unique_labels = np.unique(labels)
    intra_distances = []
    inter_distances = []
    
    for label in unique_labels:
        class_features = features_reduced[labels == label]
        other_features = features_reduced[labels != label]
        
        # Intra-class distances
        if len(class_features) > 1:
            from scipy.spatial.distance import pdist
            intra_dist = np.mean(pdist(class_features))
            intra_distances.append(intra_dist)
        
        # Inter-class distances
        if len(other_features) > 0:
            from scipy.spatial.distance import cdist
            inter_dist = np.mean(cdist(class_features, other_features))
            inter_distances.append(inter_dist)
    
    avg_intra = np.mean(intra_distances)
    avg_inter = np.mean(inter_distances)
    separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    return {
        'silhouette_score': silhouette,
        'intra_class_distance': avg_intra,
        'inter_class_distance': avg_inter,
        'separation_ratio': separation_ratio
    }
```

### Feature Stability

```python
def assess_feature_stability(model, images, noise_levels=[0.01, 0.05, 0.1]):
    """
    Assess how stable features are to input perturbations
    """
    model.eval()
    
    # Extract original features
    with torch.no_grad():
        original_features = model(images)
    
    stability_scores = []
    
    for noise_level in noise_levels:
        # Add noise to images
        noise = torch.randn_like(images) * noise_level
        noisy_images = images + noise
        
        # Extract features from noisy images
        with torch.no_grad():
            noisy_features = model(noisy_images)
        
        # Calculate similarity
        similarity = torch.cosine_similarity(original_features, noisy_features, dim=1)
        stability_scores.append(similarity.mean().item())
    
    return stability_scores
```

## Practical Applications

### 1. Visual Search Engine

```python
class VisualSearchEngine:
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.database_features = None
        self.database_images = None
    
    def build_database(self, image_paths):
        """Build feature database from image collection"""
        features = []
        
        for img_path in image_paths:
            # Load and preprocess image
            img = self.load_and_preprocess(img_path)
            
            # Extract features
            feature = self.extractor.extract_features(img.unsqueeze(0))
            features.append(feature)
        
        self.database_features = np.vstack(features)
        self.database_images = image_paths
    
    def search(self, query_image, top_k=5):
        """Search for similar images"""
        # Extract query features
        query_features = self.extractor.extract_features(query_image.unsqueeze(0))
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_features, self.database_features)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            {
                'image_path': self.database_images[idx],
                'similarity': similarities[idx]
            }
            for idx in top_indices
        ]
        
        return results
```

### 2. Content-Based Image Retrieval

```python
def build_cbir_system(image_directory, feature_extractor):
    """
    Build Content-Based Image Retrieval system
    """
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features from all images
    image_paths = []
    features = []
    
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_directory, filename)
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                
                # Extract features
                feature = feature_extractor.extract_features(img_tensor)
                
                image_paths.append(img_path)
                features.append(feature)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Stack features
    feature_matrix = np.vstack(features)
    
    return image_paths, feature_matrix

def query_cbir_system(query_image_path, image_paths, feature_matrix, 
                     feature_extractor, top_k=10):
    """
    Query the CBIR system with a new image
    """
    # Process query image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    query_img = Image.open(query_image_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0)
    
    # Extract query features
    query_features = feature_extractor.extract_features(query_tensor)
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_features, feature_matrix)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'image_path': image_paths[idx],
            'similarity': similarities[idx]
        })
    
    return results
```

## Advanced Topics

### Self-Supervised Learning for Features

```python
class SimCLR(nn.Module):
    """Simple Contrastive Learning of Visual Representations"""
    
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        
        # Backbone network
        self.backbone = backbone
        backbone_dim = list(backbone.children())[-1].in_features
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, projection_dim)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Project features
        projections = self.projection(features)
        
        return features, projections

def contrastive_loss(projections, temperature=0.1):
    """
    Contrastive loss for self-supervised learning
    """
    batch_size = projections.shape[0] // 2
    
    # Normalize projections
    projections = F.normalize(projections, dim=1)
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create labels (positive pairs)
    labels = torch.cat([torch.arange(batch_size) + batch_size, 
                       torch.arange(batch_size)]).to(projections.device)
    
    # Remove self-similarity
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(projections.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Calculate loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss
```

### Feature Fusion Techniques

```python
class MultiModalFeatureFusion(nn.Module):
    """Fuse features from multiple modalities or scales"""
    
    def __init__(self, feature_dims, fusion_method='concat'):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.output_dim = sum(feature_dims)
        elif fusion_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(feature_dims)))
            self.output_dim = feature_dims[0]  # Assume same dimension
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(feature_dims[0], num_heads=8)
            self.output_dim = feature_dims[0]
    
    def forward(self, feature_list):
        if self.fusion_method == 'concat':
            return torch.cat(feature_list, dim=1)
        
        elif self.fusion_method == 'weighted':
            weighted_features = []
            for i, features in enumerate(feature_list):
                weighted_features.append(self.weights[i] * features)
            return torch.stack(weighted_features).sum(dim=0)
        
        elif self.fusion_method == 'attention':
            # Stack features for attention
            stacked = torch.stack(feature_list, dim=0)  # (num_features, batch, dim)
            
            # Apply attention
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # Average across feature types
            return attended.mean(dim=0)
```

## Next Steps

This comprehensive understanding of modern feature learning prepares you for:

1. **Advanced Architectures**: Vision transformers, efficient networks, neural architecture search
2. **Multimodal Learning**: Combining vision with text, audio, and other modalities
3. **Self-Supervised Learning**: Learning representations without labels
4. **Domain Adaptation**: Transferring features across different domains
5. **Production Deployment**: Optimizing feature extraction for real-world applications

Modern feature learning represents the cutting edge of computer vision, enabling systems that can automatically discover the most relevant patterns for any given task.
