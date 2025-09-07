# Image Segmentation Techniques 🧩

## Overview

Image segmentation is the process of partitioning an image into multiple meaningful regions or segments. Unlike classification (what?) or detection (where?), segmentation answers "where exactly?" at the pixel level.

## Types of Image Segmentation

### 1. Semantic Segmentation

**Definition:** Assign a class label to every pixel in the image.

**Characteristics:**
- Pixel-level classification
- Same class objects are treated as one entity
- No distinction between different instances of the same class

**Example:** All pixels belonging to "car" are labeled as "car", regardless of how many cars are in the image.

### 2. Instance Segmentation

**Definition:** Detect and segment individual object instances.

**Characteristics:**
- Combines object detection and segmentation
- Each object instance gets a unique mask
- Distinguishes between different instances of the same class

**Example:** Multiple cars in an image get separate segmentation masks.

### 3. Panoptic Segmentation

**Definition:** Combines semantic and instance segmentation.

**Characteristics:**
- Every pixel gets both a class label and instance ID
- Handles both "stuff" (background, sky) and "things" (objects)
- Provides complete scene understanding

## Semantic Segmentation

### U-Net Architecture

**Key Innovation:** Encoder-decoder architecture with skip connections for precise localization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net implementation for semantic segmentation"""
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (expansive path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits

# Training function for U-Net
def train_unet(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Training loop for U-Net semantic segmentation"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('-' * 50)
    
    return model
```

### FCN (Fully Convolutional Networks)

**Key Innovation:** Replace fully connected layers with convolutional layers for pixel-wise prediction.

```python
class FCN(nn.Module):
    """Fully Convolutional Network for semantic segmentation"""
    def __init__(self, num_classes, backbone='vgg16'):
        super(FCN, self).__init__()
        
        if backbone == 'vgg16':
            # Load pre-trained VGG16
            vgg = models.vgg16(pretrained=True)
            
            # Convert to fully convolutional
            self.features = vgg.features
            
            # Replace classifier with convolutional layers
            self.classifier = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, num_classes, 1)
            )
            
            # Skip connections for finer details
            self.score_pool3 = nn.Conv2d(256, num_classes, 1)
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Forward through feature extraction
        pool3 = pool4 = None
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 16:  # After pool3
                pool3 = x
            elif i == 23:  # After pool4
                pool4 = x
        
        # Classifier
        x = self.classifier(x)
        
        # Upsample and add skip connections
        # FCN-32s: Direct upsampling
        score = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # FCN-16s: Add pool4 skip connection
        if pool4 is not None:
            score_pool4 = self.score_pool4(pool4)
            score = F.interpolate(x, size=score_pool4.size()[2:], mode='bilinear', align_corners=False)
            score += score_pool4
            score = F.interpolate(score, size=input_size, mode='bilinear', align_corners=False)
        
        # FCN-8s: Add pool3 skip connection
        if pool3 is not None:
            score_pool3 = self.score_pool3(pool3)
            score = F.interpolate(score, size=score_pool3.size()[2:], mode='bilinear', align_corners=False)
            score += score_pool3
            score = F.interpolate(score, size=input_size, mode='bilinear', align_corners=False)
        
        return score
```

### DeepLab with Atrous Convolution

**Key Innovation:** Atrous (dilated) convolutions to increase receptive field without losing resolution.

```python
class AtrousConv2d(nn.Module):
    """Atrous (Dilated) Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding=None):
        super(AtrousConv2d, self).__init__()
        
        if padding is None:
            padding = dilation
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             dilation=dilation, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # Different dilation rates
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = AtrousConv2d(in_channels, out_channels, 1, dilation=dilations[0])
        self.aspp2 = AtrousConv2d(in_channels, out_channels, 3, dilation=dilations[1])
        self.aspp3 = AtrousConv2d(in_channels, out_channels, 3, dilation=dilations[2])
        self.aspp4 = AtrousConv2d(in_channels, out_channels, 3, dilation=dilations[3])
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class DeepLabV3(nn.Module):
    """DeepLab v3 implementation"""
    def __init__(self, num_classes, backbone='resnet50'):
        super(DeepLabV3, self).__init__()
        
        # Backbone with modified stride
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            
            # Modify last two blocks to use atrous convolution
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            # Modify stride in last layers
            self._modify_resnet_stride()
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def _modify_resnet_stride(self):
        """Modify ResNet to use dilated convolutions"""
        # This is a simplified version
        # In practice, you'd modify the stride and dilation of specific layers
        pass
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Extract features
        x = self.backbone(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
```

## Instance Segmentation

### Mask R-CNN

**Key Innovation:** Extends Faster R-CNN by adding a mask prediction branch.

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(nn.Module):
    """Mask R-CNN for instance segmentation"""
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        
        # Load pre-trained Mask R-CNN model
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

def train_mask_rcnn(model, train_loader, num_epochs=10, device='cuda'):
    """Training loop for Mask R-CNN"""
    
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses.item():.4f}')
        
        lr_scheduler.step()
        print(f'Epoch {epoch} completed. Average loss: {epoch_loss/len(train_loader):.4f}')
    
    return model

def predict_mask_rcnn(model, image, device='cuda', confidence_threshold=0.5):
    """Inference with Mask R-CNN"""
    model.eval()
    
    with torch.no_grad():
        # Ensure image is on correct device and in right format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        image = image.to(device).unsqueeze(0)
        
        # Prediction
        predictions = model(image)
        
        # Filter predictions by confidence
        prediction = predictions[0]
        
        # Filter by confidence threshold
        keep = prediction['scores'] > confidence_threshold
        
        filtered_prediction = {
            'boxes': prediction['boxes'][keep],
            'labels': prediction['labels'][keep],
            'scores': prediction['scores'][keep],
            'masks': prediction['masks'][keep]
        }
        
        return filtered_prediction

def visualize_instance_segmentation(image, prediction, class_names=None):
    """Visualize instance segmentation results"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmentation result
    ax2.imshow(image)
    
    # Color map for different instances
    colors = plt.cm.Set3(np.linspace(0, 1, len(prediction['masks'])))
    
    for i, (box, label, score, mask) in enumerate(zip(
        prediction['boxes'], prediction['labels'], 
        prediction['scores'], prediction['masks']
    )):
        # Draw bounding box
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=colors[i], 
                               facecolor='none')
        ax2.add_patch(rect)
        
        # Draw mask
        mask_np = mask[0].cpu().numpy()
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[:, :, :3] = colors[i][:3]
        colored_mask[:, :, 3] = mask_np * 0.5  # Semi-transparent
        
        ax2.imshow(colored_mask)
        
        # Add label
        label_text = f'Class {label.item()}: {score:.2f}'
        if class_names and label.item() < len(class_names):
            label_text = f'{class_names[label.item()]}: {score:.2f}'
        
        ax2.text(x1, y1-5, label_text, fontsize=10, 
                bbox=dict(facecolor=colors[i], alpha=0.7))
    
    ax2.set_title('Instance Segmentation')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### SOLO (Segmenting Objects by Locations)

**Key Innovation:** Location-based instance segmentation without bounding box detection.

```python
class SOLO(nn.Module):
    """SOLO for instance segmentation"""
    def __init__(self, num_classes, backbone='resnet50'):
        super(SOLO, self).__init__()
        
        # Backbone FPN
        self.backbone = self._build_backbone(backbone)
        
        # SOLO heads
        self.solo_head = SOLOHead(256, num_classes)
    
    def _build_backbone(self, backbone_name):
        """Build backbone with FPN"""
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            return torchvision.ops.FeaturePyramidNetwork(
                in_channels_list=[256, 512, 1024, 2048],
                out_channels=256
            )
    
    def forward(self, images):
        # Extract multi-scale features
        features = self.backbone(images)
        
        # SOLO head predictions
        cls_outputs, mask_outputs = self.solo_head(features)
        
        return cls_outputs, mask_outputs

class SOLOHead(nn.Module):
    """SOLO detection head"""
    def __init__(self, in_channels, num_classes):
        super(SOLOHead, self).__init__()
        
        # Grid sizes for different feature levels
        self.grid_sizes = [40, 36, 24, 16, 12]
        
        # Classification branch
        self.cls_convs = nn.ModuleList()
        for i in range(4):  # 4 conv layers
            self.cls_convs.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
        
        self.cls_output = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        
        # Mask branch
        self.mask_convs = nn.ModuleList()
        for i in range(4):
            self.mask_convs.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
        
        # Different grid sizes need different output channels
        self.mask_outputs = nn.ModuleList()
        for grid_size in self.grid_sizes:
            self.mask_outputs.append(
                nn.Conv2d(in_channels, grid_size * grid_size, 1)
            )
    
    def forward(self, features):
        cls_outputs = []
        mask_outputs = []
        
        for i, feature in enumerate(features):
            # Classification branch
            cls_feat = feature
            for conv in self.cls_convs:
                cls_feat = F.relu(conv(cls_feat))
            
            cls_output = self.cls_output(cls_feat)
            
            # Resize to grid size
            grid_size = self.grid_sizes[i]
            cls_output = F.interpolate(cls_output, size=(grid_size, grid_size))
            cls_outputs.append(cls_output)
            
            # Mask branch
            mask_feat = feature
            for conv in self.mask_convs:
                mask_feat = F.relu(conv(mask_feat))
            
            mask_output = self.mask_outputs[i](mask_feat)
            mask_outputs.append(mask_output)
        
        return cls_outputs, mask_outputs

def solo_loss(cls_outputs, mask_outputs, targets):
    """SOLO loss function"""
    # Classification loss
    cls_loss = 0
    for cls_output, target in zip(cls_outputs, targets['cls_targets']):
        cls_loss += F.focal_loss(cls_output, target)
    
    # Mask loss
    mask_loss = 0
    for mask_output, target in zip(mask_outputs, targets['mask_targets']):
        mask_loss += F.binary_cross_entropy_with_logits(mask_output, target)
    
    return cls_loss + mask_loss
```

## Advanced Segmentation Techniques

### Attention U-Net

**Innovation:** Incorporates attention mechanisms to focus on relevant features.

```python
class AttentionGate(nn.Module):
    """Attention gate for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # Gating signal
        g1 = self.W_g(g)
        
        # Feature signal
        x1 = self.W_x(x)
        
        # Attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi

class AttentionUNet(nn.Module):
    """U-Net with attention gates"""
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder with attention
        self.up1 = Up(1024, 512)
        self.att1 = AttentionGate(512, 512, 256)
        
        self.up2 = Up(512, 256)
        self.att2 = AttentionGate(256, 256, 128)
        
        self.up3 = Up(256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        
        self.up4 = Up(128, 64)
        self.att4 = AttentionGate(64, 64, 32)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        d4 = self.up1(x5)
        x4_att = self.att1(d4, x4)
        d4 = torch.cat([x4_att, d4], dim=1)
        
        d3 = self.up2(d4)
        x3_att = self.att2(d3, x3)
        d3 = torch.cat([x3_att, d3], dim=1)
        
        d2 = self.up3(d3)
        x2_att = self.att3(d2, x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        
        d1 = self.up4(d2)
        x1_att = self.att4(d1, x1)
        d1 = torch.cat([x1_att, d1], dim=1)
        
        return self.outc(d1)
```

### Multi-Scale Segmentation

```python
class MultiScaleSegmentation(nn.Module):
    """Multi-scale segmentation network"""
    def __init__(self, num_classes, scales=[0.5, 1.0, 1.5, 2.0]):
        super(MultiScaleSegmentation, self).__init__()
        
        self.scales = scales
        self.base_model = DeepLabV3(num_classes)
        
        # Scale-specific refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Conv2d(num_classes, num_classes, 3, padding=1)
            for _ in scales
        ])
        
        # Final fusion layer
        self.fusion = nn.Conv2d(num_classes * len(scales), num_classes, 1)
    
    def forward(self, x):
        original_size = x.size()[2:]
        multi_scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Resize input
            if scale != 1.0:
                scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_x = F.interpolate(x, size=scaled_size, mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Get prediction at this scale
            output = self.base_model(scaled_x)
            
            # Resize back to original size
            if scale != 1.0:
                output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
            
            # Apply scale-specific refinement
            output = self.refinement_layers[i](output)
            multi_scale_outputs.append(output)
        
        # Fuse multi-scale predictions
        fused = torch.cat(multi_scale_outputs, dim=1)
        final_output = self.fusion(fused)
        
        return final_output
```

## Evaluation Metrics for Segmentation

### IoU and Dice Coefficient

```python
def calculate_iou(pred, target, num_classes):
    """Calculate Intersection over Union (IoU) for segmentation"""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return ious

def calculate_dice_coefficient(pred, target, num_classes):
    """Calculate Dice coefficient for segmentation"""
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum() + target_cls.sum()
        
        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2 * intersection) / total
        
        dice_scores.append(dice.item())
    
    return dice_scores

def pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy"""
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item()

def mean_iou(pred, target, num_classes):
    """Calculate mean IoU across all classes"""
    ious = calculate_iou(pred, target, num_classes)
    return np.mean(ious)

class SegmentationMetrics:
    """Comprehensive segmentation evaluation"""
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.total_iou = np.zeros(self.num_classes)
        self.total_dice = np.zeros(self.num_classes)
        self.total_accuracy = 0.0
        self.count = 0
    
    def update(self, pred, target):
        """Update metrics with new predictions"""
        # Convert predictions to class indices
        if pred.dim() > 3:  # If logits, convert to predictions
            pred = torch.argmax(pred, dim=1)
        
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            pred_i = pred[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            
            # Calculate metrics
            ious = calculate_iou(pred_i, target_i, self.num_classes)
            dice_scores = calculate_dice_coefficient(pred_i, target_i, self.num_classes)
            accuracy = pixel_accuracy(pred_i, target_i)
            
            # Accumulate
            self.total_iou += np.array(ious)
            self.total_dice += np.array(dice_scores)
            self.total_accuracy += accuracy
            self.count += 1
    
    def compute(self):
        """Compute final metrics"""
        if self.count == 0:
            return {}
        
        mean_ious = self.total_iou / self.count
        mean_dice = self.total_dice / self.count
        mean_accuracy = self.total_accuracy / self.count
        
        results = {
            'pixel_accuracy': mean_accuracy,
            'mean_iou': np.mean(mean_ious),
            'mean_dice': np.mean(mean_dice),
            'class_ious': dict(zip(self.class_names, mean_ious)),
            'class_dice': dict(zip(self.class_names, mean_dice))
        }
        
        return results
    
    def print_results(self):
        """Print detailed results"""
        results = self.compute()
        
        print("Segmentation Results:")
        print("=" * 50)
        print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        print(f"Mean Dice: {results['mean_dice']:.4f}")
        print()
        
        print("Per-Class Results:")
        print("-" * 30)
        for class_name in self.class_names:
            iou = results['class_ious'][class_name]
            dice = results['class_dice'][class_name]
            print(f"{class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
```

## Loss Functions for Segmentation

### Focal Loss for Class Imbalance

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combination of Cross Entropy and Dice Loss"""
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice
```

## Real-World Applications

### Medical Image Segmentation

```python
class MedicalSegmentationPipeline:
    """Complete pipeline for medical image segmentation"""
    def __init__(self, model_path=None):
        self.model = UNet(n_channels=1, n_classes=2)  # Binary segmentation
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def preprocess_medical_image(self, image_path):
        """Preprocess medical image (DICOM, etc.)"""
        # Load medical image (simplified)
        import SimpleITK as sitk
        
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Normalize to [0, 1]
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        
        return image_tensor
    
    def segment_organ(self, image_path, organ_name='liver'):
        """Segment specific organ from medical image"""
        # Preprocess
        image_tensor = self.preprocess_medical_image(image_path)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.softmax(prediction, dim=1)
            mask = torch.argmax(prediction, dim=1)
        
        return mask.squeeze().numpy()
    
    def calculate_volume(self, mask, pixel_spacing):
        """Calculate organ volume from segmentation mask"""
        # Count pixels belonging to organ
        organ_pixels = np.sum(mask == 1)
        
        # Calculate volume (simplified)
        pixel_volume = np.prod(pixel_spacing)  # mm³ per pixel
        total_volume = organ_pixels * pixel_volume / 1000  # Convert to cm³
        
        return total_volume
    
    def generate_3d_visualization(self, mask):
        """Generate 3D visualization of segmented organ"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Find organ boundaries
        organ_coords = np.where(mask == 1)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot organ points
        ax.scatter(organ_coords[0], organ_coords[1], organ_coords[2], 
                  alpha=0.1, s=1, c='red')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Organ Segmentation')
        
        plt.show()

# Usage example
def medical_segmentation_workflow():
    """Complete medical segmentation workflow"""
    pipeline = MedicalSegmentationPipeline('best_medical_unet.pth')
    
    # Segment organ
    mask = pipeline.segment_organ('patient_scan.dcm', 'liver')
    
    # Calculate volume
    pixel_spacing = [1.0, 1.0, 5.0]  # mm
    volume = pipeline.calculate_volume(mask, pixel_spacing)
    print(f"Liver volume: {volume:.2f} cm³")
    
    # Generate visualization
    pipeline.generate_3d_visualization(mask)
```

### Autonomous Driving Segmentation

```python
class AutonomousDrivingSegmentation:
    """Semantic segmentation for autonomous driving"""
    def __init__(self):
        # Classes for driving scenes
        self.classes = {
            0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
            5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
            9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
            14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
        }
        
        self.model = DeepLabV3(len(self.classes))
        
        # Color map for visualization
        self.color_map = self._create_color_map()
    
    def _create_color_map(self):
        """Create color map for visualization"""
        colors = [
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32]     # bicycle
        ]
        return np.array(colors)
    
    def segment_driving_scene(self, image):
        """Segment driving scene"""
        # Preprocess
        image_tensor = self._preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.softmax(prediction, dim=1)
            mask = torch.argmax(prediction, dim=1)
        
        return mask.squeeze().cpu().numpy()
    
    def _preprocess_image(self, image):
        """Preprocess image for model"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image.unsqueeze(0)
    
    def analyze_drivable_area(self, mask):
        """Analyze drivable area from segmentation"""
        # Drivable classes: road, sidewalk (with restrictions)
        road_mask = (mask == 0)  # road class
        
        # Calculate drivable area percentage
        total_pixels = mask.size
        drivable_pixels = np.sum(road_mask)
        drivable_percentage = (drivable_pixels / total_pixels) * 100
        
        # Find road boundaries
        road_edges = self._find_road_edges(road_mask)
        
        return {
            'drivable_percentage': drivable_percentage,
            'road_edges': road_edges,
            'road_mask': road_mask
        }
    
    def _find_road_edges(self, road_mask):
        """Find road lane boundaries"""
        # Simple edge detection on road mask
        from scipy import ndimage
        
        # Sobel edge detection
        edges_x = ndimage.sobel(road_mask.astype(float), axis=1)
        edges_y = ndimage.sobel(road_mask.astype(float), axis=0)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        return edges > 0.1  # Threshold for edge detection
    
    def visualize_segmentation(self, image, mask):
        """Visualize segmentation results"""
        # Create colored segmentation
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.color_map):
            colored_mask[mask == class_id] = color
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        
        return overlay

# Usage for autonomous driving
def driving_segmentation_demo(video_path):
    """Demo for driving scene segmentation"""
    segmenter = AutonomousDrivingSegmentation()
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Segment frame
        mask = segmenter.segment_driving_scene(frame)
        
        # Analyze drivable area
        driving_analysis = segmenter.analyze_drivable_area(mask)
        
        # Visualize
        visualization = segmenter.visualize_segmentation(frame, mask)
        
        # Add text overlay
        cv2.putText(visualization, 
                   f"Drivable Area: {driving_analysis['drivable_percentage']:.1f}%",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Driving Segmentation', visualization)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Next Steps

Mastery of image segmentation prepares you for:

1. **3D Segmentation**: Extending to volumetric data
2. **Video Segmentation**: Temporal consistency in video sequences
3. **Real-time Segmentation**: Optimized networks for mobile/edge deployment
4. **Weakly Supervised Segmentation**: Learning with limited annotations
5. **Domain Adaptation**: Transferring segmentation across domains

Image segmentation is fundamental to many applications requiring precise spatial understanding, from medical diagnosis to autonomous navigation.
