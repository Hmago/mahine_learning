# Object Detection Systems 🎯

## Overview

Object detection is the task of finding and classifying objects in images. Unlike image classification which assigns a single label to an entire image, object detection must:

1. **Locate** objects (where are they?)
2. **Classify** objects (what are they?)
3. **Handle multiple objects** (how many are there?)

## Understanding Object Detection

### Detection vs Classification

**Image Classification:**
- Input: Single image
- Output: One class label
- Question: "What is in this image?"

**Object Detection:**
- Input: Single image
- Output: Multiple bounding boxes + class labels
- Question: "What objects are where in this image?"

### Key Components

**Bounding Box:** Rectangle defined by (x₁, y₁, x₂, y₂) coordinates that tightly encloses an object.

**Confidence Score:** How certain the model is that an object exists in the bounding box.

**Class Probability:** Probability distribution over possible object classes.

### Evaluation Metrics

#### Intersection over Union (IoU)

**Definition:** Overlap between predicted and ground truth bounding boxes.

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def iou_batch(boxes1, boxes2):
    """
    Vectorized IoU calculation for multiple boxes
    """
    import torch
    
    # Calculate intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # width-height
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1[:, None] + area2 - intersection
    
    return intersection / union
```

#### Mean Average Precision (mAP)

**Concept:** Average precision across all classes and IoU thresholds.

```python
def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision (AP) from precision and recall
    """
    # Add sentinel values
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate area under curve
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def calculate_map(detections, ground_truths, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    """
    aps = []
    
    for class_id in range(num_classes):
        # Get detections and ground truths for this class
        class_detections = [d for d in detections if d['class_id'] == class_id]
        class_gt = [gt for gt in ground_truths if gt['class_id'] == class_id]
        
        if len(class_gt) == 0:
            continue
        
        # Sort detections by confidence
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        gt_matched = [False] * len(class_gt)
        
        for i, detection in enumerate(class_detections):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gt):
                if gt['image_id'] == detection['image_id']:
                    iou = calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        
        recalls = cumulative_tp / len(class_gt)
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
        
        # Calculate AP
        ap = calculate_ap(precisions, recalls)
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0
```

## Evolution of Object Detection

### 1. Traditional Methods (Pre-CNN Era)

#### Sliding Window + HOG + SVM

**Approach:** Slide a window across the image and classify each window.

```python
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
import numpy as np

class SlidingWindowDetector:
    def __init__(self, window_size=(64, 128), step_size=8):
        self.window_size = window_size
        self.step_size = step_size
        self.classifier = SVC(probability=True)
        self.trained = False
    
    def extract_hog_features(self, image):
        """Extract HOG features from image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = hog(image, 
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys')
        return features
    
    def sliding_window(self, image):
        """Generate sliding windows"""
        h, w = image.shape[:2]
        win_h, win_w = self.window_size
        
        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                window = image[y:y + win_h, x:x + win_w]
                yield (x, y), window
    
    def train(self, positive_images, negative_images):
        """Train the classifier"""
        features = []
        labels = []
        
        # Extract features from positive examples
        for img in positive_images:
            resized = cv2.resize(img, self.window_size)
            feature = self.extract_hog_features(resized)
            features.append(feature)
            labels.append(1)
        
        # Extract features from negative examples
        for img in negative_images:
            resized = cv2.resize(img, self.window_size)
            feature = self.extract_hog_features(resized)
            features.append(feature)
            labels.append(0)
        
        # Train classifier
        self.classifier.fit(features, labels)
        self.trained = True
    
    def detect(self, image, threshold=0.5):
        """Detect objects in image"""
        if not self.trained:
            raise ValueError("Detector must be trained first")
        
        detections = []
        
        for (x, y), window in self.sliding_window(image):
            # Extract features
            feature = self.extract_hog_features(window)
            
            # Classify
            prob = self.classifier.predict_proba([feature])[0][1]
            
            if prob > threshold:
                detections.append({
                    'bbox': [x, y, x + self.window_size[0], y + self.window_size[1]],
                    'confidence': prob
                })
        
        return detections

# Non-Maximum Suppression for traditional methods
def nms_traditional(detections, iou_threshold=0.5):
    """
    Non-Maximum Suppression for traditional detection methods
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    indices = list(range(len(detections)))
    
    while indices:
        # Take detection with highest confidence
        current = indices.pop(0)
        keep.append(current)
        
        # Remove detections with high IoU
        remaining = []
        for i in indices:
            iou = calculate_iou(detections[current]['bbox'], detections[i]['bbox'])
            if iou <= iou_threshold:
                remaining.append(i)
        
        indices = remaining
    
    return [detections[i] for i in keep]
```

**Problems with Traditional Methods:**
- Computationally expensive (many windows)
- Limited by hand-crafted features
- Poor performance on complex scenes
- Difficulty handling scale variations

### 2. Two-Stage Detectors

#### R-CNN (2014) - Region-based CNN

**Key Innovation:** Use selective search to propose regions, then classify each region with CNN.

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import roi_pool

class RCNN(nn.Module):
    """
    Simplified R-CNN implementation
    """
    def __init__(self, num_classes, roi_size=7):
        super(RCNN, self).__init__()
        
        # CNN feature extractor
        self.backbone = models.vgg16(pretrained=True).features
        
        # RoI pooling
        self.roi_pool = roi_pool
        self.roi_size = roi_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * roi_size * roi_size, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
        # Bounding box regressor
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)
    
    def forward(self, images, rois):
        """
        Forward pass
        
        Args:
            images: Batch of images [B, C, H, W]
            rois: Region proposals [N, 5] where each row is [batch_idx, x1, y1, x2, y2]
        """
        # Extract features
        features = self.backbone(images)
        
        # RoI pooling
        pooled_features = roi_pool(features, rois, self.roi_size, spatial_scale=1/16)
        
        # Flatten
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        class_scores = self.classifier(pooled_features)
        
        # Bounding box regression
        bbox_deltas = self.bbox_regressor(pooled_features)
        
        return class_scores, bbox_deltas

def selective_search_proposals(image, num_proposals=2000):
    """
    Simplified selective search (in practice, use cv2.ximgproc.segmentation)
    """
    import cv2
    
    # Create selective search object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    
    # Get proposals
    rects = ss.process()
    
    # Convert to [x1, y1, x2, y2] format
    proposals = []
    for i, rect in enumerate(rects[:num_proposals]):
        x, y, w, h = rect
        proposals.append([x, y, x + w, y + h])
    
    return np.array(proposals)

def train_rcnn_stage(model, dataloader, num_epochs=10):
    """
    Training loop for R-CNN
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Generate proposals for each image
            all_rois = []
            all_labels = []
            all_bbox_targets = []
            
            for i, image in enumerate(images):
                # Get proposals
                proposals = selective_search_proposals(image.permute(1, 2, 0).numpy())
                
                # Add batch index
                rois = torch.cat([torch.full((len(proposals), 1), i), 
                                 torch.tensor(proposals).float()], dim=1)
                
                all_rois.append(rois)
                
                # Generate labels and bbox targets (simplified)
                # In practice, you'd match proposals to ground truth
                labels = torch.randint(0, model.num_classes, (len(proposals),))
                bbox_targets = torch.randn(len(proposals), 4)
                
                all_labels.append(labels)
                all_bbox_targets.append(bbox_targets)
            
            # Concatenate all RoIs
            rois = torch.cat(all_rois, dim=0)
            labels = torch.cat(all_labels, dim=0)
            bbox_targets = torch.cat(all_bbox_targets, dim=0)
            
            # Forward pass
            class_scores, bbox_deltas = model(images, rois)
            
            # Calculate losses
            cls_loss = criterion_cls(class_scores, labels)
            bbox_loss = criterion_bbox(bbox_deltas, bbox_targets)
            total_loss = cls_loss + bbox_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
```

**R-CNN Limitations:**
- Very slow (CNN forward pass for each proposal)
- Complex training pipeline
- Requires separate training stages

#### Fast R-CNN (2015) - Shared CNN Features

**Key Improvement:** Share CNN computation across all proposals.

```python
class FastRCNN(nn.Module):
    """
    Fast R-CNN implementation
    """
    def __init__(self, num_classes, roi_size=7):
        super(FastRCNN, self).__init__()
        
        # Shared CNN backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # RoI pooling
        self.roi_pool = roi_pool
        self.roi_size = roi_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048 * roi_size * roi_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        # Classification head
        self.cls_head = nn.Linear(1024, num_classes + 1)  # +1 for background
        
        # Bounding box regression head
        self.bbox_head = nn.Linear(1024, num_classes * 4)
    
    def forward(self, images, rois):
        # Extract shared features
        features = self.backbone(images)
        
        # RoI pooling
        pooled = roi_pool(features, rois, self.roi_size, spatial_scale=1/32)
        
        # Flatten and pass through FC layers
        x = pooled.view(pooled.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Classification and regression
        cls_scores = self.cls_head(x)
        bbox_deltas = self.bbox_head(x)
        
        return cls_scores, bbox_deltas

def fast_rcnn_loss(cls_scores, bbox_deltas, labels, bbox_targets, lambda_bbox=1.0):
    """
    Fast R-CNN multi-task loss
    """
    # Classification loss
    cls_loss = F.cross_entropy(cls_scores, labels)
    
    # Only compute bbox loss for positive examples
    positive_mask = labels > 0
    if positive_mask.sum() > 0:
        bbox_loss = F.smooth_l1_loss(
            bbox_deltas[positive_mask], 
            bbox_targets[positive_mask]
        )
    else:
        bbox_loss = torch.tensor(0.0, device=cls_scores.device)
    
    return cls_loss + lambda_bbox * bbox_loss
```

#### Faster R-CNN (2015) - End-to-End Training

**Key Innovation:** Replace selective search with learnable Region Proposal Network (RPN).

```python
class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network for Faster R-CNN
    """
    def __init__(self, in_channels, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1.0, 2.0]):
        super(RegionProposalNetwork, self).__init__()
        
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        
        # 3x3 conv for feature processing
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        
        # Classification head (object/background)
        self.cls_head = nn.Conv2d(512, self.num_anchors * 2, 1)
        
        # Regression head (bbox deltas)
        self.bbox_head = nn.Conv2d(512, self.num_anchors * 4, 1)
    
    def generate_anchors(self, feature_size, stride=16):
        """
        Generate anchor boxes for feature map
        """
        h, w = feature_size
        anchors = []
        
        for i in range(h):
            for j in range(w):
                center_x = j * stride + stride // 2
                center_y = i * stride + stride // 2
                
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        width = scale * np.sqrt(ratio)
                        height = scale / np.sqrt(ratio)
                        
                        x1 = center_x - width // 2
                        y1 = center_y - height // 2
                        x2 = center_x + width // 2
                        y2 = center_y + height // 2
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, features):
        # Process features
        x = torch.relu(self.conv(features))
        
        # Classification and regression
        cls_scores = self.cls_head(x)
        bbox_deltas = self.bbox_head(x)
        
        # Reshape outputs
        batch_size, _, h, w = cls_scores.shape
        cls_scores = cls_scores.view(batch_size, self.num_anchors, 2, h, w)
        bbox_deltas = bbox_deltas.view(batch_size, self.num_anchors, 4, h, w)
        
        return cls_scores, bbox_deltas

class FasterRCNN(nn.Module):
    """
    Faster R-CNN implementation
    """
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        
        # Backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # RPN
        self.rpn = RegionProposalNetwork(2048)
        
        # Fast R-CNN head
        self.fast_rcnn = FastRCNN(num_classes)
    
    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        
        # RPN forward
        rpn_cls_scores, rpn_bbox_deltas = self.rpn(features)
        
        if self.training:
            # Generate proposals from RPN
            proposals = self.generate_proposals(rpn_cls_scores, rpn_bbox_deltas, targets)
            
            # Fast R-CNN forward
            cls_scores, bbox_deltas = self.fast_rcnn(images, proposals)
            
            return {
                'rpn_cls_scores': rpn_cls_scores,
                'rpn_bbox_deltas': rpn_bbox_deltas,
                'cls_scores': cls_scores,
                'bbox_deltas': bbox_deltas,
                'proposals': proposals
            }
        else:
            # Inference mode
            proposals = self.generate_proposals(rpn_cls_scores, rpn_bbox_deltas)
            cls_scores, bbox_deltas = self.fast_rcnn(images, proposals)
            
            return self.postprocess_detections(cls_scores, bbox_deltas, proposals)
    
    def generate_proposals(self, cls_scores, bbox_deltas, targets=None):
        """
        Generate object proposals from RPN outputs
        """
        # Implementation would include:
        # 1. Apply bbox_deltas to anchors
        # 2. Filter by classification score
        # 3. Apply NMS
        # 4. Select top proposals
        
        # Simplified version - return dummy proposals
        batch_size = cls_scores.shape[0]
        proposals = torch.randn(batch_size * 1000, 5)  # [batch_idx, x1, y1, x2, y2]
        return proposals
    
    def postprocess_detections(self, cls_scores, bbox_deltas, proposals):
        """
        Post-process detections for inference
        """
        # Apply bbox regression
        # Apply NMS per class
        # Filter by confidence threshold
        
        # Simplified return
        return {
            'boxes': proposals[:100, 1:],  # Top 100 boxes
            'scores': torch.softmax(cls_scores, dim=1)[:100],
            'labels': torch.argmax(cls_scores, dim=1)[:100]
        }

def faster_rcnn_loss(outputs, targets):
    """
    Faster R-CNN multi-task loss
    """
    # RPN losses
    rpn_cls_loss = rpn_classification_loss(outputs['rpn_cls_scores'], targets)
    rpn_bbox_loss = rpn_regression_loss(outputs['rpn_bbox_deltas'], targets)
    
    # Fast R-CNN losses  
    fast_rcnn_cls_loss = fast_rcnn_classification_loss(outputs['cls_scores'], targets)
    fast_rcnn_bbox_loss = fast_rcnn_regression_loss(outputs['bbox_deltas'], targets)
    
    total_loss = rpn_cls_loss + rpn_bbox_loss + fast_rcnn_cls_loss + fast_rcnn_bbox_loss
    
    return {
        'total_loss': total_loss,
        'rpn_cls_loss': rpn_cls_loss,
        'rpn_bbox_loss': rpn_bbox_loss,
        'fast_rcnn_cls_loss': fast_rcnn_cls_loss,
        'fast_rcnn_bbox_loss': fast_rcnn_bbox_loss
    }
```

### 3. Single-Stage Detectors

#### YOLO (You Only Look Once) - Real-time Detection

**Key Innovation:** Directly predict bounding boxes and class probabilities in a single forward pass.

```python
class YOLO(nn.Module):
    """
    YOLO v1 implementation
    """
    def __init__(self, num_classes, grid_size=7, num_boxes=2):
        super(YOLO, self).__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        
        # Backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(192, 128, 1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Additional conv layers...
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * (num_boxes * 5 + num_classes))
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        
        # Reshape to [batch, grid, grid, num_boxes * 5 + num_classes]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.grid_size, self.grid_size, -1)
        
        return x

def yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    YOLO loss function
    
    Args:
        predictions: [batch, grid, grid, num_boxes * 5 + num_classes]
        targets: Ground truth in same format
    """
    batch_size, grid_size, _, _ = predictions.shape
    
    # Split predictions
    num_boxes = 2
    num_classes = predictions.shape[-1] - num_boxes * 5
    
    # Extract box predictions and class predictions
    box_predictions = predictions[..., :num_boxes * 5].view(batch_size, grid_size, grid_size, num_boxes, 5)
    class_predictions = predictions[..., num_boxes * 5:]
    
    # Extract targets
    box_targets = targets[..., :num_boxes * 5].view(batch_size, grid_size, grid_size, num_boxes, 5)
    class_targets = targets[..., num_boxes * 5:]
    
    # Object presence mask
    obj_mask = box_targets[..., 4] > 0  # Confidence > 0 means object present
    noobj_mask = ~obj_mask
    
    # Coordinate loss (only for cells with objects)
    coord_loss = 0
    if obj_mask.sum() > 0:
        # x, y coordinates
        coord_loss += F.mse_loss(
            box_predictions[obj_mask][..., :2], 
            box_targets[obj_mask][..., :2]
        )
        
        # width, height (square root)
        coord_loss += F.mse_loss(
            torch.sqrt(box_predictions[obj_mask][..., 2:4] + 1e-6),
            torch.sqrt(box_targets[obj_mask][..., 2:4] + 1e-6)
        )
    
    # Confidence loss
    conf_loss_obj = F.mse_loss(
        box_predictions[obj_mask][..., 4],
        box_targets[obj_mask][..., 4]
    ) if obj_mask.sum() > 0 else 0
    
    conf_loss_noobj = F.mse_loss(
        box_predictions[noobj_mask][..., 4],
        torch.zeros_like(box_predictions[noobj_mask][..., 4])
    ) if noobj_mask.sum() > 0 else 0
    
    # Classification loss
    class_loss = F.mse_loss(class_predictions, class_targets)
    
    # Total loss
    total_loss = (lambda_coord * coord_loss + 
                 conf_loss_obj + 
                 lambda_noobj * conf_loss_noobj + 
                 class_loss)
    
    return total_loss

def yolo_postprocess(predictions, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Post-process YOLO predictions
    """
    batch_size, grid_size, _, _ = predictions.shape
    
    detections = []
    
    for b in range(batch_size):
        image_detections = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for box_idx in range(2):  # 2 boxes per cell
                    # Extract box prediction
                    box_start = box_idx * 5
                    x, y, w, h, conf = predictions[b, i, j, box_start:box_start+5]
                    
                    if conf > confidence_threshold:
                        # Convert to absolute coordinates
                        center_x = (j + x) / grid_size
                        center_y = (i + y) / grid_size
                        width = w
                        height = h
                        
                        # Convert to corner format
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # Get class probabilities
                        class_probs = predictions[b, i, j, 10:]  # After 2 boxes * 5 values
                        class_conf = conf * class_probs
                        class_id = torch.argmax(class_conf)
                        
                        image_detections.append({
                            'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                            'confidence': class_conf[class_id].item(),
                            'class_id': class_id.item()
                        })
        
        # Apply NMS
        if image_detections:
            image_detections = nms_traditional(image_detections, nms_threshold)
        
        detections.append(image_detections)
    
    return detections
```

#### SSD (Single Shot MultiBox Detector)

**Key Innovation:** Multi-scale feature maps for detecting objects at different scales.

```python
class SSD(nn.Module):
    """
    SSD (Single Shot MultiBox Detector) implementation
    """
    def __init__(self, num_classes, backbone='vgg16'):
        super(SSD, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone network
        if backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.backbone = nn.ModuleList(list(vgg.features.children()))
        
        # Additional feature layers
        self.extras = nn.ModuleList([
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.Conv2d(1024, 1024, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        ])
        
        # Default boxes per feature map
        self.default_boxes = [4, 6, 6, 6, 4, 4]  # Number of default boxes per location
        
        # Classification heads
        self.classification_heads = nn.ModuleList()
        self.regression_heads = nn.ModuleList()
        
        feature_channels = [512, 1024, 512, 256, 256, 256]
        
        for i, (channels, num_boxes) in enumerate(zip(feature_channels, self.default_boxes)):
            # Classification head
            self.classification_heads.append(
                nn.Conv2d(channels, num_boxes * num_classes, 3, padding=1)
            )
            
            # Regression head (4 coordinates per box)
            self.regression_heads.append(
                nn.Conv2d(channels, num_boxes * 4, 3, padding=1)
            )
    
    def forward(self, x):
        features = []
        
        # Extract features from backbone
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 22:  # Conv4_3 features
                features.append(x)
        
        # Continue through backbone
        for i in range(23, len(self.backbone)):
            x = self.backbone[i](x)
        
        features.append(x)  # FC7 features
        
        # Additional feature extraction
        for i in range(0, len(self.extras), 2):
            x = F.relu(self.extras[i](x))
            x = F.relu(self.extras[i + 1](x))
            features.append(x)
        
        # Apply detection heads
        classifications = []
        regressions = []
        
        for feature, cls_head, reg_head in zip(features, self.classification_heads, self.regression_heads):
            # Classification
            cls_output = cls_head(feature)
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous()
            classifications.append(cls_output.view(cls_output.size(0), -1, self.num_classes))
            
            # Regression
            reg_output = reg_head(feature)
            reg_output = reg_output.permute(0, 2, 3, 1).contiguous()
            regressions.append(reg_output.view(reg_output.size(0), -1, 4))
        
        # Concatenate all predictions
        classifications = torch.cat(classifications, dim=1)
        regressions = torch.cat(regressions, dim=1)
        
        return classifications, regressions

def generate_default_boxes(feature_map_sizes, image_size=300):
    """
    Generate default boxes for SSD
    """
    default_boxes = []
    
    # Parameters for each feature map
    scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    
    for k, (fmap_size, ratios) in enumerate(zip(feature_map_sizes, aspect_ratios)):
        for i in range(fmap_size):
            for j in range(fmap_size):
                # Center coordinates
                cx = (j + 0.5) / fmap_size
                cy = (i + 0.5) / fmap_size
                
                # Default box with aspect ratio 1
                scale = scales[k]
                default_boxes.append([cx, cy, scale, scale])
                
                # Additional box with aspect ratio 1 and larger scale
                if k < len(scales) - 1:
                    scale_next = scales[k + 1]
                    scale_extra = np.sqrt(scale * scale_next)
                    default_boxes.append([cx, cy, scale_extra, scale_extra])
                
                # Boxes with different aspect ratios
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    default_boxes.append([cx, cy, w, h])
                    default_boxes.append([cx, cy, h, w])  # Reciprocal aspect ratio
    
    return torch.tensor(default_boxes, dtype=torch.float32)

def ssd_loss(predictions, targets, default_boxes, alpha=1.0):
    """
    SSD multi-task loss
    """
    cls_predictions, reg_predictions = predictions
    
    # Match default boxes to ground truth
    matched_boxes, matched_labels = match_targets_to_defaults(targets, default_boxes)
    
    # Classification loss (focal loss or cross entropy)
    cls_loss = F.cross_entropy(cls_predictions.view(-1, cls_predictions.size(-1)), 
                              matched_labels.view(-1))
    
    # Regression loss (smooth L1 loss)
    positive_mask = matched_labels > 0
    if positive_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(reg_predictions[positive_mask], 
                                   matched_boxes[positive_mask])
    else:
        reg_loss = torch.tensor(0.0)
    
    return cls_loss + alpha * reg_loss
```

## Modern Object Detection

### YOLOv5/YOLOv8 - State-of-the-Art Real-time Detection

```python
from ultralytics import YOLO

class ModernYOLOTraining:
    def __init__(self, model_size='n'):  # n, s, m, l, x
        """
        Modern YOLO training pipeline
        """
        self.model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
    
    def train_custom_dataset(self, data_yaml_path, epochs=100, imgsz=640):
        """
        Train on custom dataset
        """
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            device='gpu' if torch.cuda.is_available() else 'cpu',
            batch=16,
            workers=8,
            optimizer='AdamW',
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            augment=True,
            mixup=0.1,
            copy_paste=0.1
        )
        return results
    
    def validate(self, data_yaml_path):
        """Validate model performance"""
        return self.model.val(data=data_yaml_path)
    
    def predict(self, source, save_results=True):
        """Run inference"""
        return self.model.predict(source, save=save_results, conf=0.5)
    
    def export(self, format='onnx'):
        """Export model for deployment"""
        return self.model.export(format=format)

# Usage example
def train_yolo_custom():
    """
    Complete YOLO training pipeline
    """
    # Initialize trainer
    trainer = ModernYOLOTraining(model_size='n')  # Nano version for speed
    
    # Train on custom dataset
    results = trainer.train_custom_dataset(
        data_yaml_path='path/to/dataset.yaml',
        epochs=100,
        imgsz=640
    )
    
    # Validate
    val_results = trainer.validate('path/to/dataset.yaml')
    
    # Export for deployment
    trainer.export(format='onnx')
    
    return results, val_results
```

### EfficientDet - Compound Scaling for Detection

**Key Innovation:** Scale backbone, FPN, and prediction head dimensions together.

```python
class EfficientDet(nn.Module):
    """
    EfficientDet implementation with compound scaling
    """
    def __init__(self, num_classes, compound_coef=0):
        super(EfficientDet, self).__init__()
        
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        
        # Scaling parameters
        backbone_scale = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0][compound_coef]
        fpn_scale = [64, 88, 112, 160, 224, 288, 384, 384][compound_coef]
        fpn_layers = [3, 4, 5, 6, 7, 7, 8, 8][compound_coef]
        
        # EfficientNet backbone
        self.backbone = self._create_efficientnet_backbone(backbone_scale)
        
        # BiFPN (Bidirectional Feature Pyramid Network)
        self.bifpn = BiFPN(fpn_scale, fpn_layers)
        
        # Detection heads
        self.classification_head = ClassificationHead(fpn_scale, num_classes)
        self.regression_head = RegressionHead(fpn_scale)
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # BiFPN processing
        pyramid_features = self.bifpn(features)
        
        # Detection heads
        classifications = []
        regressions = []
        
        for feature in pyramid_features:
            cls_output = self.classification_head(feature)
            reg_output = self.regression_head(feature)
            
            classifications.append(cls_output)
            regressions.append(reg_output)
        
        return classifications, regressions

class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network
    """
    def __init__(self, channels, num_layers):
        super(BiFPN, self).__init__()
        
        self.channels = channels
        self.num_layers = num_layers
        
        # Learnable weights for feature fusion
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        # Convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1)
            for _ in range(num_layers * 2)  # Top-down and bottom-up
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [P3, P4, P5, P6, P7]
        """
        # Top-down pathway
        for i in range(len(features) - 2, -1, -1):
            # Weighted feature fusion
            upsampled = F.interpolate(features[i + 1], size=features[i].shape[2:], mode='nearest')
            
            # Weighted combination
            w1_norm = F.relu(self.w1)
            w1_norm = w1_norm / (w1_norm.sum() + 1e-4)
            
            features[i] = w1_norm[0] * features[i] + w1_norm[1] * upsampled
            features[i] = self.conv_layers[i](features[i])
        
        # Bottom-up pathway
        for i in range(1, len(features)):
            downsampled = F.max_pool2d(features[i - 1], kernel_size=3, stride=2, padding=1)
            
            # Weighted combination (if available)
            if i < len(features) - 1:
                w2_norm = F.relu(self.w2)
                w2_norm = w2_norm / (w2_norm.sum() + 1e-4)
                
                features[i] = (w2_norm[0] * features[i] + 
                              w2_norm[1] * downsampled)
                
                if i + 1 < len(features):
                    features[i] += w2_norm[2] * features[i + 1]
            
            features[i] = self.conv_layers[len(features) + i](features[i])
        
        return features
```

## Non-Maximum Suppression (NMS)

### Traditional NMS

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: [N, 4] bounding boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    """
    # Sort by scores
    indices = torch.argsort(scores, descending=True)
    
    keep = []
    
    while len(indices) > 0:
        # Take box with highest score
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[indices[1:]]
        
        ious = iou_batch(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return torch.tensor(keep)

def class_aware_nms(boxes, scores, labels, iou_threshold=0.5):
    """
    Apply NMS separately for each class
    """
    unique_labels = torch.unique(labels)
    keep_all = []
    
    for label in unique_labels:
        mask = labels == label
        if mask.sum() == 0:
            continue
        
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_indices = torch.where(mask)[0]
        
        # Apply NMS for this class
        keep_class = nms(class_boxes, class_scores, iou_threshold)
        
        # Convert back to original indices
        keep_all.extend(class_indices[keep_class].tolist())
    
    return torch.tensor(keep_all)
```

### Soft NMS

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft Non-Maximum Suppression
    
    Instead of removing overlapping boxes, reduce their confidence scores
    """
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        
        if scores[current] < score_threshold:
            break
        
        keep.append(current.item())
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[indices[1:]]
        
        ious = iou_batch(current_box, remaining_boxes).squeeze(0)
        
        # Apply soft suppression
        weights = torch.exp(-(ious ** 2) / sigma)
        scores[indices[1:]] *= weights
        
        # Re-sort by updated scores
        indices = indices[1:]
        sort_idx = torch.argsort(scores[indices], descending=True)
        indices = indices[sort_idx]
    
    return torch.tensor(keep)
```

## Real-World Applications

### 1. Autonomous Driving

```python
class AutonomousDrivingDetector:
    """
    Object detection for autonomous driving
    """
    def __init__(self):
        # Load specialized model for driving scenarios
        self.model = YOLO('yolov8n.pt')  # Replace with driving-specific model
        
        # Driving-specific classes
        self.driving_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic_light', 11: 'stop_sign'
        }
    
    def detect_objects(self, frame):
        """Detect objects in driving scene"""
        results = self.model(frame)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in self.driving_classes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        
                        detections.append({
                            'class': self.driving_classes[class_id],
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'distance': self.estimate_distance(y2 - y1)  # Rough estimate
                        })
        
        return detections
    
    def estimate_distance(self, box_height):
        """Rough distance estimation based on bounding box height"""
        # This is a simplified estimation
        # Real systems use stereo vision, lidar, or calibrated cameras
        reference_height = 100  # pixels for object at 10 meters
        reference_distance = 10  # meters
        
        if box_height > 0:
            estimated_distance = (reference_height * reference_distance) / box_height
            return min(estimated_distance, 100)  # Cap at 100 meters
        return 100
    
    def process_driving_frame(self, frame):
        """Complete processing pipeline for driving"""
        detections = self.detect_objects(frame)
        
        # Filter by relevance for driving
        critical_objects = []
        for detection in detections:
            if detection['class'] in ['person', 'car', 'truck', 'bicycle', 'motorcycle']:
                if detection['distance'] < 50:  # Objects within 50 meters
                    critical_objects.append(detection)
        
        return critical_objects

# Usage for autonomous driving
def driving_detection_pipeline(video_path):
    """Process driving video for object detection"""
    detector = AutonomousDrivingDetector()
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.process_driving_frame(frame)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            label = f"{detection['class']}: {detection['distance']:.1f}m"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Driving Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### 2. Security and Surveillance

```python
class SecurityDetector:
    """
    Object detection for security applications
    """
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.person_tracker = {}  # Simple tracking
        self.alert_zones = []  # Restricted areas
        
    def define_alert_zone(self, points):
        """Define restricted area using polygon points"""
        self.alert_zones.append(np.array(points))
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def detect_intrusion(self, frame):
        """Detect people in restricted areas"""
        results = self.model(frame)
        
        alerts = []
        persons = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        
                        # Calculate person center point
                        center_x = (x1 + x2) / 2
                        center_y = y2  # Bottom of bounding box
                        
                        persons.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': (center_x, center_y),
                            'confidence': confidence
                        })
                        
                        # Check if person is in restricted area
                        for zone_idx, zone in enumerate(self.alert_zones):
                            if self.point_in_polygon((center_x, center_y), zone):
                                alerts.append({
                                    'type': 'intrusion',
                                    'zone': zone_idx,
                                    'person': (center_x, center_y),
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': confidence
                                })
        
        return persons, alerts
    
    def draw_security_overlay(self, frame, persons, alerts):
        """Draw security monitoring overlay"""
        # Draw alert zones
        for zone in self.alert_zones:
            cv2.polylines(frame, [zone.astype(int)], True, (0, 0, 255), 2)
            cv2.fillPoly(frame, [zone.astype(int)], (0, 0, 255, 50))
        
        # Draw person detections
        for person in persons:
            x1, y1, x2, y2 = map(int, person['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(person['center'][0]), int(person['center'][1])), 
                      5, (0, 255, 0), -1)
        
        # Draw alerts
        for alert in alerts:
            x1, y1, x2, y2 = map(int, alert['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, 'ALERT: INTRUSION', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

# Usage for security monitoring
def security_monitoring_system(camera_source=0):
    """Real-time security monitoring"""
    detector = SecurityDetector()
    
    # Define restricted area (example: center of frame)
    detector.define_alert_zone([(200, 200), (400, 200), (400, 400), (200, 400)])
    
    cap = cv2.VideoCapture(camera_source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect intrusions
        persons, alerts = detector.detect_intrusion(frame)
        
        # Draw overlay
        frame = detector.draw_security_overlay(frame, persons, alerts)
        
        # Show alerts
        if alerts:
            print(f"SECURITY ALERT: {len(alerts)} intrusion(s) detected!")
        
        cv2.imshow('Security Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Next Steps

Object detection mastery opens doors to:

1. **Instance Segmentation**: Pixel-level object boundaries
2. **Panoptic Segmentation**: Combining semantic and instance segmentation
3. **3D Object Detection**: Extending to three-dimensional space
4. **Video Object Detection**: Temporal consistency and tracking
5. **Multi-modal Detection**: Combining vision with other sensors

Object detection serves as the foundation for many complex computer vision applications, from autonomous vehicles to medical imaging systems.
