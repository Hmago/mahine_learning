# Computer Vision Exercises 🏋️‍♀️

Welcome to hands-on computer vision practice! These exercises progress from beginner to advanced, giving you practical experience with real problems.

## 🎯 Exercise Structure

Each exercise includes:
- **Objective**: What you'll learn
- **Difficulty**: Beginner, Intermediate, or Advanced  
- **Tools**: Required libraries and data
- **Steps**: Guided implementation
- **Extensions**: Ideas to go further

## 📚 Exercise Categories

### 1. Image Processing Fundamentals

#### Exercise 1.1: Build a Photo Editor (Beginner)
**Objective**: Create a simple photo editing application

**What you'll build**:
- Load and display images
- Apply filters (blur, sharpen, brightness)
- Save processed images

**Implementation**:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SimplePhotoEditor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
    
    def load_image(self, path):
        # TODO: Implement image loading
        pass
    
    def apply_blur(self, kernel_size=5):
        # TODO: Apply Gaussian blur
        pass
    
    def adjust_brightness(self, value=0):
        # TODO: Adjust image brightness
        pass
    
    def save_image(self, path):
        # TODO: Save current image
        pass

# Your task: Complete the implementation
editor = SimplePhotoEditor()
```

**Extensions**:
- Add more filters (sepia, vintage, etc.)
- Create a GUI interface
- Batch process multiple images

#### Exercise 1.2: Image Quality Analyzer (Intermediate)
**Objective**: Analyze and score image quality

**What you'll build**:
- Blur detection using Laplacian variance
- Brightness and contrast analysis
- Noise level estimation
- Overall quality score

**Implementation**:
```python
def analyze_image_quality(image_path):
    """
    Analyze image quality and return metrics
    
    Returns:
        dict: Quality metrics including blur, brightness, contrast, noise
    """
    # TODO: Implement quality analysis
    # Hints:
    # - Use cv2.Laplacian() for blur detection
    # - Calculate histogram for brightness/contrast
    # - Use standard deviation for noise estimation
    pass

def batch_quality_analysis(image_folder):
    """Analyze quality of all images in a folder"""
    # TODO: Process multiple images and generate report
    pass
```

**Extensions**:
- Create quality improvement suggestions
- Automatic quality enhancement
- Compare different image formats

### 2. Feature Detection & Matching

#### Exercise 2.1: Panorama Creator (Intermediate)
**Objective**: Stitch multiple images into a panorama

**What you'll build**:
- Feature detection and matching
- Homography estimation
- Image warping and blending

**Implementation**:
```python
class PanoramaStitcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def find_features(self, image):
        # TODO: Detect keypoints and descriptors
        pass
    
    def match_features(self, desc1, desc2):
        # TODO: Match features between images
        pass
    
    def estimate_homography(self, matches, kp1, kp2):
        # TODO: Calculate transformation matrix
        pass
    
    def stitch_images(self, img1, img2):
        # TODO: Combine images into panorama
        pass

# Your task: Create panorama from 2-3 overlapping images
stitcher = PanoramaStitcher()
```

**Extensions**:
- Handle multiple images (3+ images)
- Automatic exposure blending
- Cylindrical or spherical projection

#### Exercise 2.2: Object Tracker (Advanced)
**Objective**: Track objects across video frames

**What you'll build**:
- Feature-based tracking
- Kalman filter for prediction
- Handling occlusion and re-detection

**Implementation**:
```python
class FeatureTracker:
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 0
    
    def detect_features(self, frame):
        # TODO: Detect features in current frame
        pass
    
    def match_with_tracks(self, features):
        # TODO: Match current features with existing tracks
        pass
    
    def update_tracks(self, matches):
        # TODO: Update track positions
        pass
    
    def create_new_tracks(self, unmatched_features):
        # TODO: Start new tracks for unmatched features
        pass

# Your task: Implement complete tracking system
tracker = FeatureTracker()
```

### 3. Object Detection & Recognition

#### Exercise 3.1: Custom Object Detector (Intermediate)
**Objective**: Train a detector for your own objects

**What you'll build**:
- Data collection and annotation
- Model training with transfer learning
- Evaluation and optimization

**Implementation**:
```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class CustomObjectDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        # TODO: Initialize dataset
        pass
    
    def __len__(self):
        # TODO: Return dataset size
        pass
    
    def __getitem__(self, idx):
        # TODO: Return image and annotations
        pass

class ObjectDetectorTrainer:
    def __init__(self, num_classes):
        # TODO: Initialize model and training components
        pass
    
    def train_epoch(self, dataloader):
        # TODO: Train for one epoch
        pass
    
    def validate(self, dataloader):
        # TODO: Evaluate model performance
        pass

# Your task: Collect data and train detector
trainer = ObjectDetectorTrainer(num_classes=3)
```

**Extensions**:
- Data augmentation strategies
- Advanced evaluation metrics
- Model deployment optimization

#### Exercise 3.2: Real-time Detection System (Advanced)
**Objective**: Build a production-ready detection system

**What you'll build**:
- Real-time video processing
- Performance monitoring
- Alert system for specific objects

**Implementation**:
```python
class RealTimeDetectionSystem:
    def __init__(self, model_path, alert_classes):
        # TODO: Initialize system components
        self.model = None
        self.alert_classes = alert_classes
        self.performance_monitor = {}
    
    def process_frame(self, frame):
        # TODO: Run detection on single frame
        pass
    
    def check_alerts(self, detections):
        # TODO: Check for alert-worthy detections
        pass
    
    def update_performance_metrics(self, processing_time):
        # TODO: Track system performance
        pass
    
    def run_detection_loop(self, video_source):
        # TODO: Main detection loop
        pass

# Your task: Build complete system with monitoring
detection_system = RealTimeDetectionSystem('model.pt', ['person', 'car'])
```

### 4. Advanced Computer Vision

#### Exercise 4.1: Semantic Segmentation (Advanced)
**Objective**: Implement pixel-level scene understanding

**What you'll build**:
- Custom segmentation model
- Training pipeline
- Evaluation with IoU metrics

**Implementation**:
```python
import segmentation_models_pytorch as smp

class SegmentationTrainer:
    def __init__(self, num_classes, architecture='unet'):
        # TODO: Initialize segmentation model
        pass
    
    def calculate_iou(self, pred, target):
        # TODO: Calculate Intersection over Union
        pass
    
    def train_model(self, train_loader, val_loader, epochs):
        # TODO: Implement training loop
        pass
    
    def visualize_predictions(self, image, prediction, ground_truth):
        # TODO: Create visualization of results
        pass

# Your task: Train segmentation model on cityscapes or custom data
trainer = SegmentationTrainer(num_classes=19)
```

#### Exercise 4.2: Model Optimization Pipeline (Advanced)
**Objective**: Optimize model for edge deployment

**What you'll build**:
- Quantization pipeline
- Pruning implementation
- Performance benchmarking

**Implementation**:
```python
import torch.quantization as quantization

class ModelOptimizer:
    def __init__(self, model):
        self.original_model = model
        self.optimized_models = {}
    
    def quantize_model(self):
        # TODO: Apply dynamic quantization
        pass
    
    def prune_model(self, pruning_amount=0.3):
        # TODO: Apply magnitude-based pruning
        pass
    
    def benchmark_models(self, test_data):
        # TODO: Compare performance of different models
        pass
    
    def export_for_deployment(self, model, format='onnx'):
        # TODO: Export model for production
        pass

# Your task: Optimize model for mobile deployment
optimizer = ModelOptimizer(your_trained_model)
```

## 🏆 Challenge Projects

### Challenge 1: Smart Parking System
**Difficulty**: Advanced
**Objective**: Build an automated parking monitoring system

**Requirements**:
- Detect available parking spaces
- Track occupancy over time
- Generate analytics dashboard
- Handle different weather conditions

### Challenge 2: Medical Image Analyzer
**Difficulty**: Advanced
**Objective**: Assist in medical diagnosis

**Requirements**:
- Detect abnormalities in medical images
- Provide confidence scores
- Generate detailed reports
- Ensure high precision (minimize false positives)

### Challenge 3: Augmented Reality Application
**Difficulty**: Expert
**Objective**: Create AR experience with object recognition

**Requirements**:
- Real-time object detection
- 3D pose estimation
- Virtual object placement
- Smooth user experience (60+ FPS)

## 📊 Self-Assessment Rubric

### Beginner Level (Exercises 1.1, 1.2)
- [ ] Can load and manipulate images
- [ ] Understands basic image properties
- [ ] Applies filters and transformations
- [ ] Creates simple image processing tools

### Intermediate Level (Exercises 2.1, 2.2, 3.1)
- [ ] Implements feature detection and matching
- [ ] Builds object detection systems
- [ ] Handles real-world data challenges
- [ ] Evaluates model performance

### Advanced Level (Exercises 3.2, 4.1, 4.2)
- [ ] Develops production-ready systems
- [ ] Optimizes for performance and deployment
- [ ] Handles edge cases and failures
- [ ] Integrates multiple AI components

### Expert Level (Challenge Projects)
- [ ] Solves complex, real-world problems
- [ ] Considers user experience and system design
- [ ] Implements robust error handling
- [ ] Creates scalable, maintainable code

## 🛠️ Setup Instructions

### Required Libraries
```bash
# Core computer vision
pip install opencv-python
pip install pillow
pip install matplotlib
pip install numpy

# Deep learning
pip install torch torchvision
pip install ultralytics
pip install detectron2

# Specialized tools
pip install segmentation-models-pytorch
pip install albumentations
pip install scikit-image
```

### Data Sources
- **COCO Dataset**: Object detection and segmentation
- **ImageNet**: Image classification
- **Cityscapes**: Urban scene segmentation
- **Your own data**: Collect using phone camera

### Development Environment
- **Jupyter Notebook**: For experimentation
- **VS Code**: For project development
- **Google Colab**: For GPU access
- **Local GPU**: NVIDIA with CUDA support

## 🎯 Tips for Success

### Getting Started
1. Start with simpler exercises and build up
2. Focus on understanding concepts before optimization
3. Visualize intermediate results frequently
4. Test with your own images and data

### Debugging Computer Vision Code
1. Always display intermediate results
2. Check image dimensions and data types
3. Verify color channel ordering (RGB vs BGR)
4. Monitor memory usage with large datasets

### Best Practices
1. Version control your experiments
2. Document your approach and findings
3. Create reusable utility functions
4. Profile performance bottlenecks

## 🚀 Taking It Further

### Contributing to Open Source
- OpenCV contributions
- PyTorch computer vision utilities
- Detectron2 model implementations
- Create your own computer vision library

### Building a Portfolio
- Document your projects on GitHub
- Create demo videos and presentations
- Write blog posts about your learnings
- Participate in computer vision competitions

### Continuous Learning
- Follow latest research papers
- Attend computer vision conferences
- Join online communities and forums
- Experiment with cutting-edge techniques

Remember: The best way to learn computer vision is by doing! Start with the exercises that match your current level and gradually work your way up. Each exercise builds skills you'll use in real-world applications. 🎯✨
