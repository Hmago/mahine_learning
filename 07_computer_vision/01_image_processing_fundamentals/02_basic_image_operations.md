# Basic Image Operations: Your First Digital Toolkit 🛠️

## 🎯 What You'll Master

Think of this as learning to use digital "power tools" for images. Just like a carpenter has a saw, drill, and hammer, you'll master the fundamental operations that every computer vision system uses.

By the end of this section, you'll confidently:

- Resize and reshape images for any purpose
- Rotate and flip images in any direction  
- Enhance image quality through filtering
- Understand when and why to use each operation

## 🔧 Geometric Transformations: Moving Things Around

### 1. Resizing: Making Images Fit

**The Big Idea**: Change the size of an image while keeping its content recognizable.

**Real-world analogy**: Like enlarging or shrinking a photo to fit different picture frames.

#### Basic Resizing

```python
import cv2
import matplotlib.pyplot as plt

def resize_image_demo(image_path):
    """Demonstrate different resizing techniques"""
    
    # Load original image
    image = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Original size: {image.shape[:2]}")
    
    # Different resize methods
    sizes = [
        (400, 300),    # Fixed size
        (800, 600),    # Larger fixed size
        (200, 150),    # Smaller fixed size
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original: {image.shape[1]}x{image.shape[0]}')
    
    # Show different sizes
    for i, (width, height) in enumerate(sizes):
        resized = cv2.resize(image, (width, height))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        axes[i+1].imshow(resized_rgb)
        axes[i+1].set_title(f'Resized: {width}x{height}')
    
    plt.tight_layout()
    plt.show()

# Try it out!
resize_image_demo('your_image.jpg')
```

#### Smart Resizing: Keeping Proportions

```python
def resize_keeping_aspect_ratio(image, target_width):
    """Resize image while maintaining aspect ratio"""
    
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    # Calculate new height based on target width
    target_height = int(target_width / aspect_ratio)
    
    resized = cv2.resize(image, (target_width, target_height))
    
    print(f"Original: {width}x{height}")
    print(f"Resized: {target_width}x{target_height}")
    print(f"Aspect ratio preserved: {width/height:.2f} → {target_width/target_height:.2f}")
    
    return resized

# Example: Resize to width=500 while keeping proportions
smart_resized = resize_keeping_aspect_ratio(image, 500)
```

**Why This Matters**: 
- **Web development**: Images must fit different screen sizes
- **Machine learning**: Models expect consistent input sizes
- **Storage optimization**: Smaller images = faster loading

### 2. Rotation: Spinning Things Around

**The Big Idea**: Rotate images by any angle while handling the mathematical complexities automatically.

```python
def rotation_demo(image):
    """Show different rotation angles and methods"""
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    angles = [0, 45, 90, 180, 270]
    
    fig, axes = plt.subplots(1, len(angles), figsize=(20, 4))
    
    for i, angle in enumerate(angles):
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(rotated_rgb)
        axes[i].set_title(f'{angle}°')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

rotation_demo(image)
```

#### Smart Rotation: No Cropping

```python
def rotate_and_expand(image, angle):
    """Rotate image and expand canvas to show everything"""
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation with expanded canvas
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated

# Example: Rotate 45 degrees without losing any part of the image
expanded_rotation = rotate_and_expand(image, 45)
```

### 3. Flipping: Mirror Images

**The Big Idea**: Create mirror images horizontally or vertically.

```python
def flipping_demo(image):
    """Demonstrate different types of flipping"""
    
    # Different flip types
    horizontal_flip = cv2.flip(image, 1)  # Flip around y-axis
    vertical_flip = cv2.flip(image, 0)    # Flip around x-axis  
    both_flip = cv2.flip(image, -1)       # Flip around both axes
    
    images = [
        (image, 'Original'),
        (horizontal_flip, 'Horizontal Flip'),
        (vertical_flip, 'Vertical Flip'),
        (both_flip, 'Both Flips')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

flipping_demo(image)
```

**When to Use Flipping**:
- **Data augmentation**: Increase training data for machine learning
- **Symmetry analysis**: Check if objects are symmetric
- **User interfaces**: Mirror text in right-to-left languages

## 🎨 Image Enhancement: Making Things Look Better

### 1. Brightness and Contrast: The Basics

**Brightness**: How light or dark the overall image is
**Contrast**: The difference between light and dark areas

```python
def brightness_contrast_demo(image):
    """Adjust brightness and contrast with visual feedback"""
    
    # Different adjustments
    adjustments = [
        (0, 1.0),     # Original
        (50, 1.0),    # Brighter
        (-50, 1.0),   # Darker
        (0, 1.5),     # Higher contrast
        (0, 0.5),     # Lower contrast
        (30, 1.3),    # Brighter + higher contrast
    ]
    
    titles = [
        'Original',
        'Brighter (+50)',
        'Darker (-50)', 
        'High Contrast (1.5x)',
        'Low Contrast (0.5x)',
        'Bright + Contrast'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (brightness, contrast) in enumerate(adjustments):
        # Apply brightness and contrast
        # Formula: new_pixel = contrast * old_pixel + brightness
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(adjusted_rgb)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

brightness_contrast_demo(image)
```

#### Automatic Enhancement

```python
def auto_enhance_image(image):
    """Automatically enhance image using histogram equalization"""
    
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

# Compare original vs enhanced
enhanced = auto_enhance_image(image)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
axes[1].set_title('Auto Enhanced')
plt.show()
```

### 2. Filtering: Removing Noise and Enhancing Features

#### Gaussian Blur: The Smooth Operator

**What it does**: Reduces noise and details, creates smooth images
**When to use**: Preprocessing for edge detection, reducing camera shake

```python
def gaussian_blur_demo(image):
    """Show different levels of Gaussian blur"""
    
    kernel_sizes = [1, 5, 15, 31, 51]  # Must be odd numbers
    
    fig, axes = plt.subplots(1, len(kernel_sizes), figsize=(20, 4))
    
    for i, kernel_size in enumerate(kernel_sizes):
        if kernel_size == 1:
            blurred = image  # Original (no blur)
        else:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        axes[i].imshow(blurred_rgb)
        axes[i].set_title(f'Kernel: {kernel_size}x{kernel_size}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

gaussian_blur_demo(image)
```

#### Sharpening: Bringing Out Details

```python
def sharpen_image(image):
    """Apply sharpening filter to enhance edges"""
    
    # Create sharpening kernel
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    
    return sharpened

# Compare original vs sharpened
sharpened = sharpen_image(image)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
axes[1].set_title('Sharpened')
plt.show()
```

### 3. Edge Detection: Finding Boundaries

**The Big Idea**: Identify where objects begin and end by finding rapid changes in brightness.

```python
def edge_detection_demo(image):
    """Compare different edge detection methods"""
    
    # Convert to grayscale first
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Different edge detection methods
    edges_canny = cv2.Canny(gray, 50, 150)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Convert to uint8 for display
    edges_sobel_x = np.uint8(np.absolute(edges_sobel_x))
    edges_sobel_y = np.uint8(np.absolute(edges_sobel_y))
    edges_laplacian = np.uint8(np.absolute(edges_laplacian))
    
    # Display results
    images = [
        (gray, 'Original Grayscale'),
        (edges_canny, 'Canny Edge Detection'),
        (edges_sobel_x, 'Sobel X (Vertical Edges)'),
        (edges_sobel_y, 'Sobel Y (Horizontal Edges)'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

edge_detection_demo(image)
```

## 🏗️ Building Your First Image Processing Pipeline

Let's combine everything into a complete image processing pipeline:

```python
class ImageProcessor:
    """A complete image processing toolkit"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.history = []
    
    def load_image(self, image_path):
        """Load and initialize image"""
        self.original_image = cv2.imread(image_path)
        self.processed_image = self.original_image.copy()
        self.history = ['Original loaded']
        print(f"✅ Image loaded: {self.original_image.shape}")
        return self
    
    def resize(self, width=None, height=None, keep_aspect=True):
        """Resize image with smart defaults"""
        if width is None and height is None:
            print("❌ Please specify width or height")
            return self
        
        current_h, current_w = self.processed_image.shape[:2]
        
        if keep_aspect:
            if width and not height:
                height = int(width * current_h / current_w)
            elif height and not width:
                width = int(height * current_w / current_h)
        
        self.processed_image = cv2.resize(self.processed_image, (width, height))
        self.history.append(f'Resized to {width}x{height}')
        print(f"✅ Resized to: {width}x{height}")
        return self
    
    def rotate(self, angle, expand_canvas=True):
        """Rotate image by specified angle"""
        h, w = self.processed_image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if expand_canvas:
            # Calculate new dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_w = int((h * sin_angle) + (w * cos_angle))
            new_h = int((h * cos_angle) + (w * sin_angle))
            
            # Adjust rotation matrix
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            self.processed_image = cv2.warpAffine(self.processed_image, rotation_matrix, (new_w, new_h))
        else:
            self.processed_image = cv2.warpAffine(self.processed_image, rotation_matrix, (w, h))
        
        self.history.append(f'Rotated by {angle}°')
        print(f"✅ Rotated by: {angle}°")
        return self
    
    def enhance(self, brightness=0, contrast=1.0):
        """Adjust brightness and contrast"""
        self.processed_image = cv2.convertScaleAbs(
            self.processed_image, 
            alpha=contrast, 
            beta=brightness
        )
        self.history.append(f'Enhanced: brightness={brightness}, contrast={contrast}')
        print(f"✅ Enhanced: brightness={brightness}, contrast={contrast}")
        return self
    
    def blur(self, kernel_size=5):
        """Apply Gaussian blur"""
        self.processed_image = cv2.GaussianBlur(
            self.processed_image, 
            (kernel_size, kernel_size), 
            0
        )
        self.history.append(f'Blurred with kernel size {kernel_size}')
        print(f"✅ Applied blur: kernel size {kernel_size}")
        return self
    
    def sharpen(self):
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
        self.history.append('Applied sharpening')
        print("✅ Applied sharpening filter")
        return self
    
    def show_comparison(self):
        """Display original vs processed"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed
        axes[1].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show processing history
        print("\n📋 Processing History:")
        for step in self.history:
            print(f"  • {step}")
    
    def save(self, output_path, quality=95):
        """Save processed image"""
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            cv2.imwrite(output_path, self.processed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(output_path, self.processed_image)
        
        print(f"✅ Saved to: {output_path}")
        return self

# Example usage - Method chaining for elegant processing
processor = ImageProcessor()
processor.load_image('input.jpg') \
         .resize(width=800) \
         .rotate(15) \
         .enhance(brightness=20, contrast=1.1) \
         .sharpen() \
         .show_comparison() \
         .save('processed_output.jpg')
```

## 🎯 Practice Challenges

### Challenge 1: Instagram Filter Creator

Create your own Instagram-style filters:

```python
def vintage_filter(image):
    """Create a vintage/sepia effect"""
    # Your implementation here
    pass

def cool_filter(image):
    """Create a cool blue tint"""
    # Your implementation here  
    pass

def warm_filter(image):
    """Create a warm orange tint"""
    # Your implementation here
    pass
```

### Challenge 2: Photo Batch Processor

Build a tool that processes multiple photos:

- Resizes all images to the same dimensions
- Applies consistent enhancement
- Saves with organized naming

### Challenge 3: Smart Crop Detector

Create a function that automatically crops images to focus on the most interesting part (highest edge density).

## 🌍 Real-World Applications

### Social Media Platforms

```python
def prepare_for_instagram(image):
    """Prepare image for Instagram posting"""
    
    # Instagram prefers 1080x1080 square images
    # Resize while maintaining aspect ratio, then center crop
    
    h, w = image.shape[:2]
    
    # Resize so the smaller dimension is 1080
    if h < w:
        new_h, new_w = 1080, int(1080 * w / h)
    else:
        new_h, new_w = int(1080 * h / w), 1080
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Center crop to 1080x1080
    start_x = (new_w - 1080) // 2
    start_y = (new_h - 1080) // 2
    cropped = resized[start_y:start_y+1080, start_x:start_x+1080]
    
    # Apply subtle enhancement
    enhanced = cv2.convertScaleAbs(cropped, alpha=1.1, beta=10)
    
    return enhanced
```

### Document Scanning

```python
def enhance_document_scan(image):
    """Enhance scanned document for better readability"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduce noise
    denoised = cv2.medianBlur(gray, 3)
    
    # Increase contrast using adaptive threshold
    enhanced = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return enhanced
```

### Product Photography

```python
def enhance_product_photo(image):
    """Enhance product photos for e-commerce"""
    
    # Increase saturation for more vibrant colors
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # Increase saturation
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened
```

## 📊 Performance Considerations

### Memory Management

```python
def process_large_image_efficiently(image_path):
    """Process large images without running out of memory"""
    
    # Read image properties without loading full image
    import os
    file_size = os.path.getsize(image_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Could not load image")
        return None
    
    # Check memory requirements
    memory_needed = image.nbytes / (1024*1024)
    print(f"Memory needed: {memory_needed:.1f} MB")
    
    # Resize if too large (e.g., > 100MB)
    if memory_needed > 100:
        print("⚠️  Large image detected, resizing for efficiency...")
        scale_factor = (100 / memory_needed) ** 0.5
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (new_width, new_height))
        print(f"✅ Resized to: {new_width}x{new_height}")
    
    return image
```

### Speed Optimization

```python
import time

def benchmark_operations(image):
    """Compare performance of different operations"""
    
    operations = {
        'Resize': lambda img: cv2.resize(img, (800, 600)),
        'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (15, 15), 0),
        'Edge Detection': lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150),
        'Rotation': lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 45, 1), (img.shape[1], img.shape[0]))
    }
    
    for name, operation in operations.items():
        start_time = time.time()
        result = operation(image)
        end_time = time.time()
        
        print(f"{name}: {(end_time - start_time)*1000:.1f} ms")
```

## 🧠 Key Takeaways

### What You've Learned

1. **Geometric Transformations**: Resize, rotate, flip images while preserving quality

2. **Enhancement Techniques**: Adjust brightness, contrast, and apply filters intelligently

3. **Edge Detection**: Find important boundaries and features in images

4. **Pipeline Thinking**: Chain operations together for complex transformations

5. **Real-world Applications**: Apply techniques to solve actual problems

### Mental Models

- **Transformation Pipeline**: Like an assembly line, each operation modifies the image
- **Quality vs Speed Trade-offs**: Higher quality operations take more time
- **Preprocessing Importance**: Clean, standardized images lead to better results
- **Chain of Operations**: The order of operations matters for final results

## 🚀 What's Next?

You now have a solid toolkit for basic image operations! In the next module, we'll learn how to automatically **find and describe important features** in images - the foundation for object recognition and tracking.

You'll learn to identify:
- Corners and edges that define object shapes
- Unique patterns that distinguish different objects  
- How to match features between different images

This is where we transition from manual image editing to intelligent computer vision! 🎯✨

## 💡 Quick Reference Cheat Sheet

```python
# Geometric Operations
resized = cv2.resize(image, (width, height))
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
flipped = cv2.flip(image, 1)  # 1=horizontal, 0=vertical, -1=both

# Enhancement
bright = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
edges = cv2.Canny(gray_image, low_threshold, high_threshold)

# Useful snippets
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

Keep experimenting and building! 🛠️
