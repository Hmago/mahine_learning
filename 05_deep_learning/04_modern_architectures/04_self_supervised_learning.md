# Self-Supervised Learning: Learning Without Labels

Master the art of learning from unlabeled data! Discover how modern AI systems learn powerful representations by solving cleverly designed pretext tasks, revolutionizing computer vision and NLP.

## 🎯 What You'll Master

- **Self-Supervised Fundamentals**: Understanding pretext and downstream tasks
- **Computer Vision SSL**: Contrastive learning, SimCLR, SwAV, DINO
- **NLP SSL**: BERT, GPT, and masked language modeling
- **Implementation**: Complete SSL frameworks from scratch

## 📚 The Self-Supervised Revolution

### Learning Without Labels: The Power of Pretext Tasks

**Traditional Supervised Learning:**

```text
Image + Label → Model → Prediction
```

**Self-Supervised Learning:**

```text
Unlabeled Data → Pretext Task → Representations → Downstream Task
```

**Key Insight:**
Create "fake" supervised tasks from unlabeled data to learn useful representations. It's like learning to understand language by filling in missing words, or understanding images by predicting rotations!

**Real-World Analogy:**
Think of learning to drive:
- **Pretext Task**: Practice parking, parallel parking, three-point turns
- **Learned Skill**: Spatial awareness, car control, traffic understanding  
- **Downstream Task**: Navigate to work, drive in different cities
- **Magic**: Skills from practice transfer to real driving scenarios!

## 1. Contrastive Learning Fundamentals

### The Core Idea: Pull Similar Together, Push Different Apart

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import math

class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels=None):
        """
        Args:
            features: (batch_size, feature_dim) - normalized features
            labels: (batch_size,) - for supervised contrastive learning
        """
        batch_size = features.shape[0]
        
        if labels is None:
            # Self-supervised: each sample is its own class
            labels = torch.arange(batch_size, device=features.device)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean log likelihood for positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        
        # Loss is negative log likelihood
        loss = -mean_log_prob.mean()
        
        return loss

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query, positive, negatives):
        """
        Args:
            query: (batch_size, feature_dim)
            positive: (batch_size, feature_dim) 
            negatives: (batch_size, num_negatives, feature_dim)
        """
        batch_size = query.shape[0]
        
        # Compute positive similarity
        pos_sim = torch.sum(query * positive, dim=1) / self.temperature
        
        # Compute negative similarities
        neg_sim = torch.bmm(negatives, query.unsqueeze(2)).squeeze(2) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

# Data Augmentation for Contrastive Learning
class ContrastiveTransform:
    """Contrastive data augmentation pipeline"""
    
    def __init__(self, size=224):
        # Strong augmentations for contrastive learning
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        # Return two augmented views
        return self.transform(x), self.transform(x)

# SimCLR Implementation
class SimCLREncoder(nn.Module):
    """SimCLR encoder with ResNet backbone"""
    
    def __init__(self, backbone='resnet50', feature_dim=128):
        super().__init__()
        
        # Load pre-trained ResNet backbone
        if backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=False)
            backbone_dim = 2048
        elif backbone == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=False)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, feature_dim)
        )
    
    def forward(self, x):
        # Extract features
        h = self.backbone(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Project to contrastive space
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)  # L2 normalize
        
        return h, z

class SimCLR:
    """SimCLR framework for self-supervised learning"""
    
    def __init__(self, encoder, temperature=0.07, device='cuda'):
        self.encoder = encoder.to(device)
        self.device = device
        self.temperature = temperature
        self.criterion = ContrastiveLoss(temperature)
        
    def train_step(self, batch, optimizer):
        """Single training step for SimCLR"""
        
        # Get two augmented views
        (x1, x2), _ = batch  # Ignore labels in self-supervised setting
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        batch_size = x1.size(0)
        
        # Get representations
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Create similarity matrix
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        labels = labels - torch.eye(2 * batch_size, device=self.device)
        
        # Compute loss
        loss = self.contrastive_loss(sim_matrix, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def contrastive_loss(self, sim_matrix, labels):
        """Compute contrastive loss"""
        exp_sim = torch.exp(sim_matrix)
        
        # Sum of positive pairs
        pos_sim = (exp_sim * labels).sum(dim=1)
        
        # Sum of all pairs (excluding diagonal)
        mask = torch.ones_like(sim_matrix) - torch.eye(sim_matrix.size(0), device=self.device)
        all_sim = (exp_sim * mask).sum(dim=1)
        
        # Contrastive loss
        loss = -torch.log(pos_sim / all_sim).mean()
        return loss

# SwAV Implementation
class SwAV(nn.Module):
    """SwAV: Swapping Assignments between Views"""
    
    def __init__(self, encoder, num_prototypes=3000, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Prototype layer
        feature_dim = encoder.projection_head[-1].out_features
        self.prototypes = nn.Linear(feature_dim, num_prototypes, bias=False)
        
    def forward(self, x):
        # Get features
        _, z = self.encoder(x)
        
        # Compute prototype assignments
        scores = self.prototypes(z) / self.temperature
        
        return z, scores
    
    def sinkhorn_knopp(self, scores, num_iters=3):
        """Sinkhorn-Knopp algorithm for optimal transport"""
        
        # Convert to probabilities
        Q = torch.exp(scores).T  # (num_prototypes, batch_size)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            # Normalize rows
            Q = Q / Q.sum(dim=1, keepdim=True)
            
            # Normalize columns  
            Q = Q / Q.sum(dim=0, keepdim=True)
        
        return Q.T

# DINO (Self-Distillation with No Labels)
class DINO(nn.Module):
    """DINO: Self-supervised learning with Vision Transformers"""
    
    def __init__(self, student, teacher, temperature=0.04, center_momentum=0.9):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.center_momentum = center_momentum
        
        # Initialize teacher with student weights
        for p_student, p_teacher in zip(student.parameters(), teacher.parameters()):
            p_teacher.data.copy_(p_student.data)
            p_teacher.requires_grad = False
        
        # Center for teacher outputs
        self.register_buffer('center', torch.zeros(student.projection_head[-1].out_features))
    
    def forward(self, global_crops, local_crops):
        """
        Args:
            global_crops: List of global augmented views
            local_crops: List of local augmented views
        """
        # Student processes all crops
        student_outputs = []
        for crop in global_crops + local_crops:
            _, output = self.student(crop)
            student_outputs.append(output)
        
        # Teacher processes only global crops
        teacher_outputs = []
        with torch.no_grad():
            for crop in global_crops:
                _, output = self.teacher(crop)
                teacher_outputs.append(output)
        
        return student_outputs, teacher_outputs
    
    def update_teacher(self, momentum=0.996):
        """Update teacher with momentum"""
        with torch.no_grad():
            for p_student, p_teacher in zip(self.student.parameters(), self.teacher.parameters()):
                p_teacher.data = momentum * p_teacher.data + (1 - momentum) * p_student.data
    
    def update_center(self, teacher_outputs):
        """Update center for teacher outputs"""
        with torch.no_grad():
            center = torch.cat(teacher_outputs).mean(dim=0)
            self.center = self.center_momentum * self.center + (1 - self.center_momentum) * center

# Masked Autoencoder (MAE)
class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for self-supervised learning"""
    
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def random_masking(self, x, mask_ratio):
        """Random masking for patches"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Random shuffle
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # Encode with masking
        encoded, mask, ids_restore = self.encode_with_mask(x)
        
        # Decode
        reconstructed = self.decoder(encoded, ids_restore)
        
        return reconstructed, mask
    
    def encode_with_mask(self, x):
        # Convert to patches (simplified)
        patches = self.patchify(x)
        
        # Random masking
        masked_patches, mask, ids_restore = self.random_masking(patches, self.mask_ratio)
        
        # Encode visible patches
        encoded = self.encoder(masked_patches)
        
        return encoded, mask, ids_restore
    
    def patchify(self, imgs):
        """Convert images to patches"""
        # Simplified patch extraction
        p = 16  # patch size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x

# Barlow Twins
class BarlowTwins(nn.Module):
    """Barlow Twins self-supervised learning"""
    
    def __init__(self, encoder, projector_dim=8192, lambda_param=5e-3):
        super().__init__()
        self.encoder = encoder
        self.lambda_param = lambda_param
        
        # Projector
        backbone_dim = encoder.projection_head[-1].in_features
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
    
    def forward(self, x1, x2):
        # Get representations
        h1, _ = self.encoder(x1)
        h2, _ = self.encoder(x2)
        
        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        return z1, z2
    
    def barlow_twins_loss(self, z1, z2):
        """Barlow Twins loss function"""
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        # Normalize
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
        
        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        # Barlow Twins loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(feature_dim-1, feature_dim+1)[:, :-1].pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        return loss

# Evaluation: Linear Probing
class LinearProbe(nn.Module):
    """Linear probe for evaluating learned representations"""
    
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Linear classifier
        feature_dim = encoder.projection_head[-1].in_features
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features, _ = self.encoder(x)
        
        return self.classifier(features)

# Training utilities
class SSLTrainer:
    """Self-supervised learning trainer"""
    
    def __init__(self, model, method='simclr', device='cuda'):
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.train_losses = []
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if self.method == 'simclr':
                loss = self.model.train_step(batch, optimizer)
            elif self.method == 'barlow_twins':
                (x1, x2), _ = batch
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                optimizer.zero_grad()
                z1, z2 = self.model(x1, x2)
                loss = self.model.barlow_twins_loss(z1, z2)
                loss.backward()
                optimizer.step()
                loss = loss.item()
            
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss:.4f}')
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate_linear_probe(self, train_loader, test_loader, num_classes):
        """Evaluate with linear probing"""
        
        # Create linear probe
        probe = LinearProbe(self.model.encoder, num_classes).to(self.device)
        optimizer = torch.optim.Adam(probe.classifier.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Train linear probe
        probe.train()
        for epoch in range(10):  # Quick training
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = probe(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        probe.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = probe(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy

# Demo dataset for contrastive learning
class ContrastiveDataset(Dataset):
    """Dataset that returns two augmented views"""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Apply contrastive transforms
        view1, view2 = self.transform(image)
        
        return (view1, view2), label

def demo_self_supervised_learning():
    """Demonstrate self-supervised learning methods"""
    
    print("=== Self-Supervised Learning Demo ===")
    
    # Create encoder
    encoder = SimCLREncoder('resnet18', feature_dim=128)
    
    # Test different SSL methods
    print("\n1. SimCLR")
    simclr = SimCLR(encoder)
    
    print("\n2. Barlow Twins")
    barlow_twins = BarlowTwins(encoder)
    
    print("\n3. DINO")
    teacher = SimCLREncoder('resnet18', feature_dim=128)
    dino = DINO(encoder, teacher)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        # SimCLR
        _, z = encoder(x)
        print(f"SimCLR representation: {z.shape}")
        
        # Barlow Twins
        z1, z2 = barlow_twins(x, x)
        print(f"Barlow Twins representations: {z1.shape}, {z2.shape}")
    
    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

if __name__ == "__main__":
    # Run demo
    demo_self_supervised_learning()
    
    print("\n=== SSL Methods Comparison ===")
    methods = [
        "SimCLR: Contrastive learning with strong augmentations",
        "SwAV: Clustering-based approach with prototypes", 
        "DINO: Self-distillation with Vision Transformers",
        "MAE: Masked autoencoding for representation learning",
        "Barlow Twins: Redundancy reduction objective",
        "MoCo: Momentum-based contrastive learning"
    ]
    
    for method in methods:
        print(f"• {method}")
