# Generative Adversarial Networks (GANs): Creating New Data

Master the art of generating realistic data with GANs! Learn to implement and train these powerful generative models that have revolutionized computer vision, from basic GANs to state-of-the-art architectures like StyleGAN.

## 🎯 What You'll Master

- **GAN Fundamentals**: Understanding the adversarial training process
- **Classic Architectures**: DCGAN, Conditional GANs, CycleGAN
- **Advanced Techniques**: Progressive GANs, StyleGAN, Self-Attention
- **Training Strategies**: Stable training and common pitfalls

## 📚 The GAN Revolution

### The Game of Creation and Detection

**Think of GANs like this:**
- **Generator**: An art forger trying to create fake paintings
- **Discriminator**: An art expert trying to spot fakes
- **Training**: They compete and both get better over time
- **Result**: Eventually the forger creates indistinguishable masterpieces!

**Mathematical Foundation:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

This is a minimax game where:
- Generator (G) tries to minimize the objective
- Discriminator (D) tries to maximize it
- At equilibrium, both reach optimal performance

## 1. Basic GAN Implementation

### Simple GAN for MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os

class Generator(nn.Module):
    """Simple generator for MNIST-like images"""
    
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # Calculate output size
        img_size = int(np.prod(img_shape))
        
        # Simple MLP generator
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    """Simple discriminator for MNIST-like images"""
    
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        # Calculate input size
        img_size = int(np.prod(img_shape))
        
        # Simple MLP discriminator
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class SimpleGAN:
    """Simple GAN trainer"""
    
    def __init__(self, latent_dim=100, lr=0.0002, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        
        # Initialize models
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training history
        self.g_losses = []
        self.d_losses = []
    
    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # ===== Train Discriminator =====
        self.optimizer_D.zero_grad()
        
        # Real images
        real_validity = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        fake_validity = self.discriminator(fake_imgs.detach())
        d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # ===== Train Generator =====
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        
        # Generator wants discriminator to classify fake as real
        fake_validity = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(fake_validity, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataloader, epochs=100, sample_interval=1000):
        """Train the GAN"""
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for i, (imgs, _) in enumerate(dataloader):
                # Move to device and normalize to [-1, 1]
                imgs = imgs.to(self.device)
                imgs = 2 * imgs - 1  # Normalize to [-1, 1]
                
                # Train step
                d_loss, g_loss = self.train_step(imgs)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                
                # Sample images
                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    self.sample_images(epoch, batches_done)
            
            # Record losses
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            
            print(f"Epoch {epoch}/{epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
    
    def sample_images(self, epoch, batches_done, n_samples=25):
        """Generate and save sample images"""
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            fake_imgs = self.generator(z)
            
            # Denormalize images
            fake_imgs = (fake_imgs + 1) / 2  # From [-1,1] to [0,1]
            
            # Create grid
            grid = torchvision.utils.make_grid(fake_imgs, nrow=5, normalize=True)
            
            # Plot
            plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
            plt.title(f'Generated Images - Epoch {epoch}, Batch {batches_done}')
            plt.axis('off')
            plt.show()
        
        self.generator.train()
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.show()

# Deep Convolutional GAN (DCGAN)
class DCGANGenerator(nn.Module):
    """DCGAN Generator with transposed convolutions"""
    
    def __init__(self, latent_dim=100, channels=3, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        # Calculate initial size for transposed convolutions
        # For 64x64 output: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.init_size = 4
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator with strided convolutions"""
    
    def __init__(self, channels=3, img_size=64):
        super().__init__()
        self.channels = channels
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Discriminator block with conv, norm, and activation"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            *discriminator_block(channels, 64, normalize=False),
            # 32x32 -> 16x16
            *discriminator_block(64, 128),
            # 16x16 -> 8x8
            *discriminator_block(128, 256),
            # 8x8 -> 4x4
            *discriminator_block(256, 512),
            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(validity.size(0), -1)

# Conditional GAN
class ConditionalGenerator(nn.Module):
    """Generator that takes class labels as input"""
    
    def __init__(self, latent_dim=100, n_classes=10, channels=1, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.channels = channels
        self.img_size = img_size
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        # Calculate total input size (noise + label)
        input_dim = latent_dim + n_classes
        
        # Generator network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_emb(labels)
        
        # Concatenate noise and label embeddings
        gen_input = torch.cat([noise, label_embed], dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img

class ConditionalDiscriminator(nn.Module):
    """Discriminator that takes class labels as input"""
    
    def __init__(self, n_classes=10, channels=1, img_size=28):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.img_size = img_size
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        # Calculate input size (image + label)
        input_dim = channels * img_size * img_size + n_classes
        
        # Discriminator network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Embed labels
        label_embed = self.label_emb(labels)
        
        # Concatenate image and label embeddings
        disc_input = torch.cat([img_flat, label_embed], dim=1)
        
        # Classify
        validity = self.model(disc_input)
        return validity

# Wasserstein GAN with Gradient Penalty (WGAN-GP)
class WGANCritic(nn.Module):
    """Critic network for WGAN (no sigmoid activation)"""
    
    def __init__(self, channels=3, img_size=64):
        super().__init__()
        self.channels = channels
        
        def critic_block(in_filters, out_filters, normalize=True):
            """Critic block without batch norm"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.LayerNorm([out_filters, img_size//2, img_size//2]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *critic_block(channels, 64, normalize=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0),
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(validity.size(0), -1)

class WGANGP:
    """Wasserstein GAN with Gradient Penalty"""
    
    def __init__(self, latent_dim=100, lr=0.0001, device='cuda', lambda_gp=10):
        self.device = device
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        
        # Initialize models
        self.generator = DCGANGenerator(latent_dim).to(device)
        self.critic = WGANCritic().to(device)
        
        # Optimizers (using RMSprop as recommended for WGAN)
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_C = optim.RMSprop(self.critic.parameters(), lr=lr)
        
        # Training history
        self.g_losses = []
        self.c_losses = []
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get critic scores for interpolated samples
        d_interpolates = self.critic(interpolates)
        
        # Get gradients with respect to interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_imgs, n_critic=5):
        """Training step for WGAN-GP"""
        batch_size = real_imgs.size(0)
        
        # Train Critic
        for _ in range(n_critic):
            self.optimizer_C.zero_grad()
            
            # Real images
            real_validity = self.critic(real_imgs)
            
            # Fake images
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_imgs = self.generator(z).detach()
            fake_validity = self.critic(fake_imgs)
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
            
            # Critic loss (Wasserstein loss + gradient penalty)
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            
            c_loss.backward()
            self.optimizer_C.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        fake_validity = self.critic(fake_imgs)
        
        # Generator loss
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return c_loss.item(), g_loss.item()

# Progressive GAN concepts
class ProgressiveGenerator(nn.Module):
    """Progressive GAN generator that grows during training"""
    
    def __init__(self, latent_dim=512, max_resolution=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_resolution = max_resolution
        self.current_resolution = 4
        
        # Initial 4x4 block
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Progressive blocks for different resolutions
        self.blocks = nn.ModuleDict()
        self.to_rgb = nn.ModuleDict()
        
        # Add blocks for each resolution
        in_channels = 512
        for res in [8, 16, 32, 64, 128, 256, 512, 1024]:
            if res <= max_resolution:
                out_channels = min(512, 512 // (res // 8))
                
                self.blocks[str(res)] = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2)
                )
                
                self.to_rgb[str(res)] = nn.Conv2d(out_channels, 3, 1, 1, 0)
                in_channels = out_channels
    
    def forward(self, z, alpha=1.0):
        """Forward pass with fade-in support"""
        x = z.view(z.size(0), self.latent_dim, 1, 1)
        x = self.initial_block(x)
        
        # Progressive upsampling
        resolution = 4
        while resolution < self.current_resolution:
            resolution *= 2
            x = self.blocks[str(resolution)](x)
        
        # Convert to RGB
        img = self.to_rgb[str(self.current_resolution)](x)
        
        # Apply fade-in if needed (for smooth transitions)
        if alpha < 1.0 and self.current_resolution > 4:
            prev_res = self.current_resolution // 2
            prev_img = F.interpolate(
                self.to_rgb[str(prev_res)](x), 
                scale_factor=2, 
                mode='nearest'
            )
            img = alpha * img + (1 - alpha) * prev_img
        
        return torch.tanh(img)
    
    def grow(self):
        """Grow the network to next resolution"""
        if self.current_resolution < self.max_resolution:
            self.current_resolution *= 2
            print(f"Grown to resolution: {self.current_resolution}")

# Style-based Generator (StyleGAN concepts)
class MappingNetwork(nn.Module):
    """Mapping network for StyleGAN"""
    
    def __init__(self, latent_dim=512, style_dim=512, n_layers=8):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(latent_dim if i == 0 else style_dim, style_dim),
                nn.LeakyReLU(0.2)
            ])
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping(z)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)
    
    def forward(self, x, style):
        normalized = self.norm(x)
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        return scale * normalized + bias

# Training utilities
def train_basic_gan():
    """Demo training basic GAN on MNIST"""
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize and train GAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = SimpleGAN(device=device)
    
    print("Starting GAN training...")
    gan.train(dataloader, epochs=50, sample_interval=500)
    gan.plot_losses()

def train_conditional_gan():
    """Demo training conditional GAN"""
    
    # Data loading with labels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize conditional GAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = ConditionalGenerator(n_classes=10).to(device)
    discriminator = ConditionalDiscriminator(n_classes=10).to(device)
    
    print("Conditional GAN initialized")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Evaluation metrics
def calculate_fid_score(real_features, fake_features):
    """Calculate Fréchet Inception Distance"""
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def inception_score(generated_images, num_splits=10):
    """Calculate Inception Score"""
    # This is a simplified version - normally you'd use pre-trained Inception model
    # For demonstration purposes only
    
    scores = []
    for i in range(num_splits):
        # Calculate conditional and marginal probabilities
        # IS = exp(E[KL(p(y|x) || p(y))])
        pass
    
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    # Demo different GAN architectures
    print("=== GAN Architecture Comparison ===")
    
    # Simple GAN
    simple_gen = Generator()
    simple_disc = Discriminator()
    print(f"Simple GAN - Gen: {sum(p.numel() for p in simple_gen.parameters()):,}, "
          f"Disc: {sum(p.numel() for p in simple_disc.parameters()):,}")
    
    # DCGAN
    dcgan_gen = DCGANGenerator()
    dcgan_disc = DCGANDiscriminator()
    print(f"DCGAN - Gen: {sum(p.numel() for p in dcgan_gen.parameters()):,}, "
          f"Disc: {sum(p.numel() for p in dcgan_disc.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 100)
    
    with torch.no_grad():
        simple_img = simple_gen(z)
        dcgan_img = dcgan_gen(z)
        
        print(f"Simple GAN output: {simple_img.shape}")
        print(f"DCGAN output: {dcgan_img.shape}")
    
    print("\nTo train GANs, run train_basic_gan() or train_conditional_gan()")
