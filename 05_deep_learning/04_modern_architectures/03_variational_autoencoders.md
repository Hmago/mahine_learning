# Variational Autoencoders (VAEs): Learning Data Distributions

Master Variational Autoencoders, the elegant probabilistic approach to generative modeling! Learn to implement VAEs that can generate new data while providing a meaningful latent space representation.

## 🎯 What You'll Master

- **VAE Fundamentals**: Understanding variational inference and the evidence lower bound
- **Implementation**: Complete VAE from scratch with reparameterization trick
- **Advanced VAEs**: β-VAE, Conditional VAE, Hierarchical VAE
- **Applications**: Data generation, representation learning, anomaly detection

## 📚 The Variational Autoencoder Story

### From Regular Autoencoders to VAEs

**Traditional Autoencoder:**

```text
Input → Encoder → Latent Code → Decoder → Reconstruction
```

**Variational Autoencoder:**

```text
Input → Encoder → μ, σ → Sample z ~ N(μ, σ²) → Decoder → Reconstruction
```

**Key Insight:**
Instead of learning deterministic latent codes, VAEs learn probability distributions in latent space. This enables generation of new data by sampling from these learned distributions!

Think of VAE like learning to compress photos:
- **Encoder**: Creates a "recipe" (mean and variance) for reconstructing the photo
- **Sampling**: Adds controlled randomness to the recipe
- **Decoder**: Reconstructs photo from the noisy recipe
- **Magic**: Similar recipes create similar photos, enabling generation!

## 1. VAE Theory and Mathematics

### The Variational Objective

**Evidence Lower Bound (ELBO):**

```text
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction Loss - KL Divergence
```

Where:
- `q(z|x)` is the encoder (approximate posterior)
- `p(x|z)` is the decoder (likelihood)
- `p(z)` is the prior (usually N(0,I))

**The VAE Loss Function:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import math

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction and KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (β-VAE)
    """
    # Reconstruction loss (Binary Cross Entropy for binary data)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0,I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def reparameterize(mu, logvar):
    """
    Reparameterization trick for backpropagation through sampling
    
    Instead of sampling z ~ N(mu, sigma^2), we sample:
    epsilon ~ N(0,1) and compute z = mu + sigma * epsilon
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Basic VAE Implementation
class VAE(nn.Module):
    """Variational Autoencoder"""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For binary data like MNIST
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass through VAE"""
        # Flatten input
        x_flat = x.view(-1, self.input_dim)
        
        # Encode
        mu, logvar = self.encode(x_flat)
        
        # Reparameterize
        z = reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples=16, device='cpu'):
        """Generate new samples"""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples

# Convolutional VAE for images
class ConvVAE(nn.Module):
    """Convolutional VAE for image data"""
    
    def __init__(self, channels=1, image_size=28, latent_dim=128):
        super().__init__()
        
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Calculate flattened size after convolutions
        # For 28x28 -> 14x14 -> 7x7 with 32 channels
        self.flatten_size = 32 * 7 * 7
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(),
            # 14x14 -> 7x7
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode image to latent parameters"""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent to image"""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 32, 7, 7)  # Reshape for conv
        recon = self.decoder_conv(h)
        return recon
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# β-VAE (Beta VAE)
class BetaVAE(ConvVAE):
    """β-VAE for disentangled representation learning"""
    
    def __init__(self, channels=1, image_size=28, latent_dim=128, beta=4.0):
        super().__init__(channels, image_size, latent_dim)
        self.beta = beta
    
    def loss_function(self, recon_x, x, mu, logvar):
        """β-VAE loss with adjustable β parameter"""
        return vae_loss(recon_x, x, mu, logvar, beta=self.beta)

# Conditional VAE
class ConditionalVAE(nn.Module):
    """Conditional VAE that takes class labels as input"""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, num_classes)
        
        # Encoder (input + class)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (latent + class)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        """Encode input with class condition"""
        x_flat = x.view(-1, self.input_dim)
        c_emb = self.class_emb(c)
        
        # Concatenate input and class
        x_c = torch.cat([x_flat, c_emb], dim=1)
        
        h = self.encoder(x_c)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, c):
        """Decode with class condition"""
        c_emb = self.class_emb(c)
        
        # Concatenate latent and class
        z_c = torch.cat([z, c_emb], dim=1)
        
        return self.decoder(z_c)
    
    def forward(self, x, c):
        """Full forward pass with condition"""
        mu, logvar = self.encode(x, c)
        z = reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar
    
    def sample(self, num_samples=16, class_label=0, device='cpu'):
        """Generate samples for specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            c = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            samples = self.decode(z, c)
            return samples

# Hierarchical VAE
class HierarchicalVAE(nn.Module):
    """VAE with hierarchical latent structure"""
    
    def __init__(self, input_dim=784, latent_dims=[20, 10]):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        
        # Encoder networks for each level
        self.encoders = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        
        # Build hierarchical encoder
        prev_dim = input_dim
        for i, latent_dim in enumerate(latent_dims):
            encoder = nn.Sequential(
                nn.Linear(prev_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU()
            )
            self.encoders.append(encoder)
            self.mu_layers.append(nn.Linear(400, latent_dim))
            self.logvar_layers.append(nn.Linear(400, latent_dim))
            prev_dim = latent_dim
        
        # Decoder networks
        self.decoders = nn.ModuleList()
        
        # Build hierarchical decoder
        for i in range(self.num_levels - 1, -1, -1):
            if i == self.num_levels - 1:
                # Top level decoder
                decoder = nn.Sequential(
                    nn.Linear(latent_dims[i], 400),
                    nn.ReLU(),
                    nn.Linear(400, 400),
                    nn.ReLU()
                )
            else:
                # Lower level decoders
                decoder = nn.Sequential(
                    nn.Linear(latent_dims[i+1] + latent_dims[i], 400),
                    nn.ReLU(),
                    nn.Linear(400, 400),
                    nn.ReLU()
                )
            self.decoders.append(decoder)
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Hierarchical encoding"""
        h = x.view(-1, self.input_dim)
        
        mus, logvars = [], []
        for i in range(self.num_levels):
            h = self.encoders[i](h)
            mu = self.mu_layers[i](h)
            logvar = self.logvar_layers[i](h)
            
            mus.append(mu)
            logvars.append(logvar)
            
            # Sample for next level
            h = reparameterize(mu, logvar)
        
        return mus, logvars
    
    def decode(self, zs):
        """Hierarchical decoding"""
        h = None
        
        # Decode from top to bottom
        for i, z in enumerate(reversed(zs)):
            if i == 0:
                # Top level
                h = self.decoders[i](z)
            else:
                # Concatenate with previous level
                h = torch.cat([h, z], dim=1)
                h = self.decoders[i](h)
        
        return self.output_layer(h)
    
    def forward(self, x):
        """Hierarchical VAE forward pass"""
        mus, logvars = self.encode(x)
        
        # Sample at each level
        zs = [reparameterize(mu, logvar) for mu, logvar in zip(mus, logvars)]
        
        recon_x = self.decode(zs)
        
        return recon_x, mus, logvars

# Training utilities
class VAETrainer:
    """VAE training utilities"""
    
    def __init__(self, model, device='cuda', beta=1.0):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader, optimizer):
        """Train VAE for one epoch"""
        self.model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.device)
            
            optimizer.zero_grad()
            
            if isinstance(self.model, ConditionalVAE):
                # For conditional VAE, we need labels
                _, labels = _  # Unpack labels
                labels = labels.to(self.device)
                recon_batch, mu, logvar = self.model(data, labels)
            else:
                recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss
            if isinstance(self.model, HierarchicalVAE):
                loss = self.hierarchical_loss(recon_batch, data, mu, logvar)
            else:
                loss, recon_loss, kl_loss = vae_loss(
                    recon_batch, data.view(-1, self.model.input_dim), 
                    mu, logvar, self.beta
                )
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = train_loss / len(dataloader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def hierarchical_loss(self, recon_x, x, mus, logvars):
        """Loss for hierarchical VAE"""
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(
            recon_x, x.view(-1, self.model.input_dim), reduction='sum'
        )
        
        # KL loss for each level
        kl_loss = 0
        for mu, logvar in zip(mus, logvars):
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
    
    def validate(self, dataloader):
        """Validate VAE"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                
                if isinstance(self.model, ConditionalVAE):
                    labels = _.to(self.device)
                    recon_batch, mu, logvar = self.model(data, labels)
                else:
                    recon_batch, mu, logvar = self.model(data)
                
                if isinstance(self.model, HierarchicalVAE):
                    loss = self.hierarchical_loss(recon_batch, data, mu, logvar)
                else:
                    loss, _, _ = vae_loss(
                        recon_batch, data.view(-1, self.model.input_dim), 
                        mu, logvar, self.beta
                    )
                
                val_loss += loss.item()
        
        avg_loss = val_loss / len(dataloader.dataset)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('VAE Training Progress')
        plt.show()

# Visualization utilities
def visualize_vae_results(model, dataloader, device, num_samples=8):
    """Visualize VAE reconstruction and generation"""
    model.eval()
    
    # Get some test data
    data_iter = iter(dataloader)
    data, labels = next(data_iter)
    data = data[:num_samples].to(device)
    
    with torch.no_grad():
        # Reconstructions
        if isinstance(model, ConditionalVAE):
            labels = labels[:num_samples].to(device)
            recon, _, _ = model(data, labels)
        else:
            recon, _, _ = model(data)
        
        # Generated samples
        samples = model.sample(num_samples, device=device)
        
        # Plot results
        fig, axes = plt.subplots(3, num_samples, figsize=(12, 6))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            if hasattr(model, 'input_dim') and model.input_dim == 784:
                recon_img = recon[i].cpu().view(28, 28)
            else:
                recon_img = recon[i].cpu().squeeze()
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
            
            # Generated
            if hasattr(model, 'input_dim') and model.input_dim == 784:
                sample_img = samples[i].cpu().view(28, 28)
            else:
                sample_img = samples[i].cpu().squeeze()
            axes[2, i].imshow(sample_img, cmap='gray')
            axes[2, i].set_title('Generated')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()

def interpolate_latent_space(model, point1, point2, steps=10, device='cpu'):
    """Interpolate between two points in latent space"""
    model.eval()
    
    # Create interpolation path
    alphas = np.linspace(0, 1, steps)
    interpolations = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation
            z = alpha * point1 + (1 - alpha) * point2
            z = z.to(device)
            
            # Decode
            if hasattr(model, 'decode'):
                sample = model.decode(z)
            else:
                sample = model.decoder(z)
            
            interpolations.append(sample.cpu())
    
    # Plot interpolation
    fig, axes = plt.subplots(1, steps, figsize=(15, 3))
    for i, sample in enumerate(interpolations):
        if len(sample.shape) == 2:  # Flattened image
            img = sample.view(28, 28)
        else:
            img = sample.squeeze()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'α={alphas[i]:.1f}')
        axes[i].axis('off')
    
    plt.title('Latent Space Interpolation')
    plt.tight_layout()
    plt.show()

def explore_latent_dimensions(model, latent_dim=0, range_vals=(-3, 3), steps=10, device='cpu'):
    """Explore effect of varying single latent dimension"""
    model.eval()
    
    values = np.linspace(range_vals[0], range_vals[1], steps)
    samples = []
    
    with torch.no_grad():
        # Create base latent vector
        z_base = torch.zeros(1, model.latent_dim).to(device)
        
        for val in values:
            z = z_base.clone()
            z[0, latent_dim] = val
            
            if hasattr(model, 'decode'):
                sample = model.decode(z)
            else:
                sample = model.decoder(z)
            
            samples.append(sample.cpu())
    
    # Plot results
    fig, axes = plt.subplots(1, steps, figsize=(15, 3))
    for i, sample in enumerate(samples):
        if len(sample.shape) == 2:
            img = sample.view(28, 28)
        else:
            img = sample.squeeze()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{values[i]:.1f}')
        axes[i].axis('off')
    
    plt.suptitle(f'Latent Dimension {latent_dim} Variation')
    plt.tight_layout()
    plt.show()

# Demo functions
def demo_basic_vae():
    """Demonstrate basic VAE"""
    print("=== Basic VAE Demo ===")
    
    # Create model
    model = VAE(input_dim=784, latent_dim=20)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    
    recon_x, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon_x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Calculate loss
    loss, recon_loss, kl_loss = vae_loss(recon_x, x.view(-1, 784), mu, logvar)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Generate samples
    samples = model.sample(4)
    print(f"Generated samples shape: {samples.shape}")

def demo_conv_vae():
    """Demonstrate convolutional VAE"""
    print("\n=== Convolutional VAE Demo ===")
    
    model = ConvVAE(channels=1, latent_dim=128)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    
    recon_x, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon_x.shape}")
    print(f"Latent dimensions: {mu.shape[1]}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    # Run demos
    demo_basic_vae()
    demo_conv_vae()
    
    print("\n=== VAE Variants Comparison ===")
    
    models = {
        'Basic VAE': VAE(latent_dim=20),
        'Convolutional VAE': ConvVAE(latent_dim=128),
        'β-VAE': BetaVAE(latent_dim=128, beta=4.0),
        'Conditional VAE': ConditionalVAE(latent_dim=20, num_classes=10),
        'Hierarchical VAE': HierarchicalVAE(latent_dims=[20, 10])
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total_params:,} parameters")
