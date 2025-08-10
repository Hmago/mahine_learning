# Python Installation and Setup Guide

## 🐍 Installing Python

### Option 1: Official Python (Recommended)
1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation: `python --version`

### Option 2: Anaconda (Data Science Focus)
1. Download Anaconda from [anaconda.com](https://www.anaconda.com/products/distribution)
2. Install with default settings
3. Use Anaconda Navigator or `conda` commands

### Option 3: Package Managers

#### Windows
```powershell
# Install using winget (Windows 10/11)
winget install Python.Python.3.9

# Or using chocolatey
choco install python
```

#### macOS
```bash
# Install using Homebrew (recommended)
brew install python@3.11

# Or using MacPorts
sudo port install python311

# Or using pyenv (version management)
brew install pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

## 🚀 Environment Setup

### Step 1: Create Virtual Environment

#### Windows
```powershell
# Navigate to learning folder
cd "c:\Users\harshitmago\Documents\learning"

# Create virtual environment
python -m venv ml_env

# Activate environment (Windows)
ml_env\Scripts\activate

# You should see (ml_env) in your prompt
```

#### macOS/Linux
```bash
# Navigate to learning folder
cd ~/Documents/learning/machine_learning/mahine_learning

# Create virtual environment
python3 -m venv ml_env

# Activate environment (macOS/Linux)
source ml_env/bin/activate

# You should see (ml_env) in your prompt
```

### Step 2: Install Packages

#### Windows
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install core packages (start with these)
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Install deep learning (after core packages work)
pip install torch torchvision tensorflow

# Install LLM and AI packages
pip install openai langchain streamlit

# Or install everything at once
pip install -r requirements.txt
```

#### macOS/Linux
```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Install core packages (start with these)
pip3 install numpy pandas matplotlib seaborn scikit-learn jupyter

# Install deep learning (after core packages work)
pip3 install torch torchvision tensorflow

# Install LLM and AI packages
pip3 install openai langchain streamlit

# Or install everything at once
pip3 install -r requirements.txt
```

### Step 3: Verify Installation

#### Windows
```powershell
# Test core packages
python -c "import numpy, pandas, sklearn, matplotlib; print('Core packages OK!')"

# Launch Jupyter
jupyter lab
```

#### macOS/Linux
```bash
# Test core packages
python3 -c "import numpy, pandas, sklearn, matplotlib; print('Core packages OK!')"

# Launch Jupyter
jupyter lab
```

## 🛠 Alternative: Google Colab

If you have installation issues, use Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload the notebook files
3. All packages are pre-installed
4. GPU access available for free

## 🔧 IDE Setup

### VS Code (Recommended)
1. Install [VS Code](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension
4. Select Python interpreter from your virtual environment

### Jupyter Lab
```bash
# After activating environment (cross-platform)
jupyter lab
```

### PyCharm
- Configure interpreter to use your virtual environment
- Enable scientific mode for data analysis

## 📦 Package Management

### Essential Packages (Install First)
```
numpy>=1.24.0       # Numerical computing
pandas>=2.0.0       # Data manipulation
matplotlib>=3.7.0   # Basic plotting
seaborn>=0.12.0     # Statistical plotting
scikit-learn>=1.3.0 # Machine learning
jupyter>=1.0.0      # Interactive notebooks
```

### Deep Learning (Install Second)
```
torch>=2.0.0        # PyTorch
tensorflow>=2.13.0  # TensorFlow
```

### AI/LLM Packages (Install Third)
```
openai>=1.0.0       # OpenAI API
langchain>=0.1.0    # AI applications
streamlit>=1.25.0   # Web apps
```

## 🚨 Troubleshooting

### Common Issues

**1. Python not found**
- **Windows**: Reinstall Python with "Add to PATH" checked, restart terminal/VS Code
- **macOS**: Install via Homebrew (`brew install python@3.11`) or ensure `/usr/local/bin` is in PATH

**2. Permission errors**
- **Windows**: Run terminal as administrator, use `--user` flag: `pip install --user package_name`
- **macOS**: Use `sudo` for system installs or `--user` flag: `pip3 install --user package_name`

**3. Package conflicts**
- Create fresh virtual environment
- Install packages one by one

**4. Jupyter not starting**
- Try: `python -m jupyter lab` (Windows) or `python3 -m jupyter lab` (macOS)
- Install: `pip install ipykernel`

**5. macOS-specific issues**
- **Command Line Tools**: Install with `xcode-select --install`
- **Homebrew**: Install from [brew.sh](https://brew.sh) if not present
- **M1/M2 Macs**: Some packages may need specific conda-forge versions

## 💻 macOS-Specific Tips

### Terminal Setup
- Use **Terminal** (built-in) or **iTerm2** (enhanced features)
- Add to `.zshrc` or `.bash_profile` for permanent PATH changes:
  ```bash
  export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
  ```

### Homebrew Installation
```bash
# Install Homebrew (macOS package manager)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and common tools
brew install python@3.11 git
```

### Apple Silicon (M1/M2/M3) Considerations
- Use `conda` for some packages that may not have native ARM builds
- Install Rosetta 2 if needed: `softwareupdate --install-rosetta`
- Consider using `conda-forge` channel for better compatibility

### Getting Help
- Check package documentation
- Use `pip list` to see installed packages
- Use `pip show package_name` for package info

## 🎯 Next Steps

Once Python is set up:

1. **Test the environment**: Run the fundamentals notebook
2. **Join communities**: Reddit r/learnpython, Discord servers
3. **Start coding**: Begin with `01_fundamentals/`
4. **Build projects**: Apply what you learn immediately

Remember: The goal is to start learning, not to have a perfect setup. Start with what works and improve as you go!
