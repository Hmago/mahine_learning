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

### Option 3: Windows Package Manager
```powershell
# Install using winget (Windows 10/11)
winget install Python.Python.3.9

# Or using chocolatey
choco install python
```

## 🚀 Environment Setup

### Step 1: Create Virtual Environment
```powershell
# Navigate to learning folder
cd "c:\Users\harshitmago\Documents\learning"

# Create virtual environment
python -m venv ml_env

# Activate environment (Windows)
ml_env\Scripts\activate

# You should see (ml_env) in your prompt
```

### Step 2: Install Packages
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

### Step 3: Verify Installation
```powershell
# Test core packages
python -c "import numpy, pandas, sklearn, matplotlib; print('Core packages OK!')"

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
```powershell
# After activating environment
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
- Reinstall Python with "Add to PATH" checked
- Restart terminal/VS Code

**2. Permission errors**
- Run terminal as administrator
- Use `--user` flag: `pip install --user package_name`

**3. Package conflicts**
- Create fresh virtual environment
- Install packages one by one

**4. Jupyter not starting**
- Try: `python -m jupyter lab`
- Install: `pip install ipykernel`

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
