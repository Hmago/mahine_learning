# ML/AI Learning Environment Setup

# Create virtual environment
python -m venv ml_env

# Activate environment (Windows)
ml_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Verify installation
python -c "import numpy, pandas, sklearn, torch, tensorflow; print('All core packages installed successfully!')"

# Launch Jupyter Lab
jupyter lab
