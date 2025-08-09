"""
Common utility functions for ML projects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(filepath, target_column=None):
    """
    Load data and provide basic exploration
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    target_column : str, optional
        Name of the target column for supervised learning
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    # Load data
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    if target_column and target_column in df.columns:
        print(f"\nTarget variable distribution:")
        print(df[target_column].value_counts())
    
    return df

def plot_missing_data(df):
    """Plot missing data patterns"""
    plt.figure(figsize=(12, 6))
    
    # Missing data heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Data Heatmap')
    
    # Missing data percentage
    plt.subplot(1, 2, 2)
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    if len(missing_percent) > 0:
        missing_percent.plot(kind='bar')
        plt.title('Missing Data Percentage')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No missing data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Missing Data Percentage')
    
    plt.tight_layout()
    plt.show()

def plot_model_performance(y_true, y_pred, model_name="Model"):
    """
    Plot classification model performance
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for the title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} - Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Classification Report (as text)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Plot metrics
    metrics = ['precision', 'recall', 'f1-score']
    classes = [col for col in report_df.index if col not in ['accuracy', 'macro avg', 'weighted avg']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report_df.loc[cls, metric] for cls in classes]
        axes[1].bar(x + i*width, values, width, label=metric)
    
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'{model_name} - Performance Metrics')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(classes)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed report
    print(f"\n{model_name} - Detailed Classification Report:")
    print(classification_report(y_true, y_pred))

def plot_learning_curves(estimator, X, y, cv=5, scoring='accuracy'):
    """
    Plot learning curves to diagnose bias/variance
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(f'{scoring.title()} Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_models(models_dict, X_train, X_test, y_train, y_test):
    """
    Compare multiple models and return results
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_instance}
    X_train, X_test, y_train, y_test : array-like
        Train/test splits
    
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='weighted')
        recall = recall_score(y_test, test_pred, average='weighted')
        f1 = f1_score(y_test, test_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Overfit': train_acc - test_acc
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    metrics = ['Train_Accuracy', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    x = np.arange(len(results_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, results_df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width*2, results_df['Model'], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return results_df

def create_feature_importance_plot(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Return as DataFrame
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        return importance_df
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None

# Data preprocessing utilities
def handle_missing_data(df, strategy='mean', threshold=0.5):
    """
    Handle missing data in DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing data ('mean', 'median', 'mode', 'drop')
    threshold : float
        Threshold for dropping columns (percentage of missing data)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Drop columns with too much missing data
    missing_percent = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_percent[missing_percent > threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    if len(cols_to_drop) > 0:
        print(f"Dropped columns with >{threshold*100}% missing data: {list(cols_to_drop)}")
    
    # Handle remaining missing data
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['object', 'category']:
                # Categorical data
                if strategy == 'mode':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                # Numerical data
                if strategy == 'mean':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif strategy == 'median':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
    
    print(f"Data shape after cleaning: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    print("ML Utilities loaded successfully!")
    print("Available functions:")
    print("- load_and_explore_data()")
    print("- plot_missing_data()")
    print("- plot_model_performance()")
    print("- plot_learning_curves()")
    print("- compare_models()")
    print("- create_feature_importance_plot()")
    print("- handle_missing_data()")
