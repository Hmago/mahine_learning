# MLOps Best Practices: Professional ML Workflows

Learn to build production-grade machine learning workflows that are reproducible, scalable, and maintainable. Master the art of ML engineering with industry best practices.

## 🎯 What You'll Master

- **ML Pipeline Automation**: End-to-end automated workflows
- **Version Control**: Track models, data, and experiments
- **CI/CD for ML**: Continuous integration and deployment for models
- **Infrastructure as Code**: Reproducible ML environments

## 📚 The MLOps Philosophy

### Traditional Software vs MLOps

**Traditional Software Development:**
```
Code → Build → Test → Deploy → Monitor
```

**ML Development (MLOps):**
```
Data → Feature Engineering → Model Training → Validation → Deploy → Monitor → Retrain
```

**The MLOps Challenge:**
```
Code + Data + Model + Infrastructure + Monitoring = Complex System
```

Think of MLOps like conducting an orchestra - every instrument (component) needs to work in harmony to create beautiful music (reliable ML systems)!

## 1. ML Pipeline Automation

### End-to-End ML Pipeline

**Concept:** Automate the entire ML workflow from data ingestion to model deployment.

**Analogy:** Like a factory assembly line - each stage feeds into the next automatically.

```python
import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from dataclasses import dataclass
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Pipeline configuration
@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    
    # Data configuration
    data_source: str
    data_validation_rules: Dict
    feature_columns: List[str]
    target_column: str
    
    # Model configuration
    model_type: str
    model_params: Dict
    training_params: Dict
    
    # Validation configuration
    validation_split: float
    validation_metrics: List[str]
    performance_thresholds: Dict
    
    # Deployment configuration
    model_registry: str
    deployment_target: str
    
    # Monitoring configuration
    monitoring_config: Dict

class MLPipeline:
    """Complete ML pipeline with automation"""
    
    def __init__(self, config: PipelineConfig, run_id: str = None):
        self.config = config
        self.run_id = run_id or f"run_{int(datetime.now().timestamp())}"
        self.artifacts = {}
        self.metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup MLflow
        mlflow.set_experiment("ml_pipeline")
        
        self.logger.info(f"Pipeline initialized with run_id: {self.run_id}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        
        with mlflow.start_run(run_name=self.run_id):
            try:
                # Log configuration
                mlflow.log_params(self._flatten_config())
                
                # Step 1: Data ingestion and validation
                self.logger.info("Step 1: Data ingestion and validation")
                data = self._ingest_data()
                self._validate_data(data)
                
                # Step 2: Feature engineering
                self.logger.info("Step 2: Feature engineering")
                features, target = self._engineer_features(data)
                
                # Step 3: Data splitting
                self.logger.info("Step 3: Data splitting")
                X_train, X_val, y_train, y_val = self._split_data(features, target)
                
                # Step 4: Model training
                self.logger.info("Step 4: Model training")
                model = self._train_model(X_train, y_train)
                
                # Step 5: Model validation
                self.logger.info("Step 5: Model validation")
                validation_results = self._validate_model(model, X_val, y_val)
                
                # Step 6: Model registration (if validation passes)
                if self._should_register_model(validation_results):
                    self.logger.info("Step 6: Model registration")
                    model_version = self._register_model(model, validation_results)
                    
                    # Step 7: Deployment preparation
                    self.logger.info("Step 7: Deployment preparation")
                    deployment_artifacts = self._prepare_deployment(model, model_version)
                    
                    pipeline_result = {
                        'status': 'success',
                        'run_id': self.run_id,
                        'model_version': model_version,
                        'validation_results': validation_results,
                        'deployment_artifacts': deployment_artifacts,
                        'artifacts': self.artifacts
                    }
                else:
                    pipeline_result = {
                        'status': 'validation_failed',
                        'run_id': self.run_id,
                        'validation_results': validation_results,
                        'reason': 'Model did not meet performance thresholds'
                    }
                
                # Log final results
                mlflow.log_metrics(validation_results)
                mlflow.log_dict(pipeline_result, "pipeline_result.json")
                
                return pipeline_result
                
            except Exception as e:
                self.logger.error(f"Pipeline failed: {e}")
                error_result = {
                    'status': 'failed',
                    'run_id': self.run_id,
                    'error': str(e)
                }
                mlflow.log_dict(error_result, "pipeline_result.json")
                return error_result
    
    def _ingest_data(self) -> pd.DataFrame:
        """Ingest data from configured source"""
        
        if self.config.data_source.endswith('.csv'):
            data = pd.read_csv(self.config.data_source)
        elif self.config.data_source.endswith('.parquet'):
            data = pd.read_parquet(self.config.data_source)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
        
        self.logger.info(f"Ingested {len(data)} samples from {self.config.data_source}")
        
        # Save data artifacts
        data_path = f"artifacts/data_{self.run_id}.csv"
        os.makedirs("artifacts", exist_ok=True)
        data.to_csv(data_path, index=False)
        self.artifacts['raw_data'] = data_path
        
        mlflow.log_artifact(data_path, "data")
        mlflow.log_metric("data_samples", len(data))
        
        return data
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate data quality"""
        
        validation_rules = self.config.data_validation_rules
        validation_results = {}
        
        # Check required columns
        required_columns = self.config.feature_columns + [self.config.target_column]
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data quality rules
        for rule_name, rule_config in validation_rules.items():
            if rule_name == 'missing_threshold':
                missing_rate = data.isnull().sum().max() / len(data)
                validation_results['max_missing_rate'] = missing_rate
                if missing_rate > rule_config['threshold']:
                    raise ValueError(f"Missing data rate {missing_rate:.2%} exceeds threshold {rule_config['threshold']:.2%}")
            
            elif rule_name == 'duplicate_threshold':
                duplicate_rate = data.duplicated().sum() / len(data)
                validation_results['duplicate_rate'] = duplicate_rate
                if duplicate_rate > rule_config['threshold']:
                    raise ValueError(f"Duplicate rate {duplicate_rate:.2%} exceeds threshold {rule_config['threshold']:.2%}")
            
            elif rule_name == 'value_ranges':
                for column, value_range in rule_config.items():
                    if column in data.columns:
                        min_val, max_val = value_range
                        out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()
                        if out_of_range > 0:
                            self.logger.warning(f"Column {column} has {out_of_range} values out of range [{min_val}, {max_val}]")
        
        mlflow.log_metrics({f"data_validation_{k}": v for k, v in validation_results.items()})
        
        self.logger.info("Data validation passed")
    
    def _engineer_features(self, data: pd.DataFrame) -> tuple:
        """Engineer features for model training"""
        
        # Extract features and target
        features = data[self.config.feature_columns].copy()
        target = data[self.config.target_column].copy()
        
        # Basic feature engineering (can be extended)
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Feature scaling (if needed)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_df = pd.DataFrame(features_scaled, columns=self.config.feature_columns)
        
        # Save feature engineering artifacts
        scaler_path = f"artifacts/scaler_{self.run_id}.joblib"
        joblib.dump(scaler, scaler_path)
        self.artifacts['scaler'] = scaler_path
        
        mlflow.log_artifact(scaler_path, "preprocessing")
        
        self.logger.info(f"Features engineered: {features_df.shape}")
        
        return features_df, target
    
    def _split_data(self, features: pd.DataFrame, target: pd.Series) -> tuple:
        """Split data into training and validation sets"""
        
        X_train, X_val, y_train, y_val = train_test_split(
            features, target,
            test_size=self.config.validation_split,
            random_state=42,
            stratify=target if target.dtype == 'object' or len(target.unique()) < 20 else None
        )
        
        mlflow.log_metrics({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ratio": len(X_train) / (len(X_train) + len(X_val))
        })
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model"""
        
        if self.config.model_type == 'sklearn_classifier':
            return self._train_sklearn_model(X_train, y_train)
        elif self.config.model_type == 'pytorch_classifier':
            return self._train_pytorch_model(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _train_sklearn_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train scikit-learn model"""
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        model_map = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }
        
        model_class = model_map[self.config.model_params['algorithm']]
        model = model_class(**self.config.model_params.get('params', {}))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        model_path = f"artifacts/model_{self.run_id}.joblib"
        joblib.dump(model, model_path)
        self.artifacts['model'] = model_path
        
        mlflow.sklearn.log_model(model, "model")
        
        return model
    
    def _train_pytorch_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train PyTorch model"""
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Simple neural network
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.LongTensor(y_train.values)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = SimpleNN(
            input_size=X_train.shape[1],
            hidden_size=self.config.model_params.get('hidden_size', 64),
            num_classes=len(y_train.unique())
        )
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.training_params.get('learning_rate', 0.001))
        
        num_epochs = self.config.training_params.get('epochs', 100)
        
        for epoch in range(num_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                mlflow.log_metric("training_loss", loss.item(), step=epoch)
        
        # Save model
        model_path = f"artifacts/model_{self.run_id}.pth"
        torch.save(model.state_dict(), model_path)
        self.artifacts['model'] = model_path
        
        mlflow.pytorch.log_model(model, "model")
        
        return model
    
    def _validate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Validate model performance"""
        
        # Make predictions
        if self.config.model_type == 'sklearn_classifier':
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        else:  # PyTorch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val.values)
                outputs = model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1).numpy()
                y_prob = torch.softmax(outputs, dim=1).numpy()[:, 1] if outputs.shape[1] == 2 else None
        
        # Calculate metrics
        results = {}
        
        if 'accuracy' in self.config.validation_metrics:
            results['accuracy'] = accuracy_score(y_val, y_pred)
        
        if 'precision' in self.config.validation_metrics:
            results['precision'] = precision_score(y_val, y_pred, average='weighted')
        
        if 'recall' in self.config.validation_metrics:
            results['recall'] = recall_score(y_val, y_pred, average='weighted')
        
        if 'f1' in self.config.validation_metrics:
            results['f1'] = f1_score(y_val, y_pred, average='weighted')
        
        # Log confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = f"artifacts/confusion_matrix_{self.run_id}.png"
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path, "validation")
        
        self.logger.info(f"Validation results: {results}")
        
        return results
    
    def _should_register_model(self, validation_results: Dict[str, float]) -> bool:
        """Check if model meets performance thresholds"""
        
        for metric, threshold in self.config.performance_thresholds.items():
            if metric in validation_results:
                if validation_results[metric] < threshold:
                    self.logger.warning(f"Model failed threshold for {metric}: {validation_results[metric]:.4f} < {threshold}")
                    return False
        
        return True
    
    def _register_model(self, model, validation_results: Dict[str, float]) -> str:
        """Register model in model registry"""
        
        model_name = f"{self.config.model_registry}_{self.config.model_type}"
        
        # Register with MLflow
        model_version = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=model_name,
            tags={
                "validation_accuracy": str(validation_results.get('accuracy', 'N/A')),
                "pipeline_run_id": self.run_id,
                "model_type": self.config.model_type
            }
        )
        
        self.logger.info(f"Model registered: {model_name} version {model_version.version}")
        
        return model_version.version
    
    def _prepare_deployment(self, model, model_version: str) -> Dict[str, str]:
        """Prepare deployment artifacts"""
        
        deployment_artifacts = {}
        
        # Create deployment configuration
        deployment_config = {
            'model_name': f"{self.config.model_registry}_{self.config.model_type}",
            'model_version': model_version,
            'model_type': self.config.model_type,
            'preprocessing_artifacts': {
                'scaler': self.artifacts.get('scaler')
            },
            'monitoring_config': self.config.monitoring_config,
            'deployment_timestamp': datetime.now().isoformat()
        }
        
        config_path = f"artifacts/deployment_config_{self.run_id}.json"
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        deployment_artifacts['config'] = config_path
        mlflow.log_artifact(config_path, "deployment")
        
        # Create inference script
        inference_script = self._generate_inference_script()
        script_path = f"artifacts/inference_{self.run_id}.py"
        with open(script_path, 'w') as f:
            f.write(inference_script)
        
        deployment_artifacts['inference_script'] = script_path
        mlflow.log_artifact(script_path, "deployment")
        
        # Create Docker file
        dockerfile = self._generate_dockerfile()
        docker_path = f"artifacts/Dockerfile_{self.run_id}"
        with open(docker_path, 'w') as f:
            f.write(dockerfile)
        
        deployment_artifacts['dockerfile'] = docker_path
        mlflow.log_artifact(docker_path, "deployment")
        
        return deployment_artifacts
    
    def _generate_inference_script(self) -> str:
        """Generate inference script for deployment"""
        
        script = f"""
import joblib
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any

class ModelInference:
    def __init__(self, model_path: str, scaler_path: str = None):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.feature_columns = {self.config.feature_columns}
    
    def preprocess(self, input_data: Dict) -> np.ndarray:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Select feature columns
        features = df[self.feature_columns]
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def predict(self, input_data: Dict) -> Dict[str, Any]:
        # Preprocess
        features = self.preprocess(input_data)
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
        
        return {{
            'prediction': int(prediction),
            'confidence': confidence,
            'model_version': '{self.run_id}'
        }}

# Flask API wrapper
from flask import Flask, request, jsonify

app = Flask(__name__)
model_inference = ModelInference('model.joblib', 'scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = model_inference.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({{'error': str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy', 'model_version': '{self.run_id}'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
        return script
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for deployment"""
        
        dockerfile = f"""
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model artifacts
COPY model.joblib .
COPY scaler.joblib .
COPY inference.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "inference.py"]
"""
        return dockerfile
    
    def _flatten_config(self) -> Dict[str, Any]:
        """Flatten configuration for MLflow logging"""
        
        flat_config = {}
        
        for attr_name in dir(self.config):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.config, attr_name)
                if isinstance(attr_value, (str, int, float, bool)):
                    flat_config[attr_name] = attr_value
                elif isinstance(attr_value, (dict, list)):
                    flat_config[attr_name] = str(attr_value)
        
        return flat_config

# Pipeline orchestration with Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import timedelta

def create_ml_pipeline_dag():
    """Create Airflow DAG for ML pipeline"""
    
    default_args = {
        'owner': 'ml-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    }
    
    dag = DAG(
        'ml_training_pipeline',
        default_args=default_args,
        description='ML model training pipeline',
        schedule_interval='@daily',  # Run daily
        catchup=False
    )
    
    # Data validation task
    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=run_data_validation,
        dag=dag
    )
    
    # Model training task
    model_training = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
        dag=dag
    )
    
    # Model validation task
    model_validation = PythonOperator(
        task_id='model_validation',
        python_callable=run_model_validation,
        dag=dag
    )
    
    # Model deployment task
    model_deployment = BashOperator(
        task_id='model_deployment',
        bash_command='python deploy_model.py',
        dag=dag
    )
    
    # Set dependencies
    data_validation >> model_training >> model_validation >> model_deployment
    
    return dag

def run_data_validation():
    """Run data validation step"""
    # Implementation here
    pass

def run_model_training():
    """Run model training step"""
    # Implementation here
    pass

def run_model_validation():
    """Run model validation step"""
    # Implementation here
    pass

# Example usage
def run_example_pipeline():
    """Run example ML pipeline"""
    
    # Define configuration
    config = PipelineConfig(
        data_source="data/training_data.csv",
        data_validation_rules={
            'missing_threshold': {'threshold': 0.1},
            'duplicate_threshold': {'threshold': 0.05}
        },
        feature_columns=['feature1', 'feature2', 'feature3', 'feature4'],
        target_column='target',
        model_type='sklearn_classifier',
        model_params={
            'algorithm': 'random_forest',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        training_params={},
        validation_split=0.2,
        validation_metrics=['accuracy', 'precision', 'recall', 'f1'],
        performance_thresholds={'accuracy': 0.8, 'f1': 0.75},
        model_registry='production_model',
        deployment_target='kubernetes',
        monitoring_config={'drift_detection': True, 'performance_monitoring': True}
    )
    
    # Run pipeline
    pipeline = MLPipeline(config)
    result = pipeline.run()
    
    print(f"Pipeline completed with status: {result['status']}")
    return result

if __name__ == "__main__":
    # Example pipeline run
    result = run_example_pipeline()
    print(json.dumps(result, indent=2, default=str))
```

### CI/CD for Machine Learning

**Concept:** Implement continuous integration and deployment specifically for ML models.

**Analogy:** Like having a quality control team that automatically tests and approves your models before they go to production.

```python
# GitHub Actions workflow for ML CI/CD
github_actions_workflow = """
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

env:
  PYTHON_VERSION: 3.8
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  # Data quality checks
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data validation
      run: |
        python scripts/validate_data.py --config config/data_validation.yml
    
    - name: Upload data quality report
      uses: actions/upload-artifact@v3
      with:
        name: data-quality-report
        path: reports/data_quality.html

  # Model training and validation
  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model: [random_forest, logistic_regression, neural_network]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train model
      run: |
        python scripts/train_model.py \\
          --model-type ${{ matrix.model }} \\
          --config config/training.yml \\
          --output models/${{ matrix.model }}
    
    - name: Validate model
      run: |
        python scripts/validate_model.py \\
          --model-path models/${{ matrix.model }} \\
          --config config/validation.yml
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-${{ matrix.model }}
        path: models/${{ matrix.model }}

  # Model testing
  model-testing:
    needs: model-training
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-random_forest
        path: models/
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=html
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run model performance tests
      run: |
        python tests/performance_tests.py
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: htmlcov/

  # Security and compliance checks
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt
    
    - name: Run code quality check
      run: |
        pip install flake8 black isort
        flake8 src/
        black --check src/
        isort --check-only src/

  # Deployment to staging
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: [model-testing, security-scan]
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-random_forest
        path: models/
    
    - name: Build Docker image
      run: |
        docker build -t ml-model:staging .
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        kubectl apply -f k8s/staging/ --namespace=staging
    
    - name: Run smoke tests
      run: |
        python tests/smoke_tests.py --env staging

  # Deployment to production
  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: [model-testing, security-scan]
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-random_forest
        path: models/
    
    - name: Build Docker image
      run: |
        docker build -t ml-model:${{ github.sha }} .
        docker tag ml-model:${{ github.sha }} ml-model:latest
    
    - name: Push to registry
      run: |
        docker push ml-model:${{ github.sha }}
        docker push ml-model:latest
    
    - name: Deploy to production (blue-green)
      run: |
        python scripts/blue_green_deploy.py \\
          --image ml-model:${{ github.sha }} \\
          --namespace production
    
    - name: Run production tests
      run: |
        python tests/production_tests.py
    
    - name: Update model registry
      run: |
        python scripts/update_model_registry.py \\
          --version ${{ github.sha }} \\
          --status production
"""

# Model testing framework
import pytest
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
import requests
import time

class ModelTestSuite:
    """Comprehensive test suite for ML models"""
    
    def __init__(self, model_path: str, test_data_path: str):
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
        
    def test_model_loading(self):
        """Test that model loads correctly"""
        assert self.model is not None
        assert hasattr(self.model, 'predict')
    
    def test_input_validation(self):
        """Test model input validation"""
        # Test with correct input
        sample_input = self.test_data.iloc[0:1, :-1]  # Exclude target
        prediction = self.model.predict(sample_input)
        assert prediction is not None
        
        # Test with wrong number of features
        with pytest.raises(Exception):
            wrong_input = self.test_data.iloc[0:1, :-2]  # Missing feature
            self.model.predict(wrong_input)
    
    def test_output_format(self):
        """Test model output format"""
        sample_input = self.test_data.iloc[0:10, :-1]
        predictions = self.model.predict(sample_input)
        
        # Check output shape
        assert len(predictions) == 10
        
        # Check output type (adjust based on your model)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
    
    def test_performance_requirements(self):
        """Test model meets performance requirements"""
        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1]
        
        predictions = self.model.predict(X_test)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        
        # Performance threshold
        assert accuracy >= 0.80, f"Model accuracy {accuracy:.3f} below threshold 0.80"
    
    def test_inference_speed(self):
        """Test model inference speed"""
        sample_input = self.test_data.iloc[0:1, :-1]
        
        # Warmup
        for _ in range(5):
            self.model.predict(sample_input)
        
        # Measure inference time
        times = []
        for _ in range(100):
            start_time = time.time()
            self.model.predict(sample_input)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        
        # Speed requirement (adjust as needed)
        assert avg_time < 0.1, f"Average inference time {avg_time:.4f}s exceeds 0.1s"
    
    def test_prediction_stability(self):
        """Test prediction stability across multiple runs"""
        sample_input = self.test_data.iloc[0:1, :-1]
        
        predictions = []
        for _ in range(10):
            pred = self.model.predict(sample_input)[0]
            predictions.append(pred)
        
        # All predictions should be the same for the same input
        assert len(set(predictions)) == 1, "Model predictions are not stable"
    
    def test_edge_cases(self):
        """Test model behavior with edge cases"""
        sample_input = self.test_data.iloc[0:1, :-1].copy()
        
        # Test with extreme values
        extreme_input = sample_input.copy()
        extreme_input.iloc[0, :] = 999999  # Very large values
        
        try:
            prediction = self.model.predict(extreme_input)
            assert prediction is not None
        except Exception as e:
            pytest.fail(f"Model failed with extreme values: {e}")
        
        # Test with zero values
        zero_input = sample_input.copy()
        zero_input.iloc[0, :] = 0
        
        try:
            prediction = self.model.predict(zero_input)
            assert prediction is not None
        except Exception as e:
            pytest.fail(f"Model failed with zero values: {e}")

# API testing
class APITestSuite:
    """Test suite for model API"""
    
    def __init__(self, api_base_url: str):
        self.base_url = api_base_url
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert 'status' in health_data
        assert health_data['status'] == 'healthy'
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        test_input = {
            'feature1': 1.0,
            'feature2': 2.0,
            'feature3': 3.0,
            'feature4': 4.0
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=test_input,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        
        prediction_data = response.json()
        assert 'prediction' in prediction_data
        assert 'confidence' in prediction_data
    
    def test_api_response_time(self):
        """Test API response time"""
        test_input = {
            'feature1': 1.0,
            'feature2': 2.0,
            'feature3': 3.0,
            'feature4': 4.0
        }
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/predict", json=test_input)
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0, f"API response time {response_time:.3f}s exceeds 1.0s"
    
    def test_concurrent_requests(self):
        """Test API under concurrent load"""
        import concurrent.futures
        
        def make_request():
            test_input = {
                'feature1': np.random.random(),
                'feature2': np.random.random(),
                'feature3': np.random.random(),
                'feature4': np.random.random()
            }
            response = requests.post(f"{self.base_url}/predict", json=test_input)
            return response.status_code
        
        # Make 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(status == 200 for status in results), "Some concurrent requests failed"

# Example test configuration
def run_model_tests():
    """Run comprehensive model tests"""
    
    # Unit tests for model
    model_tests = ModelTestSuite("models/production_model.joblib", "data/test_data.csv")
    
    test_methods = [
        model_tests.test_model_loading,
        model_tests.test_input_validation,
        model_tests.test_output_format,
        model_tests.test_performance_requirements,
        model_tests.test_inference_speed,
        model_tests.test_prediction_stability,
        model_tests.test_edge_cases
    ]
    
    print("Running model tests...")
    for test_method in test_methods:
        try:
            test_method()
            print(f"✓ {test_method.__name__}")
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
    
    # API tests (if API is running)
    try:
        api_tests = APITestSuite("http://localhost:5000")
        
        api_test_methods = [
            api_tests.test_health_endpoint,
            api_tests.test_prediction_endpoint,
            api_tests.test_api_response_time,
            api_tests.test_concurrent_requests
        ]
        
        print("\nRunning API tests...")
        for test_method in api_test_methods:
            try:
                test_method()
                print(f"✓ {test_method.__name__}")
            except Exception as e:
                print(f"✗ {test_method.__name__}: {e}")
                
    except requests.exceptions.ConnectionError:
        print("API not running, skipping API tests")

if __name__ == "__main__":
    run_model_tests()
```

## 2. Version Control for ML

### Data Version Control (DVC)

```python
# DVC pipeline configuration (dvc.yaml)
dvc_pipeline_config = """
stages:
  data_preparation:
    cmd: python src/data_preparation.py --input data/raw --output data/processed
    deps:
    - data/raw
    - src/data_preparation.py
    outs:
    - data/processed/train.csv
    - data/processed/test.csv
    metrics:
    - reports/data_stats.json

  feature_engineering:
    cmd: python src/feature_engineering.py --input data/processed --output data/features
    deps:
    - data/processed/train.csv
    - data/processed/test.csv
    - src/feature_engineering.py
    outs:
    - data/features/train_features.csv
    - data/features/test_features.csv
    params:
    - feature_engineering.num_features
    - feature_engineering.scaling_method

  model_training:
    cmd: python src/train_model.py --data data/features --output models
    deps:
    - data/features/train_features.csv
    - src/train_model.py
    outs:
    - models/model.joblib
    - models/scaler.joblib
    params:
    - training.algorithm
    - training.hyperparameters
    metrics:
    - reports/training_metrics.json

  model_evaluation:
    cmd: python src/evaluate_model.py --model models/model.joblib --data data/features/test_features.csv
    deps:
    - models/model.joblib
    - data/features/test_features.csv
    - src/evaluate_model.py
    metrics:
    - reports/evaluation_metrics.json
    plots:
    - plots/confusion_matrix.png
    - plots/roc_curve.png
"""

# Parameters file (params.yaml)
params_config = """
feature_engineering:
  num_features: 10
  scaling_method: standard
  
training:
  algorithm: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
    
evaluation:
  test_size: 0.2
  cv_folds: 5
"""

# Model versioning with MLflow
class ModelVersionManager:
    """Manage model versions with MLflow Model Registry"""
    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model_uri: str, model_name: str, 
                      description: str = None, tags: Dict = None) -> str:
        """Register a new model version"""
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags or {}
        )
        
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        print(f"Registered model {model_name} version {model_version.version}")
        return model_version.version
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to a specific stage"""
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"Promoted {model_name} v{version} to {stage}")
    
    def get_latest_model(self, model_name: str, stage: str = "Production"):
        """Get latest model in specified stage"""
        
        latest_version = self.client.get_latest_versions(
            model_name, 
            stages=[stage]
        )
        
        if latest_version:
            return latest_version[0]
        return None
    
    def compare_models(self, model_name: str, version1: str, version2: str):
        """Compare two model versions"""
        
        # Get model versions
        mv1 = self.client.get_model_version(model_name, version1)
        mv2 = self.client.get_model_version(model_name, version2)
        
        # Get metrics for both versions
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)
        
        comparison = {
            'version1': {
                'version': version1,
                'metrics': run1.data.metrics,
                'params': run1.data.params,
                'created': mv1.creation_timestamp
            },
            'version2': {
                'version': version2,
                'metrics': run2.data.metrics,
                'params': run2.data.params,
                'created': mv2.creation_timestamp
            }
        }
        
        return comparison
    
    def archive_old_models(self, model_name: str, keep_versions: int = 5):
        """Archive old model versions"""
        
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
        
        # Archive versions beyond keep_versions
        for version in versions[keep_versions:]:
            if version.current_stage not in ['Production', 'Staging']:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage='Archived'
                )
                print(f"Archived {model_name} v{version.version}")

# Experiment tracking
class ExperimentTracker:
    """Track ML experiments with comprehensive logging"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def start_experiment(self, run_name: str = None, tags: Dict = None):
        """Start new experiment run"""
        
        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})
        return self.run
    
    def log_dataset_info(self, dataset: pd.DataFrame, dataset_name: str):
        """Log dataset information"""
        
        dataset_info = {
            f"{dataset_name}_samples": len(dataset),
            f"{dataset_name}_features": len(dataset.columns),
            f"{dataset_name}_memory_mb": dataset.memory_usage(deep=True).sum() / 1024**2
        }
        
        mlflow.log_metrics(dataset_info)
        
        # Log dataset schema
        schema = {
            'columns': list(dataset.columns),
            'dtypes': {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            'shape': dataset.shape,
            'missing_values': dataset.isnull().sum().to_dict()
        }
        
        mlflow.log_dict(schema, f"{dataset_name}_schema.json")
    
    def log_preprocessing_steps(self, steps: List[Dict]):
        """Log preprocessing steps"""
        
        mlflow.log_dict(steps, "preprocessing_steps.json")
        
        # Log step-specific metrics
        for i, step in enumerate(steps):
            if 'metrics' in step:
                for metric, value in step['metrics'].items():
                    mlflow.log_metric(f"preprocessing_step_{i}_{metric}", value)
    
    def log_hyperparameters(self, params: Dict):
        """Log hyperparameters"""
        
        # Flatten nested parameters
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_training_progress(self, epoch: int, metrics: Dict):
        """Log training progress"""
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=epoch)
    
    def log_model_artifacts(self, model, model_name: str, signature=None):
        """Log model and related artifacts"""
        
        # Log model
        if hasattr(model, 'predict'):  # Sklearn-like model
            mlflow.sklearn.log_model(
                model, 
                model_name,
                signature=signature
            )
        elif isinstance(model, torch.nn.Module):  # PyTorch model
            mlflow.pytorch.log_model(
                model,
                model_name,
                signature=signature
            )
    
    def log_evaluation_results(self, metrics: Dict, plots: Dict = None):
        """Log evaluation results"""
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log plots
        if plots:
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path, "plots")
    
    def log_feature_importance(self, feature_names: List[str], 
                              importance_values: np.ndarray):
        """Log feature importance"""
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Save as CSV
        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Create importance plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        plot_path = "feature_importance_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_artifact(plot_path)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def end_experiment(self):
        """End current experiment run"""
        
        mlflow.end_run()

# Example usage
def run_tracked_experiment():
    """Example of comprehensive experiment tracking"""
    
    # Initialize tracker
    tracker = ExperimentTracker("model_development")
    
    # Start experiment
    tracker.start_experiment(
        run_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={"team": "ml-team", "project": "classification"}
    )
    
    try:
        # Load and log dataset
        data = pd.read_csv("data/train.csv")
        tracker.log_dataset_info(data, "training_data")
        
        # Log preprocessing
        preprocessing_steps = [
            {
                "step": "missing_value_imputation",
                "method": "mean",
                "metrics": {"missing_before": 150, "missing_after": 0}
            },
            {
                "step": "feature_scaling",
                "method": "standard_scaler",
                "metrics": {"features_scaled": 10}
            }
        ]
        tracker.log_preprocessing_steps(preprocessing_steps)
        
        # Log hyperparameters
        params = {
            "model": {
                "algorithm": "random_forest",
                "n_estimators": 100,
                "max_depth": 10
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42
            }
        }
        tracker.log_hyperparameters(params)
        
        # Simulate training and log progress
        for epoch in range(10):
            metrics = {
                "train_accuracy": 0.8 + epoch * 0.01,
                "val_accuracy": 0.75 + epoch * 0.008,
                "train_loss": 0.5 - epoch * 0.03
            }
            tracker.log_training_progress(epoch, metrics)
        
        # Log final evaluation
        evaluation_metrics = {
            "test_accuracy": 0.87,
            "test_precision": 0.85,
            "test_recall": 0.89,
            "test_f1": 0.87
        }
        tracker.log_evaluation_results(evaluation_metrics)
        
        # Log feature importance (example)
        feature_names = [f"feature_{i}" for i in range(10)]
        importance_values = np.random.random(10)
        tracker.log_feature_importance(feature_names, importance_values)
        
    finally:
        tracker.end_experiment()

if __name__ == "__main__":
    # Example usage
    run_tracked_experiment()
```

## 💡 Key Takeaways

1. **Automate Everything**: Build pipelines that run automatically from data to deployment
2. **Version Control All Assets**: Track code, data, models, and experiments
3. **Test Rigorously**: Implement comprehensive testing for models and APIs
4. **Monitor Continuously**: Track performance, drift, and system health
5. **Iterate Rapidly**: Use CI/CD to deploy improvements quickly and safely

## 🏆 MLOps Maturity Levels

### Level 0: Manual Process
- Manual data analysis and model building
- Scripts for training and prediction
- Manual deployment and monitoring

### Level 1: ML Pipeline Automation
- Automated training pipelines
- Experiment tracking
- Model versioning

### Level 2: CI/CD Pipeline Automation
- Automated testing and validation
- Automated deployment
- Monitoring and alerting

### Level 3: Full MLOps
- Automated retraining
- Feature stores
- Model governance
- Advanced monitoring and observability

## 🚀 Production Readiness Framework

### Technical Requirements
- [ ] Automated end-to-end pipeline
- [ ] Comprehensive testing suite
- [ ] Version control for all artifacts
- [ ] CI/CD pipeline with quality gates
- [ ] Monitoring and alerting system

### Process Requirements
- [ ] Code review process
- [ ] Model validation checklist
- [ ] Deployment approval process
- [ ] Incident response procedures
- [ ] Model governance policies

### Documentation Requirements
- [ ] Pipeline documentation
- [ ] Model documentation
- [ ] API documentation
- [ ] Runbook for operations
- [ ] Troubleshooting guides

Remember: MLOps is not just about tools - it's about building a culture of collaboration between data scientists, engineers, and operations teams to deliver reliable AI systems!

Ready to build world-class ML infrastructure? Start with automating your training pipeline and grow from there! 🚀
