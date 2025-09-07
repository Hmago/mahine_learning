# Monitoring and Maintenance: Keeping AI Healthy in Production

Learn to monitor, maintain, and continuously improve your deep learning models in production. Master the art of detecting issues before they impact users and building self-healing AI systems.

## 🎯 What You'll Master

- **Model Monitoring**: Track performance, drift, and degradation
- **Data Quality**: Ensure input data remains reliable
- **Automated Alerting**: Catch issues before users notice
- **Continuous Improvement**: Keep models performing at their best

## 📚 The Monitoring Mindset

### Why Monitoring Matters

**Real-World Scenarios:**
```
Scenario 1: Recommendation system accuracy drops 20% due to new user behaviors
Scenario 2: Image classifier fails on new camera types not seen during training
Scenario 3: NLP model performance degrades due to language evolution
Scenario 4: Fraud detection misses new attack patterns
```

**The Production Reality:**
```
Research: Static dataset, controlled environment
Production: Dynamic data, evolving patterns, real users
```

Think of monitoring like being a doctor for your AI - you need to constantly check vital signs and diagnose problems early!

## 1. Model Performance Monitoring

### Core Metrics Tracking

**Concept:** Monitor model accuracy, latency, and business metrics in real-time.

**Analogy:** Like monitoring a car's dashboard - speed, fuel, engine temperature, etc.

```python
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import json
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

class ModelPerformanceMonitor:
    """Comprehensive model performance monitoring system"""
    
    def __init__(self, model_name: str, db_path: str = "model_metrics.db"):
        self.model_name = model_name
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=10000)
        self.prediction_buffer = deque(maxlen=10000)
        self.alert_thresholds = {}
        self.baseline_metrics = {}
        
        # Initialize database
        self._init_database()
        
        # Start background monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info(f"Performance monitor initialized for {model_name}")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                timestamp DATETIME,
                prediction_id TEXT,
                input_features TEXT,
                prediction TEXT,
                confidence REAL,
                ground_truth TEXT,
                inference_time REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                timestamp DATETIME,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction_id: str, input_features: np.ndarray, 
                      prediction: Any, confidence: float, 
                      inference_time: float, ground_truth: Any = None):
        """Log individual prediction for monitoring"""
        
        prediction_record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now(),
            'input_features': input_features.tolist() if isinstance(input_features, np.ndarray) else input_features,
            'prediction': prediction,
            'confidence': confidence,
            'inference_time': inference_time,
            'ground_truth': ground_truth
        }
        
        self.prediction_buffer.append(prediction_record)
        
        # Log to database
        self._log_to_database('predictions', prediction_record)
    
    def log_metric(self, metric_name: str, value: float, metadata: Dict = None):
        """Log custom metric"""
        
        metric_record = {
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'metric_value': value,
            'metadata': json.dumps(metadata or {})
        }
        
        self.metrics_buffer.append(metric_record)
        
        # Log to database
        self._log_to_database('model_metrics', metric_record)
        
        # Check alerts
        self._check_alert_thresholds(metric_name, value)
    
    def set_baseline_metrics(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
        logging.info(f"Baseline metrics set: {metrics}")
    
    def set_alert_threshold(self, metric_name: str, threshold_type: str, 
                           threshold_value: float, severity: str = 'warning'):
        """Set alert threshold for metric"""
        
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = []
        
        self.alert_thresholds[metric_name].append({
            'type': threshold_type,  # 'min', 'max', 'change'
            'value': threshold_value,
            'severity': severity
        })
        
        logging.info(f"Alert threshold set: {metric_name} {threshold_type} {threshold_value}")
    
    def _check_alert_thresholds(self, metric_name: str, value: float):
        """Check if metric triggers any alerts"""
        
        if metric_name not in self.alert_thresholds:
            return
        
        for threshold in self.alert_thresholds[metric_name]:
            alert_triggered = False
            message = ""
            
            if threshold['type'] == 'min' and value < threshold['value']:
                alert_triggered = True
                message = f"{metric_name} below threshold: {value:.4f} < {threshold['value']:.4f}"
            
            elif threshold['type'] == 'max' and value > threshold['value']:
                alert_triggered = True
                message = f"{metric_name} above threshold: {value:.4f} > {threshold['value']:.4f}"
            
            elif threshold['type'] == 'change' and metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                change_percent = abs(value - baseline) / baseline * 100
                if change_percent > threshold['value']:
                    alert_triggered = True
                    message = f"{metric_name} changed significantly: {change_percent:.2f}% from baseline"
            
            if alert_triggered:
                self._trigger_alert(metric_name, threshold['severity'], message)
    
    def _trigger_alert(self, metric_name: str, severity: str, message: str):
        """Trigger an alert"""
        
        alert_record = {
            'timestamp': datetime.now(),
            'alert_type': metric_name,
            'severity': severity,
            'message': message,
            'resolved': False
        }
        
        # Log alert
        self._log_to_database('alerts', alert_record)
        
        # Log to console
        logging.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # Here you could add integrations with:
        # - Slack/Teams notifications
        # - Email alerts
        # - PagerDuty
        # - Custom webhooks
    
    def _log_to_database(self, table: str, record: Dict):
        """Log record to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if table == 'model_metrics':
                cursor.execute('''
                    INSERT INTO model_metrics (model_name, timestamp, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (self.model_name, record['timestamp'], record['metric_name'], 
                     record['metric_value'], record['metadata']))
            
            elif table == 'predictions':
                cursor.execute('''
                    INSERT INTO predictions (model_name, timestamp, prediction_id, input_features, 
                                           prediction, confidence, ground_truth, inference_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (self.model_name, record['timestamp'], record['prediction_id'],
                     json.dumps(record['input_features']), json.dumps(record['prediction']),
                     record['confidence'], json.dumps(record['ground_truth']), record['inference_time']))
            
            elif table == 'alerts':
                cursor.execute('''
                    INSERT INTO alerts (model_name, timestamp, alert_type, severity, message, resolved)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (self.model_name, record['timestamp'], record['alert_type'],
                     record['severity'], record['message'], record['resolved']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Failed to log to database: {e}")
    
    def _background_monitor(self):
        """Background monitoring thread"""
        
        while self.monitoring_active:
            try:
                # Calculate real-time metrics
                self._calculate_realtime_metrics()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logging.error(f"Background monitoring error: {e}")
                time.sleep(5)
    
    def _calculate_realtime_metrics(self):
        """Calculate real-time performance metrics"""
        
        if len(self.prediction_buffer) < 10:
            return
        
        recent_predictions = list(self.prediction_buffer)[-100:]  # Last 100 predictions
        
        # Average confidence
        avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
        self.log_metric('avg_confidence', avg_confidence)
        
        # Average inference time
        avg_inference_time = np.mean([p['inference_time'] for p in recent_predictions])
        self.log_metric('avg_inference_time_ms', avg_inference_time * 1000)
        
        # Prediction distribution
        predictions = [p['prediction'] for p in recent_predictions]
        if predictions and isinstance(predictions[0], (int, str)):
            from collections import Counter
            pred_dist = Counter(predictions)
            total = len(predictions)
            
            # Check for prediction bias
            most_common_ratio = pred_dist.most_common(1)[0][1] / total
            self.log_metric('prediction_bias', most_common_ratio)
        
        # Calculate accuracy if ground truth available
        with_ground_truth = [p for p in recent_predictions if p['ground_truth'] is not None]
        if with_ground_truth:
            correct = sum(1 for p in with_ground_truth 
                         if p['prediction'] == p['ground_truth'])
            accuracy = correct / len(with_ground_truth)
            self.log_metric('recent_accuracy', accuracy)
    
    def get_performance_report(self, hours: int = 24) -> Dict:
        """Generate performance report for specified time period"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get metrics for time period
        metrics_df = pd.read_sql_query('''
            SELECT * FROM model_metrics 
            WHERE model_name = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn, params=(self.model_name,))
        
        # Get predictions for time period
        predictions_df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE model_name = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn, params=(self.model_name,))
        
        # Get alerts for time period
        alerts_df = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE model_name = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn, params=(self.model_name,))
        
        conn.close()
        
        report = {
            'time_period_hours': hours,
            'total_predictions': len(predictions_df),
            'total_alerts': len(alerts_df),
            'unresolved_alerts': len(alerts_df[alerts_df['resolved'] == False]),
            'metrics_summary': {},
            'alerts_summary': alerts_df.groupby(['alert_type', 'severity']).size().to_dict() if not alerts_df.empty else {}
        }
        
        # Summarize metrics
        if not metrics_df.empty:
            for metric_name in metrics_df['metric_name'].unique():
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name]['metric_value']
                report['metrics_summary'][metric_name] = {
                    'mean': metric_data.mean(),
                    'std': metric_data.std(),
                    'min': metric_data.min(),
                    'max': metric_data.max(),
                    'latest': metric_data.iloc[-1] if not metric_data.empty else None
                }
        
        # Performance trends
        if not predictions_df.empty:
            report['avg_inference_time'] = predictions_df['inference_time'].mean()
            report['avg_confidence'] = predictions_df['confidence'].mean()
            
            # Accuracy if ground truth available
            with_truth = predictions_df[predictions_df['ground_truth'].notna()]
            if not with_truth.empty:
                # Parse JSON strings and compare
                accuracies = []
                for _, row in with_truth.iterrows():
                    try:
                        pred = json.loads(row['prediction'])
                        truth = json.loads(row['ground_truth'])
                        accuracies.append(1 if pred == truth else 0)
                    except:
                        continue
                
                if accuracies:
                    report['accuracy'] = np.mean(accuracies)
        
        return report
    
    def visualize_metrics(self, metric_names: List[str], hours: int = 24):
        """Visualize metrics over time"""
        
        conn = sqlite3.connect(self.db_path)
        
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            df = pd.read_sql_query('''
                SELECT timestamp, metric_value FROM model_metrics 
                WHERE model_name = ? AND metric_name = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours), conn, params=(self.model_name, metric_name))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                axes[i].plot(df['timestamp'], df['metric_value'])
                axes[i].set_title(f'{metric_name} over last {hours} hours')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        conn.close()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

# Example usage
def setup_model_monitoring():
    """Example of setting up comprehensive model monitoring"""
    
    # Initialize monitor
    monitor = ModelPerformanceMonitor("recommendation_model_v1")
    
    # Set baseline metrics (from validation)
    monitor.set_baseline_metrics({
        'accuracy': 0.85,
        'avg_confidence': 0.75,
        'avg_inference_time_ms': 45
    })
    
    # Set alert thresholds
    monitor.set_alert_threshold('accuracy', 'min', 0.80, 'critical')
    monitor.set_alert_threshold('accuracy', 'change', 10, 'warning')  # 10% change
    monitor.set_alert_threshold('avg_inference_time_ms', 'max', 100, 'warning')
    monitor.set_alert_threshold('avg_confidence', 'min', 0.60, 'warning')
    
    return monitor

# Integration with model serving
class MonitoredModelServer:
    """Model server with integrated monitoring"""
    
    def __init__(self, model, monitor: ModelPerformanceMonitor):
        self.model = model
        self.monitor = monitor
        self.prediction_counter = 0
    
    def predict(self, input_data, ground_truth=None):
        """Make prediction with monitoring"""
        
        start_time = time.time()
        
        # Make prediction
        prediction = self.model(input_data)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))
        
        inference_time = time.time() - start_time
        
        # Log to monitor
        self.prediction_counter += 1
        prediction_id = f"pred_{self.prediction_counter}_{int(time.time())}"
        
        self.monitor.log_prediction(
            prediction_id=prediction_id,
            input_features=input_data,
            prediction=predicted_class,
            confidence=confidence,
            inference_time=inference_time,
            ground_truth=ground_truth
        )
        
        return predicted_class, confidence

# Example monitoring workflow
if __name__ == "__main__":
    # Setup monitoring
    monitor = setup_model_monitoring()
    
    # Simulate predictions with monitoring
    for i in range(100):
        # Simulate prediction data
        input_data = np.random.randn(10)
        prediction = np.random.randint(0, 5)
        confidence = np.random.uniform(0.5, 0.95)
        inference_time = np.random.uniform(0.02, 0.08)
        ground_truth = prediction if np.random.random() > 0.15 else np.random.randint(0, 5)
        
        # Log prediction
        monitor.log_prediction(
            prediction_id=f"test_pred_{i}",
            input_features=input_data,
            prediction=prediction,
            confidence=confidence,
            inference_time=inference_time,
            ground_truth=ground_truth
        )
        
        time.sleep(0.1)  # Simulate real-time predictions
    
    # Generate report
    report = monitor.get_performance_report(hours=1)
    print("Performance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Stop monitoring
    monitor.stop_monitoring()
```

### Data Drift Detection

**Concept:** Detect when input data distribution changes from training data.

**Analogy:** Like noticing your GPS routes have changed - the roads are the same, but traffic patterns are different.

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import jensen_shannon_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Union, Dict, List, Tuple

class DataDriftDetector:
    """Detect data drift in production inputs"""
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str] = None):
        """
        Initialize drift detector with reference data
        
        Args:
            reference_data: Training/validation data as reference
            feature_names: Names of features for reporting
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        self.num_features = reference_data.shape[1]
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
        
        # PCA for multivariate drift detection
        self.pca = PCA(n_components=min(10, reference_data.shape[1]))
        self.scaler = StandardScaler()
        
        # Fit PCA on reference data
        scaled_ref = self.scaler.fit_transform(reference_data)
        self.reference_pca = self.pca.fit_transform(scaled_ref)
        
        print(f"Drift detector initialized with {len(reference_data)} reference samples")
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict:
        """Calculate statistical properties of data"""
        
        stats_dict = {}
        
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            
            stats_dict[f'feature_{i}'] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q50': np.percentile(feature_data, 50),
                'q75': np.percentile(feature_data, 75),
                'skew': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data)
            }
        
        return stats_dict
    
    def detect_drift(self, new_data: np.ndarray, 
                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Detect drift between reference and new data
        
        Args:
            new_data: New data to test for drift
            alpha: Significance level for statistical tests
            
        Returns:
            Dictionary with drift detection results
        """
        
        if new_data.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {new_data.shape[1]}")
        
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'multivariate_drift': {},
            'summary': {}
        }
        
        # 1. Univariate drift detection (per feature)
        feature_drift_scores = []
        
        for i in range(self.num_features):
            feature_name = self.feature_names[i]
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_feature, new_feature)
            
            # Mann-Whitney U test (for distribution differences)
            mw_stat, mw_pvalue = stats.mannwhitneyu(ref_feature, new_feature, 
                                                    alternative='two-sided')
            
            # Jensen-Shannon divergence (for probability distributions)
            js_distance = self._jensen_shannon_distance(ref_feature, new_feature)
            
            # Statistical moments comparison
            ref_stats = self.reference_stats[f'feature_{i}']
            new_stats = self._calculate_statistics(new_data[:, i:i+1])['feature_0']
            
            # Relative changes in statistics
            mean_change = abs(new_stats['mean'] - ref_stats['mean']) / (abs(ref_stats['mean']) + 1e-8)
            std_change = abs(new_stats['std'] - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            
            feature_drift = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ks_drift': ks_pvalue < alpha,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'mw_drift': mw_pvalue < alpha,
                'js_distance': js_distance,
                'js_drift': js_distance > 0.1,  # Threshold for JS distance
                'mean_change': mean_change,
                'std_change': std_change,
                'statistical_drift': mean_change > 0.2 or std_change > 0.2,
                'drift_detected': ks_pvalue < alpha or js_distance > 0.1 or mean_change > 0.2
            }
            
            drift_results['feature_drifts'][feature_name] = feature_drift
            
            # Aggregate drift score for this feature
            feature_score = (1 - ks_pvalue) * 0.4 + js_distance * 0.4 + mean_change * 0.2
            feature_drift_scores.append(feature_score)
        
        # 2. Multivariate drift detection
        try:
            # Transform new data using fitted scaler and PCA
            scaled_new = self.scaler.transform(new_data)
            new_pca = self.pca.transform(scaled_new)
            
            # Compare PCA distributions
            multivariate_ks_scores = []
            for i in range(self.reference_pca.shape[1]):
                ks_stat, ks_pvalue = stats.ks_2samp(
                    self.reference_pca[:, i], 
                    new_pca[:, i]
                )
                multivariate_ks_scores.append(1 - ks_pvalue)
            
            multivariate_drift_score = np.mean(multivariate_ks_scores)
            
            drift_results['multivariate_drift'] = {
                'pca_drift_score': multivariate_drift_score,
                'drift_detected': multivariate_drift_score > 0.8,
                'component_scores': multivariate_ks_scores
            }
            
        except Exception as e:
            warnings.warn(f"Multivariate drift detection failed: {e}")
            drift_results['multivariate_drift'] = {'error': str(e)}
        
        # 3. Overall drift assessment
        overall_drift_score = np.mean(feature_drift_scores)
        drift_detected = (
            overall_drift_score > 0.5 or 
            sum(1 for fd in drift_results['feature_drifts'].values() 
                if fd['drift_detected']) > len(self.feature_names) * 0.3
        )
        
        drift_results['drift_detected'] = drift_detected
        drift_results['drift_score'] = overall_drift_score
        
        # 4. Summary
        drifted_features = [name for name, fd in drift_results['feature_drifts'].items() 
                           if fd['drift_detected']]
        
        drift_results['summary'] = {
            'total_features': len(self.feature_names),
            'drifted_features': len(drifted_features),
            'drifted_feature_names': drifted_features,
            'drift_percentage': len(drifted_features) / len(self.feature_names) * 100,
            'overall_drift_score': overall_drift_score,
            'recommendation': self._get_drift_recommendation(drift_results)
        }
        
        return drift_results
    
    def _jensen_shannon_distance(self, X: np.ndarray, Y: np.ndarray, 
                                num_bins: int = 50) -> float:
        """Calculate Jensen-Shannon distance between two distributions"""
        
        try:
            # Create histograms
            min_val = min(X.min(), Y.min())
            max_val = max(X.max(), Y.max())
            bins = np.linspace(min_val, max_val, num_bins)
            
            hist_X, _ = np.histogram(X, bins=bins, density=True)
            hist_Y, _ = np.histogram(Y, bins=bins, density=True)
            
            # Normalize to probabilities
            hist_X = hist_X / (hist_X.sum() + 1e-10)
            hist_Y = hist_Y / (hist_Y.sum() + 1e-10)
            
            # Add small epsilon to avoid log(0)
            hist_X = hist_X + 1e-10
            hist_Y = hist_Y + 1e-10
            
            # Calculate Jensen-Shannon divergence
            M = 0.5 * (hist_X + hist_Y)
            js_div = 0.5 * stats.entropy(hist_X, M) + 0.5 * stats.entropy(hist_Y, M)
            
            # Convert to distance (square root of divergence)
            js_distance = np.sqrt(js_div)
            
            return js_distance
            
        except Exception as e:
            warnings.warn(f"JS distance calculation failed: {e}")
            return 0.0
    
    def _get_drift_recommendation(self, drift_results: Dict) -> str:
        """Get recommendation based on drift detection results"""
        
        drift_score = drift_results['drift_score']
        num_drifted = drift_results['summary']['drifted_features']
        total_features = drift_results['summary']['total_features']
        
        if not drift_results['drift_detected']:
            return "No significant drift detected. Continue monitoring."
        
        elif drift_score > 0.8 or num_drifted > total_features * 0.5:
            return "CRITICAL: High drift detected. Consider retraining model immediately."
        
        elif drift_score > 0.6 or num_drifted > total_features * 0.3:
            return "WARNING: Moderate drift detected. Plan model retraining soon."
        
        else:
            return "NOTICE: Minor drift detected. Monitor closely and consider retraining."
    
    def update_reference(self, new_reference_data: np.ndarray):
        """Update reference data (e.g., with recent production data)"""
        
        self.reference_data = new_reference_data
        self.reference_stats = self._calculate_statistics(new_reference_data)
        
        # Refit PCA
        scaled_ref = self.scaler.fit_transform(new_reference_data)
        self.reference_pca = self.pca.fit_transform(scaled_ref)
        
        print(f"Reference data updated with {len(new_reference_data)} samples")

# Continuous drift monitoring
class ContinuousDriftMonitor:
    """Monitor data drift continuously in production"""
    
    def __init__(self, detector: DataDriftDetector, window_size: int = 1000):
        self.detector = detector
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.drift_history = []
        
    def add_sample(self, sample: np.ndarray):
        """Add new sample to monitoring buffer"""
        self.data_buffer.append(sample)
        
        # Check drift when buffer is full
        if len(self.data_buffer) == self.window_size:
            self._check_drift()
    
    def _check_drift(self):
        """Check for drift with current buffer"""
        
        current_data = np.array(list(self.data_buffer))
        drift_results = self.detector.detect_drift(current_data)
        
        # Store drift results with timestamp
        drift_record = {
            'timestamp': datetime.now(),
            'drift_detected': drift_results['drift_detected'],
            'drift_score': drift_results['drift_score'],
            'drifted_features': drift_results['summary']['drifted_features'],
            'recommendation': drift_results['summary']['recommendation']
        }
        
        self.drift_history.append(drift_record)
        
        # Alert if drift detected
        if drift_results['drift_detected']:
            logging.warning(f"DRIFT ALERT: {drift_results['summary']['recommendation']}")
            
            # Here you could trigger:
            # - Automatic model retraining
            # - Notifications to ML team
            # - Model rollback procedures
    
    def get_drift_report(self, hours: int = 24) -> Dict:
        """Get drift monitoring report"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_drift = [d for d in self.drift_history if d['timestamp'] > cutoff_time]
        
        if not recent_drift:
            return {'message': 'No drift checks in specified time period'}
        
        drift_events = sum(1 for d in recent_drift if d['drift_detected'])
        
        return {
            'time_period_hours': hours,
            'total_drift_checks': len(recent_drift),
            'drift_events': drift_events,
            'drift_rate': drift_events / len(recent_drift),
            'latest_drift_score': recent_drift[-1]['drift_score'],
            'latest_recommendation': recent_drift[-1]['recommendation'],
            'drift_trend': [d['drift_score'] for d in recent_drift[-10:]]  # Last 10 scores
        }

# Example usage
def demo_drift_detection():
    """Demonstrate drift detection capabilities"""
    
    # Generate reference data (training data simulation)
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, (1000, 5))
    feature_names = ['age', 'income', 'score', 'clicks', 'time_spent']
    
    # Initialize drift detector
    detector = DataDriftDetector(reference_data, feature_names)
    
    # Test Case 1: No drift (same distribution)
    no_drift_data = np.random.normal(0, 1, (500, 5))
    results_no_drift = detector.detect_drift(no_drift_data)
    print("No Drift Test:")
    print(f"Drift detected: {results_no_drift['drift_detected']}")
    print(f"Drift score: {results_no_drift['drift_score']:.3f}")
    print(f"Recommendation: {results_no_drift['summary']['recommendation']}")
    print()
    
    # Test Case 2: Mean shift (concept drift)
    drift_data = np.random.normal(1.5, 1, (500, 5))  # Mean shifted
    results_drift = detector.detect_drift(drift_data)
    print("Mean Shift Drift Test:")
    print(f"Drift detected: {results_drift['drift_detected']}")
    print(f"Drift score: {results_drift['drift_score']:.3f}")
    print(f"Drifted features: {results_drift['summary']['drifted_feature_names']}")
    print(f"Recommendation: {results_drift['summary']['recommendation']}")
    print()
    
    # Test Case 3: Variance change
    variance_drift_data = np.random.normal(0, 3, (500, 5))  # Higher variance
    results_variance = detector.detect_drift(variance_drift_data)
    print("Variance Drift Test:")
    print(f"Drift detected: {results_variance['drift_detected']}")
    print(f"Drift score: {results_variance['drift_score']:.3f}")
    print(f"Recommendation: {results_variance['summary']['recommendation']}")
    
    return detector

if __name__ == "__main__":
    # Demo drift detection
    detector = demo_drift_detection()
```

## 2. Automated Alerting Systems

### Multi-Channel Alert System

```python
import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
import logging
from datetime import datetime
import asyncio
import aiohttp

class AlertManager:
    """Comprehensive alerting system for ML models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.suppression_rules = {}
        
    def send_alert(self, alert_type: str, severity: str, message: str, 
                  metadata: Dict = None, channels: List[str] = None):
        """Send alert through multiple channels"""
        
        alert = {
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {},
            'channels': channels or ['console']
        }
        
        # Check suppression rules
        if self._is_suppressed(alert):
            logging.info(f"Alert suppressed: {alert_type}")
            return
        
        # Store alert
        self.alert_history.append(alert)
        
        # Send through configured channels
        for channel in alert['channels']:
            try:
                if channel == 'email':
                    self._send_email_alert(alert)
                elif channel == 'slack':
                    self._send_slack_alert(alert)
                elif channel == 'webhook':
                    self._send_webhook_alert(alert)
                elif channel == 'console':
                    self._send_console_alert(alert)
                elif channel == 'pagerduty':
                    self._send_pagerduty_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        
        if 'email' not in self.config:
            return
        
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['sender']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"[{alert['severity'].upper()}] ML Model Alert: {alert['alert_type']}"
        
        body = f"""
        Alert Details:
        - Type: {alert['alert_type']}
        - Severity: {alert['severity']}
        - Time: {alert['timestamp']}
        - Message: {alert['message']}
        
        Metadata: {json.dumps(alert['metadata'], indent=2)}
        
        This is an automated alert from your ML monitoring system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
    
    def _send_slack_alert(self, alert: Dict):
        """Send Slack alert"""
        
        if 'slack' not in self.config:
            return
        
        slack_config = self.config['slack']
        
        # Color coding by severity
        color_map = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#0000FF'
        }
        
        payload = {
            'channel': slack_config['channel'],
            'username': 'ML Monitor',
            'icon_emoji': ':robot_face:',
            'attachments': [{
                'color': color_map.get(alert['severity'], '#808080'),
                'title': f"{alert['alert_type']} - {alert['severity'].upper()}",
                'text': alert['message'],
                'fields': [
                    {'title': 'Time', 'value': str(alert['timestamp']), 'short': True},
                    {'title': 'Model', 'value': alert['metadata'].get('model_name', 'Unknown'), 'short': True}
                ],
                'footer': 'ML Monitoring System',
                'ts': int(alert['timestamp'].timestamp())
            }]
        }
        
        response = requests.post(slack_config['webhook_url'], json=payload)
        response.raise_for_status()
    
    def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert"""
        
        if 'webhook' not in self.config:
            return
        
        webhook_config = self.config['webhook']
        
        payload = {
            'alert': alert,
            'source': 'ml_monitoring_system'
        }
        
        response = requests.post(
            webhook_config['url'],
            json=payload,
            headers=webhook_config.get('headers', {}),
            timeout=30
        )
        response.raise_for_status()
    
    def _send_console_alert(self, alert: Dict):
        """Send console alert"""
        
        severity_colors = {
            'critical': '\033[91m',  # Red
            'warning': '\033[93m',   # Yellow
            'info': '\033[94m',      # Blue
        }
        
        color = severity_colors.get(alert['severity'], '\033[0m')
        reset_color = '\033[0m'
        
        print(f"{color}[{alert['severity'].upper()}] {alert['timestamp']}: {alert['message']}{reset_color}")
    
    def _send_pagerduty_alert(self, alert: Dict):
        """Send PagerDuty alert"""
        
        if 'pagerduty' not in self.config:
            return
        
        pd_config = self.config['pagerduty']
        
        # Only send critical alerts to PagerDuty
        if alert['severity'] != 'critical':
            return
        
        payload = {
            'routing_key': pd_config['integration_key'],
            'event_action': 'trigger',
            'dedup_key': f"{alert['alert_type']}_{alert['metadata'].get('model_name', 'unknown')}",
            'payload': {
                'summary': f"{alert['alert_type']}: {alert['message']}",
                'severity': 'critical',
                'source': alert['metadata'].get('model_name', 'ml_model'),
                'component': 'ml_monitoring',
                'group': 'ml_infrastructure',
                'class': alert['alert_type'],
                'custom_details': alert['metadata']
            }
        }
        
        response = requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json=payload,
            timeout=30
        )
        response.raise_for_status()
    
    def _is_suppressed(self, alert: Dict) -> bool:
        """Check if alert should be suppressed"""
        
        alert_key = f"{alert['alert_type']}_{alert['severity']}"
        
        if alert_key in self.suppression_rules:
            rule = self.suppression_rules[alert_key]
            
            # Check if within suppression window
            recent_alerts = [
                a for a in self.alert_history
                if (a['alert_type'] == alert['alert_type'] and 
                    a['severity'] == alert['severity'] and
                    (alert['timestamp'] - a['timestamp']).total_seconds() < rule['window_seconds'])
            ]
            
            if len(recent_alerts) >= rule['max_alerts']:
                return True
        
        return False
    
    def add_suppression_rule(self, alert_type: str, severity: str, 
                           max_alerts: int, window_seconds: int):
        """Add alert suppression rule"""
        
        rule_key = f"{alert_type}_{severity}"
        self.suppression_rules[rule_key] = {
            'max_alerts': max_alerts,
            'window_seconds': window_seconds
        }
        
        logging.info(f"Suppression rule added: {rule_key}")

# Smart alerting with escalation
class SmartAlertEscalation:
    """Intelligent alert escalation based on severity and response time"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.escalation_rules = {}
        self.pending_alerts = {}
        
    def add_escalation_rule(self, alert_type: str, escalation_levels: List[Dict]):
        """
        Add escalation rule
        
        escalation_levels: [
            {'delay_minutes': 5, 'channels': ['slack'], 'severity': 'warning'},
            {'delay_minutes': 15, 'channels': ['email', 'pagerduty'], 'severity': 'critical'}
        ]
        """
        self.escalation_rules[alert_type] = escalation_levels
    
    async def handle_alert_with_escalation(self, alert_type: str, message: str, 
                                         initial_severity: str, metadata: Dict = None):
        """Handle alert with automatic escalation"""
        
        alert_id = f"{alert_type}_{int(datetime.now().timestamp())}"
        
        # Send initial alert
        initial_channels = self._get_initial_channels(initial_severity)
        self.alert_manager.send_alert(
            alert_type, initial_severity, message, metadata, initial_channels
        )
        
        # Start escalation if rules exist
        if alert_type in self.escalation_rules:
            self.pending_alerts[alert_id] = {
                'alert_type': alert_type,
                'message': message,
                'metadata': metadata,
                'start_time': datetime.now(),
                'escalated': False
            }
            
            # Schedule escalation
            asyncio.create_task(self._escalate_alert(alert_id))
    
    async def _escalate_alert(self, alert_id: str):
        """Escalate alert according to rules"""
        
        if alert_id not in self.pending_alerts:
            return
        
        alert_info = self.pending_alerts[alert_id]
        escalation_levels = self.escalation_rules[alert_info['alert_type']]
        
        for level in escalation_levels:
            # Wait for delay
            await asyncio.sleep(level['delay_minutes'] * 60)
            
            # Check if alert was resolved
            if alert_id not in self.pending_alerts:
                return
            
            # Escalate
            escalated_message = f"[ESCALATED] {alert_info['message']} - No response in {level['delay_minutes']} minutes"
            
            self.alert_manager.send_alert(
                alert_info['alert_type'],
                level['severity'],
                escalated_message,
                alert_info['metadata'],
                level['channels']
            )
            
            self.pending_alerts[alert_id]['escalated'] = True
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved to stop escalation"""
        if alert_id in self.pending_alerts:
            del self.pending_alerts[alert_id]
    
    def _get_initial_channels(self, severity: str) -> List[str]:
        """Get appropriate channels for initial alert based on severity"""
        
        if severity == 'critical':
            return ['console', 'slack', 'email']
        elif severity == 'warning':
            return ['console', 'slack']
        else:
            return ['console']

# Example configuration and usage
def setup_production_alerting():
    """Setup production alerting system"""
    
    alert_config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'ml-alerts@company.com',
            'password': 'app_password',
            'sender': 'ml-alerts@company.com',
            'recipients': ['ml-team@company.com', 'devops@company.com']
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'channel': '#ml-alerts'
        },
        'webhook': {
            'url': 'https://api.company.com/alerts',
            'headers': {'Authorization': 'Bearer YOUR_TOKEN'}
        },
        'pagerduty': {
            'integration_key': 'YOUR_PAGERDUTY_INTEGRATION_KEY'
        }
    }
    
    # Initialize alert manager
    alert_manager = AlertManager(alert_config)
    
    # Add suppression rules (prevent spam)
    alert_manager.add_suppression_rule('accuracy_drop', 'warning', max_alerts=3, window_seconds=3600)
    alert_manager.add_suppression_rule('high_latency', 'warning', max_alerts=5, window_seconds=1800)
    
    # Setup smart escalation
    escalation = SmartAlertEscalation(alert_manager)
    
    # Add escalation rules
    escalation.add_escalation_rule('model_failure', [
        {'delay_minutes': 5, 'channels': ['slack'], 'severity': 'warning'},
        {'delay_minutes': 15, 'channels': ['email'], 'severity': 'critical'},
        {'delay_minutes': 30, 'channels': ['pagerduty'], 'severity': 'critical'}
    ])
    
    return alert_manager, escalation

# Integration with monitoring
class IntegratedMonitoringSystem:
    """Complete monitoring and alerting system"""
    
    def __init__(self, model_name: str, alert_config: Dict):
        self.model_name = model_name
        self.performance_monitor = ModelPerformanceMonitor(model_name)
        self.alert_manager = AlertManager(alert_config)
        self.escalation = SmartAlertEscalation(self.alert_manager)
        
        # Setup monitoring alerts
        self._setup_monitoring_alerts()
    
    def _setup_monitoring_alerts(self):
        """Setup automatic alerts based on monitoring metrics"""
        
        # Performance alerts
        self.performance_monitor.set_alert_threshold('accuracy', 'min', 0.80, 'critical')
        self.performance_monitor.set_alert_threshold('accuracy', 'change', 10, 'warning')
        self.performance_monitor.set_alert_threshold('avg_inference_time_ms', 'max', 200, 'warning')
        self.performance_monitor.set_alert_threshold('prediction_bias', 'max', 0.8, 'warning')
        
        # Override alert sending to use our alert manager
        original_trigger_alert = self.performance_monitor._trigger_alert
        
        def enhanced_trigger_alert(metric_name: str, severity: str, message: str):
            # Call original method
            original_trigger_alert(metric_name, severity, message)
            
            # Send through alert manager
            self.alert_manager.send_alert(
                alert_type=f"metric_{metric_name}",
                severity=severity,
                message=message,
                metadata={'model_name': self.model_name, 'metric': metric_name},
                channels=['console', 'slack'] if severity != 'critical' else ['console', 'slack', 'email']
            )
        
        self.performance_monitor._trigger_alert = enhanced_trigger_alert

if __name__ == "__main__":
    # Demo integrated monitoring and alerting
    alert_config = {
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/DEMO/WEBHOOK/URL',
            'channel': '#ml-alerts'
        }
    }
    
    system = IntegratedMonitoringSystem("demo_model", alert_config)
    
    # Simulate some metrics that would trigger alerts
    system.performance_monitor.log_metric('accuracy', 0.75)  # Below threshold
    system.performance_monitor.log_metric('avg_inference_time_ms', 250)  # Above threshold
    
    print("Integrated monitoring and alerting system demo completed!")
```

## 💡 Key Takeaways

1. **Monitor Everything**: Track model performance, data quality, and system health
2. **Detect Issues Early**: Use drift detection and anomaly detection to catch problems
3. **Alert Smartly**: Avoid alert fatigue with proper suppression and escalation
4. **Automate Response**: Build self-healing systems where possible
5. **Learn and Improve**: Use monitoring data to continuously improve your models

Remember: Good monitoring is like having a 24/7 health check for your AI systems - it keeps them running smoothly and catches problems before they impact users!

Ready to build bulletproof monitoring for your AI? Start with basic performance tracking and expand from there! 🔍
