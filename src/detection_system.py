import joblib
import numpy as np
import pandas as pd
from datetime import datetime

class ElectricityTheftDetector:
    def __init__(self, model_path='models/theft_detector.pkl'):
        self.model_data = joblib.load(model_path)
        self.preprocessor = self.model_data['preprocessor']
        self.iso_forest = self.model_data['iso_forest']
        self.rf_classifier = self.model_data['rf_classifier']
        self.oc_svm = self.model_data['oc_svm']
        self.feature_columns = self.preprocessor.feature_columns
    
    def detect_theft(self, consumption_kwh, voltage, current, power_factor, 
                     hour=None, day_of_week=None, is_night=None):
        """Detect theft for single reading"""
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        if is_night is None:
            is_night = 1 if (hour >= 22 or hour <= 6) else 0
        
        # Create feature vector
        features = np.array([[
            hour, day_of_week, 0, is_night,  # is_weekend=0 for single reading
            consumption_kwh, voltage, current, power_factor,
            consumption_kwh, 0, 1.0,  # rolling stats approximated
            voltage * current, voltage * current * power_factor,
            0.95, 0, 0, 0  # power_ratio, anomaly flags
        ]])
        
        features_scaled = self.preprocessor.transform(features)
        
        # Get predictions from all models
        iso_pred = self.iso_forest.predict(features_scaled)[0]
        rf_pred = self.rf_classifier.predict(features_scaled)[0]
        oc_pred = self.oc_svm.predict(features_scaled)[0]
        
        # Ensemble decision
        theft_score = np.mean([iso_pred == -1, rf_pred, oc_pred == -1])
        is_theft = 1 if theft_score > 0.5 else 0
        
        return {
            'is_theft': bool(is_theft),
            'theft_probability': float(theft_score),
            'rf_prediction': int(rf_pred),
            'iso_forest_anomaly': iso_pred == -1,
            'oc_svm_anomaly': oc_pred == -1,
            'consumption_kwh': consumption_kwh,
            'voltage': voltage,
            'power_factor': power_factor
        }
    
    def batch_detect(self, data_df):
        """Detect theft for batch of readings"""
        data_df = data_df.copy()
        if 'timestamp' not in data_df.columns:
            data_df['timestamp'] = pd.Timestamp.now()

        # Create engineered features for raw batch inputs
        data_df = self.preprocessor.create_features(data_df)

        # Prepare features
        features_df = data_df[self.feature_columns].fillna(0)
        features_scaled = self.preprocessor.transform(features_df.values)
        
        # Predictions
        rf_probs = self.rf_classifier.predict_proba(features_scaled)
        if rf_probs.shape[1] == 1:
            rf_probs = np.hstack([1 - rf_probs, rf_probs])
        rf_preds = rf_probs[:, 1]
        iso_preds = self.iso_forest.predict(features_scaled)
        oc_preds = self.oc_svm.predict(features_scaled)
        
        # Ensemble scoring
        theft_scores = 0.4 * rf_preds + 0.3 * (iso_preds == -1) + 0.3 * (oc_preds == -1)
        
        data_df['theft_probability'] = theft_scores
        data_df['is_theft'] = (theft_scores > 0.5).astype(int)
        
        return data_df