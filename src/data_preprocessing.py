import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_combine_data(self):
        """Load and combine normal and theft data"""
        normal_df = pd.read_csv('data/normal_consumption.csv')
        theft_df = pd.read_csv('data/theft_consumption.csv')
        return pd.concat([normal_df, theft_df], ignore_index=True)
    
    def create_features(self, df):
        """Create engineered features for better detection"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Consumption patterns
        df['consumption_rolling_mean_24h'] = df['consumption_kwh'].rolling(24, min_periods=1).mean()
        df['consumption_rolling_std_24h'] = df['consumption_kwh'].rolling(24, min_periods=1).std()
        df['consumption_ratio'] = df['consumption_kwh'] / (df['consumption_rolling_mean_24h'] + 1e-8)
        
        # Electrical features
        df['apparent_power'] = df['voltage'] * df['current']
        df['real_power'] = df['apparent_power'] * df['power_factor']
        df['power_ratio'] = df['real_power'] / (df['apparent_power'] + 1e-8)
        
        # Anomaly indicators
        df['low_consumption'] = (df['consumption_kwh'] < 0.5).astype(int)
        df['high_pf_variation'] = (abs(df['power_factor'] - 0.9) > 0.2).astype(int)
        df['voltage_anomaly'] = (abs(df['voltage'] - 230) > 20).astype(int)
        
        self.feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'consumption_kwh', 'voltage', 'current', 'power_factor',
            'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
            'consumption_ratio', 'apparent_power', 'real_power',
            'power_ratio', 'low_consumption', 'high_pf_variation', 'voltage_anomaly'
        ]
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        df_features = df[self.feature_columns].fillna(0)
        X = df_features.values
        y = df['is_theft'].values
        
        return X, y
    
    def fit_transform(self, X):
        """Fit and transform features"""
        return self.scaler.fit_transform(X)
    
    def transform(self, X):
        """Transform new data"""
        return self.scaler.transform(X)