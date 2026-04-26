import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_normal_consumption(n_samples=10000):
    """Generate normal consumption patterns"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    data = []
    for i in range(n_samples):
        hour = dates[i].hour
        # Normal residential pattern: peaks in evening, low at night
        base_load = np.random.normal(2.5, 0.5)
        daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
        random_noise = np.random.normal(0, 0.3)
        
        consumption = base_load * daily_pattern + random_noise
        consumption = max(0, consumption)
        
        data.append({
            'timestamp': dates[i],
            'consumption_kwh': consumption,
            'voltage': np.random.normal(230, 5),
            'current': consumption * 4.3,  # Approximate current
            'power_factor': np.random.uniform(0.85, 0.99),
            'is_theft': 0
        })
    
    return pd.DataFrame(data)

def generate_theft_consumption(n_samples=2000):
    """Generate theft patterns (night usage, sudden drops, etc.)"""
    np.random.seed(123)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    data = []
    for i in range(n_samples):
        hour = dates[i].hour
        theft_type = np.random.choice(['night_theft', 'bypass', 'tamper'])
        
        if theft_type == 'night_theft':
            # High usage at night when rates are low
            consumption = np.random.normal(4.5, 1.0) if hour >= 22 or hour <= 6 else 0.5
        elif theft_type == 'bypass':
            # Sudden drop in consumption
            consumption = np.random.normal(0.3, 0.1)
        else:  # tamper
            # Irregular patterns
            consumption = np.random.exponential(1.5)
        
        consumption = max(0, consumption)
        
        data.append({
            'timestamp': dates[i],
            'consumption_kwh': consumption,
            'voltage': np.random.normal(210, 15) if theft_type != 'bypass' else 240,
            'current': consumption * 4.3,
            'power_factor': np.random.uniform(0.4, 0.7) if theft_type != 'bypass' else 0.95,
            'is_theft': 1
        })
    
    return pd.DataFrame(data)

# Generate and save data
os.makedirs('data', exist_ok=True)
normal_df = generate_normal_consumption(10000)
theft_df = generate_theft_consumption(2000)

normal_df.to_csv('data/normal_consumption.csv', index=False)
theft_df.to_csv('data/theft_consumption.csv', index=False)

full_dataset = pd.concat([normal_df, theft_df], ignore_index=True)
full_dataset.to_csv('data/full_dataset.csv', index=False)

print("Dataset generated successfully!")
print(f"Normal samples: {len(normal_df)}")
print(f"Theft samples: {len(theft_df)}")
print(f"Total samples: {len(full_dataset)}")