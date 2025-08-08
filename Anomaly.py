import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Simulated data (replace with actual data loading)
np.random.seed(42)
n_services = 600
n_records = 10000
services = [f"svc_{i}" for i in range(n_services)]
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2025-08-01', periods=n_records, freq='min'),
    'service_name': np.random.choice(services, size=n_records),
    'TPS': np.random.lognormal(mean=2, sigma=1, size=n_records).clip(1, 100),
    'RT': np.random.lognormal(mean=5, sigma=1, size=n_records).clip(50, 5000)
})

# Add some anomalies (high RT, low TPS for specific services)
anomaly_services = ['svc_10', 'svc_50', 'svc_100', 'svc_200', 'svc_300']
anomaly_indices = data['service_name'].isin(anomaly_services)
data.loc[anomaly_indices, 'RT'] *= 2  # Double RT for anomalies
data.loc[anomaly_indices, 'TPS'] *= 0.5  # Halve TPS for anomalies

# Feature engineering
data['response_time_per_transaction'] = data['RT'] / data['TPS']
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Aggregate per service
agg_data = data.groupby('service_name').agg({
    'RT': ['mean', 'median', 'max'],
    'TPS': ['mean', 'median', 'max'],
    'response_time_per_transaction': ['mean', 'median', 'max'],
    'hour': 'mean',  # Average hour to capture temporal patterns
    'day_of_week': 'mean'
}).reset_index()

# Flatten column names
agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_data.columns]

# Features for model
features = [
    'RT_mean', 'RT_median', 'RT_max',
    'TPS_mean', 'TPS_median', 'TPS_max',
    'response_time_per_transaction_mean',
    'response_time_per_transaction_median',
    'response_time_per_transaction_max',
    'hour_mean', 'day_of_week_mean'
]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(agg_data[features])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
anomaly_scores = iso_forest.fit_predict(X)
agg_data['anomaly_score'] = -iso_forest.score_samples(X)  # Higher score = more anomalous

# Get top 5 worst-performing services
top_5_worst = agg_data[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].sort_values(by='anomaly_score', ascending=False).head(5)

# Output results
print("Top 5 worst-performing microservices:")
print(top_5_worst[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].round(2))

# Save results to CSV
top_5_worst.to_csv('worst_performing_services.csv', index=False)