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

# Add anomalies
anomaly_services = ['svc_10', 'svc_50', 'svc_100', 'svc_200', 'svc_300']
anomaly_indices = data['service_name'].isin(anomaly_services)
data.loc[anomaly_indices, 'RT'] *= 2
data.loc[anomaly_indices, 'TPS'] *= 0.5

# Feature engineering: Per-service normalization
def normalize_per_service(group):
    group['RT_zscore'] = (group['RT'] - group['RT'].mean()) / group['RT'].std()
    group['TPS_zscore'] = (group['TPS'] - group['TPS'].mean()) / group['TPS'].std()
    group['RT_log'] = np.log1p(group['RT'])
    group['TPS_log'] = np.log1p(group['TPS'])
    group['response_time_per_transaction'] = group['RT'] / group['TPS']
    group['response_time_per_transaction_log'] = np.log1p(group['response_time_per_transaction'])
    return group

data = data.groupby('service_name').apply(normalize_per_service).reset_index(drop=True)

# Extract temporal features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Aggregate per service
agg_data = data.groupby('service_name').agg({
    'RT_zscore': ['mean', 'max'],
    'TPS_zscore': ['mean', 'max'],
    'RT_log': ['mean', 'median', 'max'],
    'TPS_log': ['mean', 'median', 'max'],
    'response_time_per_transaction_log': ['mean', 'median', 'max'],
    'hour': 'mean',
    'day_of_week': 'mean',
    'RT': ['mean', 'count']  # Keep original RT for output
}).reset_index()

# Flatten column names
agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_data.columns]

# Filter services with low data volume
agg_data = agg_data[agg_data['RT_count'] >= 10]

# Features for model
features = [
    'RT_zscore_mean', 'RT_zscore_max',
    'TPS_zscore_mean', 'TPS_zscore_max',
    'RT_log_mean', 'RT_log_median', 'RT_log_max',
    'TPS_log_mean', 'TPS_log_median', 'TPS_log_max',
    'response_time_per_transaction_log_mean',
    'response_time_per_transaction_log_median',
    'response_time_per_transaction_log_max',
    'hour_mean', 'day_of_week_mean'
]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(agg_data[features])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
anomaly_scores = iso_forest.fit_predict(X)
agg_data['anomaly_score'] = -iso_forest.score_samples(X)

# Get top 5 worst-performing services
top_5_worst = agg_data[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].sort_values(by='anomaly_score', ascending=False).head(5)

# Output results
print("Top 5 worst-performing microservices:")
print(top_5_worst[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].round(2))

# Save results to CSV
top_5_worst.to_csv('worst_performing_services_refined.csv', index=False)