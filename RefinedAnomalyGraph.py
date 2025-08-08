import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated training data (replace with actual data loading)
np.random.seed(42)
n_services = 600
n_records_train = 10000
services = [f"svc_{i}" for i in range(n_services)]
train_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2025-08-01', periods=n_records_train, freq='min'),
    'service_name': np.random.choice(services, size=n_records_train),
    'TPS': np.random.lognormal(mean=2, sigma=1, size=n_records_train).clip(1, 100),
    'RT': np.random.lognormal(mean=5, sigma=1, size=n_records_train).clip(50, 5000)
})
# Add anomalies to training data
anomaly_services = ['svc_10', 'svc_50', 'svc_100', 'svc_200', 'svc_300']
anomaly_indices = train_data['service_name'].isin(anomaly_services)
train_data.loc[anomaly_indices, 'RT'] *= 2
train_data.loc[anomaly_indices, 'TPS'] *= 0.5

# Simulated prediction data (replace with actual data loading)
n_records_predict = 5000
new_services = services + [f"svc_new_{i}" for i in range(50)]
predict_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2025-08-02', periods=n_records_predict, freq='min'),
    'service_name': np.random.choice(new_services, size=n_records_predict),
    'TPS': np.random.lognormal(mean=2, sigma=1, size=n_records_predict).clip(1, 100),
    'RT': np.random.lognormal(mean=5, sigma=1, size=n_records_predict).clip(50, 5000)
})
# Add anomalies to prediction data
anomaly_indices_predict = predict_data['service_name'].isin(['svc_100', 'svc_new_1', 'svc_new_2'])
predict_data.loc[anomaly_indices_predict, 'RT'] *= 2
predict_data.loc[anomaly_indices_predict, 'TPS'] *= 0.5

# Function to compute per-service normalization
def normalize_per_service(group, service_stats=None):
    if service_stats is None:
        # Training: Compute mean and std for normalization
        group['RT_zscore'] = (group['RT'] - group['RT'].mean()) / group['RT'].std()
        group['TPS_zscore'] = (group['TPS'] - group['TPS'].mean()) / group['TPS'].std()
        service_stats = {
            'RT_mean': group['RT'].mean(),
            'RT_std': group['RT'].std(),
            'TPS_mean': group['TPS'].mean(),
            'TPS_std': group['TPS'].std()
        }
    else:
        # Prediction: Use training dataset's mean and std for unseen services
        group['RT_zscore'] = (group['RT'] - service_stats['RT_mean']) / service_stats['RT_std']
        group['TPS_zscore'] = (group['TPS'] - service_stats['TPS_mean']) / service_stats['TPS_std']
    group['RT_log'] = np.log1p(group['RT'])
    group['TPS_log'] = np.log1p(group['TPS'])
    group['response_time_per_transaction'] = group['RT'] / group['TPS']
    group['response_time_per_transaction_log'] = np.log1p(group['response_time_per_transaction'])
    return group, service_stats

# Training phase
train_data_stats = {}
train_data_groups = []
for service, group in train_data.groupby('service_name'):
    normalized_group, stats = normalize_per_service(group)
    train_data_groups.append(normalized_group)
    train_data_stats[service] = stats
train_data = pd.concat(train_data_groups).reset_index(drop=True)

# Extract temporal features
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek

# Aggregate per service
train_agg_data = train_data.groupby('service_name').agg({
    'RT_zscore': ['mean', 'max'],
    'TPS_zscore': ['mean', 'max'],
    'RT_log': ['mean', 'median', 'max'],
    'TPS_log': ['mean', 'median', 'max'],
    'response_time_per_transaction_log': ['mean', 'median', 'max'],
    'hour': 'mean',
    'day_of_week': 'mean',
    'RT': ['mean', 'count'],
    'TPS': 'mean'
}).reset_index()

# Flatten column names
train_agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in train_agg_data.columns]

# Filter services with low data volume
train_agg_data = train_agg_data[train_agg_data['RT_count'] >= 10]

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
X_train = scaler.fit_transform(train_agg_data[features])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

# Save model and scaler
dump(iso_forest, 'isolation_forest.joblib')
dump(scaler, 'scaler.joblib')
dump(train_data_stats, 'train_data_stats.joblib')

# Prediction phase
iso_forest = load('isolation_forest.joblib')
scaler = load('scaler.joblib')
train_data_stats = load('train_data_stats.joblib')

# Compute per-service normalization for prediction data
predict_data_groups = []
default_stats = {
    'RT_mean': train_data['RT'].mean(),
    'RT_std': train_data['RT'].std(),
    'TPS_mean': train_data['TPS'].mean(),
    'TPS_std': train_data['TPS'].std()
}
for service, group in predict_data.groupby('service_name'):
    stats = train_data_stats.get(service, default_stats)
    normalized_group, _ = normalize_per_service(group, stats)
    predict_data_groups.append(normalized_group)
predict_data = pd.concat(predict_data_groups).reset_index(drop=True)

# Extract temporal features
predict_data['hour'] = predict_data['timestamp'].dt.hour
predict_data['day_of_week'] = predict_data['timestamp'].dt.dayofweek

# Aggregate per service
predict_agg_data = predict_data.groupby('service_name').agg({
    'RT_zscore': ['mean', 'max'],
    'TPS_zscore': ['mean', 'max'],
    'RT_log': ['mean', 'median', 'max'],
    'TPS_log': ['mean', 'median', 'max'],
    'response_time_per_transaction_log': ['mean', 'median', 'max'],
    'hour': 'mean',
    'day_of_week': 'mean',
    'RT': ['mean', 'count'],
    'TPS': 'mean'
}).reset_index()

# Flatten column names
predict_agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in predict_agg_data.columns]

# Filter services with low data volume
predict_agg_data = predict_agg_data[predict_agg_data['RT_count'] >= 10]

# Transform prediction data using the trained scaler
X_predict = scaler.transform(predict_agg_data[features])

# Predict anomaly scores (negated for higher = worse)
predict_agg_data['anomaly_score'] = -iso_forest.score_samples(X_predict)

# Get top 5 worst-performing services
top_5_worst = predict_agg_data[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].sort_values(by='anomaly_score', ascending=False).head(5)

# Output results
print("Top 5 worst-performing microservices in prediction dataset:")
print(top_5_worst[['service_name', 'anomaly_score', 'RT_mean', 'TPS_mean']].round(2))

# Save results to CSV
top_5_worst.to_csv('worst_performing_services_predict.csv', index=False)

# Visualization 1: Scatter Plot of RT_mean vs TPS_mean
plt.figure(figsize=(10, 6))
scatter = plt.scatter(predict_agg_data['RT_mean'], predict_agg_data['TPS_mean'],
                     c=predict_agg_data['anomaly_score'], cmap='RdBu_r',
                     alpha=0.6, s=100)
plt.colorbar(scatter, label='Anomaly Score (higher = worse)')
plt.xlabel('Mean Response Time (ms)')
plt.ylabel('Mean TPS')
plt.title('Microservices Performance: RT vs TPS')
# Annotate top 5 worst services
for idx, row in top_5_worst.iterrows():
    plt.annotate(row['service_name'], (row['RT_mean'], row['TPS_mean']),
                 textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
plt.grid(True)
plt.savefig('rt_vs_tps_scatter.png')
plt.close()

# Visualization 2: Bar Plot of Top 5 Worst Services
plt.figure(figsize=(8, 5))
sns.barplot(x='service_name', y='anomaly_score', data=top_5_worst)
plt.xlabel('Service Name')
plt.ylabel('Anomaly Score (higher = worse)')
plt.title('Top 5 Worst-Performing Microservices')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_5_worst_bar.png')
plt.close()