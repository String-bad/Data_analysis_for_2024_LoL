import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'player_statistics_cleaned_solo_kill.csv'
data = pd.read_csv(file_path)

# Filter data for ADC players
adc_data = data[data['Position'] == 'Adc'][['PlayerName', 'KP%', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSPerMin', 'GoldPerMin']]

# Display basic stats for ADC players
print("ADC Players Analysis:\n", adc_data.describe())

# Prepare data for clustering ADC players
adc_features = adc_data[['KP%', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSPerMin', 'GoldPerMin']]
scaler = StandardScaler()
adc_scaled_features = scaler.fit_transform(adc_features)

# Perform KMeans clustering for ADC players
adc_kmeans = KMeans(n_clusters=3, random_state=42)
adc_data['Cluster'] = adc_kmeans.fit_predict(adc_scaled_features)

# Visualization: Clustering results for ADC players
sns.pairplot(adc_data, hue='Cluster', vars=['KP%', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSPerMin', 'GoldPerMin'], palette='Set1', markers=['o', 's', 'D'])
plt.suptitle('Cluster Analysis of ADC Players', y=1.02, fontsize=16)
plt.show()

# Visualization: Heatmap of Cluster Centers for ADC players
adc_cluster_centers = pd.DataFrame(adc_kmeans.cluster_centers_, columns=adc_features.columns)
plt.figure(figsize=(10, 6))
sns.heatmap(adc_cluster_centers, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, xticklabels=adc_features.columns, yticklabels=[f'Cluster {i}' for i in range(3)])
plt.title('Cluster Centers for ADC Players', fontsize=16)
plt.show()

# Additional Analysis: Compare KP% and CSPerMin for ADC players
plt.figure(figsize=(10, 6))
plt.scatter(adc_data['KP%'], adc_data['CSPerMin'], c=adc_data['Cluster'], cmap='viridis', edgecolor='k', s=100)
for i, txt in enumerate(adc_data['PlayerName']):
    plt.annotate(txt, (adc_data['KP%'].iloc[i] + 0.01, adc_data['CSPerMin'].iloc[i] + 0.01), fontsize=9)
plt.title('KP% vs CSPerMin for ADC Players', fontsize=16)
plt.xlabel('KP% (Kill Participation)', fontsize=12)
plt.ylabel('CSPerMin (Creep Score Per Minute)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save ADC clustered data (optional)
adc_clustered_file_path = 'adc_players_clustered.csv'
adc_data.to_csv(adc_clustered_file_path, index=False)
print(f"Clustered ADC players data saved to: {adc_clustered_file_path}")

# Group ADC players by clusters and list player names for each cluster
adc_clusters = adc_data.groupby('Cluster')['PlayerName'].apply(list)

# Create a bar plot to visualize the number of players in each cluster
cluster_sizes = adc_data['Cluster'].value_counts()

# Visualization: Bar chart for the number of ADC players in each cluster
plt.figure(figsize=(10, 6))
plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color='skyblue', edgecolor='black')
plt.title('Number of ADC Players in Each Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Players', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print out the players in each cluster
for cluster, players in adc_clusters.items():
    print(f"Cluster {cluster}:")
    for player in players:
        print(f" - {player}")
    print()
