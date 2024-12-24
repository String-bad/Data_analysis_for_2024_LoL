import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'player_statistics_cleaned_solo_kill.csv'
data = pd.read_csv(file_path)

# Filter data for support players
support_data = data[data['Position'] == 'Support'][['PlayerName', 'KP%', 'VSPM', 'Avg WPM', 'Avg WCPM', 'Avg VWPM']]

# Display basic stats for support players
print("Support Players Analysis:\n", support_data.describe())

# Prepare data for clustering
features = support_data[['KP%', 'VSPM', 'Avg WPM', 'Avg WCPM', 'Avg VWPM']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
support_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization: Clustering results with pairplot
sns.pairplot(support_data, hue='Cluster', vars=['KP%', 'VSPM', 'Avg WPM', 'Avg WCPM', 'Avg VWPM'], palette='Set2', markers=['o', 's', 'D'])
plt.suptitle('Cluster Analysis of Support Players', y=1.02, fontsize=16)
plt.show()

# Visualization: Heatmap of Cluster Centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, xticklabels=features.columns, yticklabels=[f'Cluster {i}' for i in range(3)])
plt.title('Cluster Centers for Support Players', fontsize=16)
plt.show()

# Save clustered data (optional)
clustered_file_path = 'support_players_clustered.csv'
support_data.to_csv(clustered_file_path, index=False)
print(f"Clustered support players data saved to: {clustered_file_path}")
