import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'player_statistics_cleaned_solo_kill.csv'
data = pd.read_csv(file_path)

# Group by 'TeamName' and perform key analyses
# Example metrics: Average KDA, Win rate, GD@15, and CSPerMin
grouped_data = data.groupby('TeamName').agg({
    'KDA': 'mean',
    'Win rate': 'mean',
    'GD@15': 'mean',
    'CSPerMin': 'mean',
    'Solo Kills': 'sum',
    'Games': 'sum'
}).reset_index()

# Sort by Win rate for easier comparison
grouped_data = grouped_data.sort_values(by='Win rate', ascending=False)

# Display the grouped and analyzed data
print("Grouped Analysis by TeamName:\n", grouped_data)

# Save the grouped data to a new CSV file (optional)
grouped_file_path = 'player_statistics_grouped_by_team.csv'
grouped_data.to_csv(grouped_file_path, index=False)

print(f"Grouped analysis data saved to: {grouped_file_path}")

# Visualization: Bar chart for Win rate by TeamName
plt.figure(figsize=(12, 8))
plt.bar(grouped_data['TeamName'], grouped_data['Win rate'], color='skyblue')
plt.title('Win Rate by Team', fontsize=16)
plt.xlabel('TeamName', fontsize=12)
plt.ylabel('Win Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualization: Scatter plot for GD@15 vs. KDA by TeamName
plt.figure(figsize=(10, 6))
plt.scatter(grouped_data['GD@15'], grouped_data['KDA'], c='orange', edgecolor='k', s=100)
for i, txt in enumerate(grouped_data['TeamName']):
    plt.annotate(txt, (grouped_data['GD@15'].iloc[i] + 0.5, grouped_data['KDA'].iloc[i] + 0.1), fontsize=9)  # Adjust label positions
plt.title('GD@15 vs. KDA by Team', fontsize=16)
plt.xlabel('Gold Difference @15 (GD@15)', fontsize=12)
plt.ylabel('Average KDA', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Analysis: Clustering teams based on performance metrics
features = grouped_data[['KDA', 'Win rate', 'GD@15', 'CSPerMin']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
grouped_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization: Clustering results with pairplot
sns.pairplot(grouped_data, hue='Cluster', vars=['KDA', 'Win rate', 'GD@15', 'CSPerMin'], palette='Set2', markers=['o', 's', 'D'])
plt.suptitle('Cluster Analysis of Teams', y=1.02, fontsize=16)
plt.show()

# Visualization: Heatmap of performance metrics by TeamName
plt.figure(figsize=(12, 8))
sns.heatmap(grouped_data.set_index('TeamName')[['KDA', 'Win rate', 'GD@15', 'CSPerMin']], annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Performance Metrics by Team', fontsize=16)
plt.show()
