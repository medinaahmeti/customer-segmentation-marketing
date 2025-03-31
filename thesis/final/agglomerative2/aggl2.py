import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Mapping of countries to their ISO codes
country_to_iso = {
    "United Kingdom": "UK", "France": "FR", "Australia": "AU", "Netherlands": "NL",
    "Germany": "DE", "Norway": "NO", "EIRE": "IE", "Switzerland": "CH",
    "Spain": "ES", "Poland": "PL", "Portugal": "PT", "Italy": "IT",
    "Belgium": "BE", "Lithuania": "LT", "Japan": "JP", "Iceland": "IS",
    "Channel Islands": "CI", "Denmark": "DK", "Cyprus": "CY", "Sweden": "SE",
    "Finland": "FI", "Austria": "AT", "Greece": "GR", "Singapore": "SG",
    "Lebanon": "LB", "United Arab Emirates": "AE", "Israel": "IL",
    "Saudi Arabia": "SA", "Czech Republic": "CZ", "Canada": "CA",
    "Unspecified": "UN", "Brazil": "BR", "USA": "US",
    "European Community": "EU", "Bahrain": "BH", "Malta": "MT", "RSA": "ZA"
}

# Step 1: Load and clean the data
df = pd.read_csv('finalDBRaw.csv')

# Drop any missing values if necessary
df = df.dropna(subset=['Quantity', 'UnitPrice'])

# Calculate TotalOrderValue (Revenue)
df['TotalOrderValue'] = df['Quantity'] * df['UnitPrice']

# Step 2: Aggregate and log-transform 'TotalOrderValue'
df_agg = df.groupby('Country').agg({
    'TotalOrderValue': 'sum'
}).reset_index()

# Map country names to ISO codes
df_agg['Country'] = df_agg['Country'].map(country_to_iso)

df_agg['LogOrderValue'] = np.log1p(df_agg['TotalOrderValue'])  # Log transformation

# Normalize data
scaler = StandardScaler()
df_agg_scaled = scaler.fit_transform(df_agg[['LogOrderValue']])

# Step 3: Plot dendrogram to find optimal clusters
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Country Clustering")
linked = linkage(df_agg_scaled, method='ward')
dendrogram(linked,
           orientation='top',
           labels=df_agg['Country'].values,
           distance_sort='descending',
           show_leaf_counts=True)
plt.savefig('dendrogram_country.png')  # Save dendrogram plot
plt.show()

# Step 4: Agglomerative Clustering for different values of k
results = {}
for k in [2, 3, 4]:
    agg_clustering = AgglomerativeClustering(n_clusters=k)
    df_agg[f'Cluster_{k}'] = agg_clustering.fit_predict(df_agg_scaled)
    results[k] = df_agg.copy()

# Step 5: Cluster renaming for each value of k
cluster_renaming = {
    0: 'AHRC',
    1: 'ALRC',
    2: 'AARC',
    3: 'AIRC'
}

# Step 6: Plot separate cluster plots for each k
shapes = ['o', 'v', '*', '+']  # Marker shapes for clusters

for k in [2, 3, 4]:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = results[k]

    for cluster_id in range(k):
        cluster_data = data[data[f'Cluster_{k}'] == cluster_id]
        ax.scatter(
            cluster_data['Country'],  # X-axis: ISO-coded Country
            cluster_data['LogOrderValue'],  # Y-axis: Log Order Value
            label=f'{cluster_renaming.get(cluster_id, f"Cluster {cluster_id}")}',
            alpha=0.8,
            marker=shapes[cluster_id % len(shapes)]
        )

    ax.set_title(f'Clusters by Country Revenue (k={k})', fontsize=12)
    ax.set_xlabel('Country (ISO Code)', fontsize=10)
    ax.set_ylabel('Log Order Value', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

    plt.tight_layout()
    plt.savefig(f'agglomerative_clusters_by_country_k{k}.png')  # Save the k clusters plot
    plt.show()

# Step 7: Save the data for the final k (e.g., k=3)
optimal_k = 3  # Choose k based on dendrogram or experiments
df_agg['Cluster'] = results[optimal_k][f'Cluster_{optimal_k}']
df_agg.to_csv('country_clusters_agglomerative_k3.csv', index=False)
print(f"Clustered data saved to 'country_clusters_agglomerative_k3.csv'.")

# Step 8: Plot the optimal k=3 cluster plot separately
plt.figure(figsize=(10, 6))
for cluster_id in range(optimal_k):
    cluster_data = df_agg[df_agg['Cluster'] == cluster_id]
    plt.scatter(
        cluster_data['Country'],  # X-axis: ISO-coded Country
        cluster_data['LogOrderValue'],  # Y-axis: Log Order Value
        label=f'{cluster_renaming.get(cluster_id, f"Cluster {cluster_id}")}',
        alpha=0.8,
        marker=shapes[cluster_id % len(shapes)]
    )

plt.title('Clusters by Country Revenue (k=3)', fontsize=12)
plt.xlabel('Country (ISO Code)', fontsize=10)
plt.ylabel('Log Order Value', fontsize=10)
plt.xticks(rotation=45, ha='right')  # Rotate country labels for readability
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('agglomerative_clusters_by_country_k3.png')  # Save the k=3 cluster plot
plt.show()

print("Cluster visualizations with ISO-coded countries saved and displayed successfully.")
