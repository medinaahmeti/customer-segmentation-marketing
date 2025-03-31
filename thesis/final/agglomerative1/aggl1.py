import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('finalDBRaw.csv')

# Step 1: Data Preprocessing
df['OrderCount'] = 1  # Each row represents an order, so we can just count rows per customer
df.dropna(subset=['CustomerID'], inplace=True)

# Step 2: Group by CustomerID to get aggregated metrics
customer_data = df.groupby('CustomerID').agg({
    'OrderCount': 'count',  # Total number of orders per customer
}).reset_index()

# Step 3: Log Transformation and Normalize the data
customer_data['LogOrderCount'] = np.log1p(customer_data['OrderCount'])
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['LogOrderCount']])

# Step 4: Dendrogram to Determine Optimal Clusters
plt.figure(figsize=(10, 7))
Z = linkage(customer_data_scaled, method='ward')
dendrogram(Z,
           p=30,  # Only show the first 30 customer labels
#            truncate_mode='lastp',  # Only display last p merged clusters
           show_leaf_counts=False,  # Hide the number of leaves for each node
           show_contracted=True,
           color_threshold=0.5)  # Added to separate vertical lines clearly
plt.xticks([])  # Disable x-axis ticks

plt.title('Dendrogram for Hierarchical Clustering', fontsize=12)
plt.xlabel('Customers', fontsize=10)
plt.ylabel('Distances', fontsize=10)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300)  # Save Dendrogram Plot with higher DPI
plt.show()

# Step 5: Run Agglomerative Clustering for Different Cluster Counts
results = {}
for k in [2, 3, 4]:  # Experiment with different cluster counts
    hierarchical_clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    customer_data[f'Cluster_{k}'] = hierarchical_clustering.fit_predict(customer_data_scaled)
    results[k] = customer_data.copy()

# Step 6: Create Separate Cluster Scatter Plots for Each k
# Cluster renaming mapping
cluster_renames = {
    0: 'ALEC',  # Cluster 0 → ALEC
    1: 'AAEC',  # Cluster 1 → AAEC
    2: 'AHEC',  # Cluster 2 → AHEC
    3: 'AIEC',  # Cluster 3 → AIEC
}

for k in [2, 3, 4]:  # Loop through k values
    fig, ax = plt.subplots(figsize=(10, 6))  # Create individual figure for each k
    # Define marker shapes for each cluster
    marker_shapes = ['o', 'v', '*', '+']

    # Plot each cluster with a different shape
    for cluster_id in range(k):  # Loop through clusters
        cluster_data = results[k][results[k][f'Cluster_{k}'] == cluster_id]
        ax.scatter(
            cluster_data['CustomerID'],
            cluster_data['LogOrderCount'],
            label=f'{cluster_renames.get(cluster_id, f"Cluster {cluster_id}")}',  # Use renamed cluster
            s=50,  # Size of markers
            alpha=0.9,
            marker=marker_shapes[cluster_id]  # Assign a different shape for each cluster
        )

    # Set plot titles and labels
    ax.set_title(f'Clusters by Order Count (k={k})', fontsize=12)
    ax.set_xlabel('Customer ID', fontsize=10)
    ax.set_ylabel('LogOrderCount', fontsize=10)
    ax.legend(fontsize=8)  # Smaller font size for the legend

    # Save the plot for the current k
    plt.tight_layout()
    plt.savefig(f'hierarchical_cluster_k{k}.png', dpi=300)  # Save each plot separately
    plt.show()  # Display the plot

# Step 7: Save the data for the final k (e.g., k=3) to a CSV file
optimal_k = 3
output_file = 'hierarchical_clustering_output_k3.csv'
results[optimal_k][['CustomerID', 'OrderCount', 'LogOrderCount', f'Cluster_{optimal_k}']].to_csv(output_file, index=False)
print(f"Customer segmentation data for k={optimal_k} saved to {output_file}")

# Step 8: Optimal Cluster Plot for k=3 with Different Marker Shapes
kmeans_3 = results[3]
fig, ax = plt.subplots(figsize=(10, 6))

# Define marker shapes for each cluster
marker_shapes = ['o', 'x', '*']

# Plot each cluster with a different shape
for cluster_id in range(3):  # Loop through clusters
    cluster_data = kmeans_3[kmeans_3[f'Cluster_{optimal_k}'] == cluster_id]
    ax.scatter(
        cluster_data['CustomerID'],
        cluster_data['LogOrderCount'],
        label=f'{cluster_renames.get(cluster_id, f"Cluster {cluster_id}")}',
        s=50,  # Size of markers
        alpha=0.9,
        marker=marker_shapes[cluster_id]  # Assign a different shape for each cluster
    )

ax.set_title('Clusters by Order Count (k=3)', fontsize=12)
ax.set_xlabel('Customer ID', fontsize=10)
ax.set_ylabel('LogOrderCount', fontsize=10)
ax.legend(fontsize=8)  # Smaller font size for the legend

# Save the static cluster plot for k=3 as an image
plt.tight_layout()
plt.savefig('optimal_hierarchical_cluster_k3_plot.png', dpi=300)  # Save Static Plot for k=3
plt.show()  # Display the plot
