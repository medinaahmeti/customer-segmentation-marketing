import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

# Step 4: Elbow Method to Find Optimal Number of Clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(customer_data_scaled)
    wcss.append(kmeans.inertia_)

# Save the Elbow Plot with reduced size and higher DPI
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal K', fontsize=14)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.grid()
plt.savefig('elbow_plot.png', dpi=300)  # Higher DPI for clarity
plt.show()

# Step 5: Run KMeans for k=2, 3, and 4
results = {}
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    customer_data[f'Cluster_{k}'] = kmeans.fit_predict(customer_data_scaled)
    results[k] = customer_data.copy()

# Define cluster renaming for consistent naming
cluster_renames = {
    0: 'KLEC',
    1: 'KAEC',
    2: 'KHEC',
    3: 'KIEC'  # Applicable for k=4 clusters I stands for Immoderate
}

# Step 6: Save individual cluster scatter plots for k=2, 3, and 4
shapes = ['o', 'v', '*', '+']  # Shapes for clusters
for k in [2, 3, 4]:
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size for individual plots
    data = results[k]
    for cluster_id in range(k):
        cluster_data = data[data[f'Cluster_{k}'] == cluster_id]
        ax.scatter(
            cluster_data['CustomerID'],
            cluster_data['LogOrderCount'],
            label=f'{cluster_renames.get(cluster_id, f"Cluster {cluster_id}")}',
            s=30,
            alpha=0.9,
            marker=shapes[cluster_id % len(shapes)]
        )
    ax.set_title(f'Clusters by Order Count (k={k})', fontsize=14)
    ax.set_xlabel('Customer ID', fontsize=12)
    ax.set_ylabel('LogOrderCount', fontsize=12)
    ax.legend(fontsize=10)

    # Save individual plots with descriptive filenames
    plt.tight_layout()
    plt.savefig(f'cluster_plot_k{k}.png', dpi=300)
    plt.show()

# Step 7: Save the data for the final k (e.g., k=3) to a CSV file
optimal_k = 3
output_file = 'kmeans1.csv'
results[optimal_k][['CustomerID', 'OrderCount', 'LogOrderCount', f'Cluster_{optimal_k}']].to_csv(output_file, index=False)
print(f"Customer segmentation data for k={optimal_k} saved to {output_file}")

# Step 8: Optimal Cluster Plot for k=3 with Renamed Clusters and Higher Resolution
kmeans_3 = results[3]
fig, ax = plt.subplots(figsize=(8, 6))  # Reduced size

# Define marker shapes and new cluster names
marker_shapes = ['o', 'x', '*']

# Plot each cluster with a different shape and renamed labels
for cluster_id in range(3):
    cluster_data = kmeans_3[kmeans_3[f'Cluster_{optimal_k}'] == cluster_id]
    ax.scatter(
        cluster_data['CustomerID'],
        cluster_data['LogOrderCount'],
        label=f'{cluster_renames.get(cluster_id)}',
        s=50,  # Size of markers
        alpha=0.9,
        marker=marker_shapes[cluster_id]  # Assign a different shape for each cluster
    )

ax.set_title('Clusters by Order Count (k=3)', fontsize=14)
ax.set_xlabel('Customer ID', fontsize=12)
ax.set_ylabel('LogOrderCount', fontsize=12)
ax.legend(fontsize=10)

# Save the static cluster plot with reduced size and higher resolution
plt.tight_layout()
plt.savefig('optimal_cluster_k3_plot.png', dpi=300)  # Higher DPI for clarity
plt.show()  # Display the plot
