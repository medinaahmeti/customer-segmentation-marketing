import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Step 2: Log transformation of 'TotalOrderValue'
df_agg = df.groupby('Country').agg({
    'TotalOrderValue': 'sum'
}).reset_index()

# Map country names to ISO codes
df_agg['Country'] = df_agg['Country'].map(country_to_iso)

df_agg['LogOrderValue'] = np.log1p(df_agg['TotalOrderValue'])  # Log transformation

# Normalize data
scaler = StandardScaler()
df_agg_scaled = scaler.fit_transform(df_agg[['LogOrderValue']])

# Step 3: Elbow Method for Optimal K
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_agg_scaled)
    wcss.append(kmeans.inertia_)

# Save and plot Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method_country.png', dpi=300)  # Save Elbow Plot with higher DPI
plt.show()

# Step 4: KMeans clustering with different values of k
results = {}
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_agg[f'Cluster_{k}'] = kmeans.fit_predict(df_agg_scaled)
    results[k] = df_agg.copy()

# Cluster renaming mapping, including `KOHRC` for cluster 4
cluster_renames = {
    0: 'KHRC',
    1: 'KLRC',
    2: 'KARC',
    3: 'KIRC'  # Applicable for k=4 clusters I stands for Immoderate
}

# Step 5: Save individual cluster scatter plots for k=2, k=3, and k=4
shapes = ['o', 'v', '*', '+']  # Shapes for clusters
for k in [2, 3, 4]:
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size for individual plots
    data = results[k]
    for cluster_id in range(k):
        cluster_data = data[data[f'Cluster_{k}'] == cluster_id]
        ax.scatter(
            cluster_data['Country'],  # X-axis: ISO-coded Country
            cluster_data['LogOrderValue'],  # Y-axis: Log Order Value
            label=f'{cluster_renames.get(cluster_id, f"Cluster {cluster_id}")}',
            alpha=0.8,
            marker=shapes[cluster_id % len(shapes)]
        )
    ax.set_title(f'Clusters by Country Revenue (k={k})', fontsize=14)
    ax.set_xlabel('Country (ISO Code)', fontsize=12)
    ax.set_ylabel('Log Order Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

    # Save each plot as a separate image
    plt.tight_layout()
    plt.savefig(f'clusters_by_country_k{k}.png', dpi=300)
    plt.show()

# Step 6: Save the data for the final k (e.g., k=3 or k=4)
optimal_k = 3  # Choose k based on elbow method or experiments
df_agg['Cluster'] = results[optimal_k][f'Cluster_{optimal_k}'].map(cluster_renames)  # Apply renaming
df_agg.to_csv('country_clusters_k3.csv', index=False)
print(f"Clustered data saved to 'country_clusters_k3.csv'.")

# Step 7: Plot clusters for the optimal k=3 separately
plt.figure(figsize=(8, 6))  # Reduced size
for cluster_id in range(optimal_k):
    cluster_data = df_agg[df_agg['Cluster'] == cluster_renames[cluster_id]]
    plt.scatter(
        cluster_data['Country'],  # X-axis: ISO-coded Country
        cluster_data['LogOrderValue'],  # Y-axis: Log Order Value
        label=f'{cluster_renames[cluster_id]}',
        alpha=0.8,
        marker=shapes[cluster_id % len(shapes)]
    )

plt.title('Clusters by Country Revenue (k=3)', fontsize=14)
plt.xlabel('Country (ISO Code)', fontsize=12)
plt.ylabel('Log Order Value', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate country labels for readability
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('clusters_by_country_k3.png', dpi=300)  # Save the k=3 cluster plot
plt.show()

print("Cluster visualizations with ISO-coded countries saved and displayed successfully.")
