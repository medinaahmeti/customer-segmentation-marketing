import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import itertools

# Load data
data_path = '../db/finalDBRaw.csv'
df = pd.read_csv(data_path)

# Ensure InvoiceDate is in datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='raise')
df.dropna(subset=['CustomerID'], inplace=True)

# Filter and clean data
df = df[df['UnitPrice'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM Calculation per customer
rfm_df = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (df['InvoiceDate'].max() - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

# Define lifecycle stages based on Recency and Frequency thresholds
def lifecycle_stage(row):
    if row['Recency'] <= 30 and row['Frequency'] >= 10:
        return 'Loyal'
    elif row['Recency'] <= 30 and row['Frequency'] < 10:
        return 'New'
    elif row['Recency'] > 30 and row['Frequency'] >= 10:
        return 'At Risk'
    else:
        return 'Churn Risk'

rfm_df['LifecycleStage'] = rfm_df.apply(lifecycle_stage, axis=1)

# Filter the dataframe to only include customers in the specified lifecycle stages
filtered_rfm_df = rfm_df[rfm_df['LifecycleStage'].isin(['Loyal', 'New', 'At Risk', 'Churn Risk'])]

# Log-transform and scale RFM metrics
filtered_rfm_df[['Recency', 'Frequency', 'Monetary']] = np.log1p(filtered_rfm_df[['Recency', 'Frequency', 'Monetary']])
scaler = MinMaxScaler()
scaled_rfm = scaler.fit_transform(filtered_rfm_df[['Recency', 'Frequency', 'Monetary']])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
filtered_rfm_df['KMeans_Cluster'] = kmeans.fit_predict(scaled_rfm)

# Rename clusters
cluster_names = {0: 'KRCB', 1: 'KHVDC', 2: 'KFHVC'}
filtered_rfm_df['Clusters'] = filtered_rfm_df['KMeans_Cluster'].map(cluster_names)

# Add a column to represent the combination of Cluster and Lifecycle Stage
filtered_rfm_df['Cluster_Lifecycle'] = filtered_rfm_df['Clusters'] + '_' + filtered_rfm_df['LifecycleStage']

# Define unique markers for combinations
unique_combinations = filtered_rfm_df['Cluster_Lifecycle'].unique()
markers = ['o', 's', 'D', '^', 'P', '*', 'X', '<', '>']  # 9 unique marker styles
marker_map = {combo: markers[i % len(markers)] for i, combo in enumerate(unique_combinations)}

# Define a color palette
palette = sns.color_palette("Set1", len(unique_combinations))
color_map = {combo: palette[i] for i, combo in enumerate(unique_combinations)}

# Scatter plot with enhanced clarity
plt.figure(figsize=(14, 10))  # Larger figure size for better readability
for combo, marker in marker_map.items():
    subset = filtered_rfm_df[filtered_rfm_df['Cluster_Lifecycle'] == combo]
    plt.scatter(
        subset['Recency'],
        subset['Monetary'],
        label=combo,
        marker=marker,
        s=300,  # Larger marker size for better visibility
        color=color_map[combo],
        edgecolors='black',  # Add black edge for marker clarity
        linewidths=0.5       # Thin edge for elegance
    )

# Customize plot
plt.title("KMeans Clustering with Lifecycle Stage", fontsize=20, fontweight='medium')
plt.xlabel("Log-Scaled Recency", fontsize=18, fontweight='medium')  # Larger x-axis label
plt.ylabel("Log-Scaled Monetary", fontsize=18, fontweight='medium')  # Larger y-axis label
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend inside the plot
plt.legend(
    title='Cluster_Lifecycle',
    loc='upper right',  # Adjust location
    fontsize=16,  # Larger legend font size
    title_fontsize=18,  # Larger legend title font size
    frameon=True
)
plt.tight_layout()

# Save with higher DPI for clarity
plt.savefig("images/scatterplot_kmeans_lifecycle.png", dpi=600, bbox_inches='tight')  # Increased DPI and tight bounding box
plt.show()

# Pairplot with enhanced clarity and larger clusters
pairplot = sns.pairplot(
    filtered_rfm_df,
    vars=['Recency', 'Frequency', 'Monetary'],
    hue='Clusters',
    palette='Set1',
    markers=list(itertools.islice(itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>']), len(filtered_rfm_df['Clusters'].unique()))),
    height=3.5  # Larger pairplot marker size
)

# Customize legend and axes
pairplot.fig.suptitle("KMeans Clusters RFM", y=1.02, fontsize=20, fontweight='medium')  # Larger plot title
pairplot.fig.set_size_inches(14, 10)
legend = pairplot._legend
legend.set_title("Clusters")
legend.set_bbox_to_anchor((1, 0.5))  # Adjust legend position
legend.set_frame_on(True)  # Add a frame
for text in legend.get_texts():
    text.set_fontsize(16)  # Larger font size for labels
legend.get_title().set_fontsize(18)  # Larger legend title font size

# Make axis labels bigger for the pairplot
for ax in pairplot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)  # Update x-axis label size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)  # Update y-axis label size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Update tick label size

plt.tight_layout()
pairplot.fig.savefig("images/pairplot_kmeans_clusters.png", dpi=600, bbox_inches='tight')
plt.show()
distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_rfm)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()