import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage='ward')
filtered_rfm_df['Agglo_Cluster'] = agglo.fit_predict(scaled_rfm)

# Rename clusters
cluster_names = {0: 'AALS', 1: 'AIAS', 2: 'ALHS'}
filtered_rfm_df['Clusters'] = filtered_rfm_df['Agglo_Cluster'].map(cluster_names)

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
plt.figure(figsize=(14, 10))
for combo, marker in marker_map.items():
    subset = filtered_rfm_df[filtered_rfm_df['Cluster_Lifecycle'] == combo]
    plt.scatter(
        subset['Recency'],
        subset['Monetary'],
        label=combo,
        marker=marker,
        s=300,  # Larger cluster size
        color=color_map[combo],
        edgecolors='black',  # Add black edge for marker clarity
        linewidths=0.5       # Thin edge for elegance
    )

# Customize plot
plt.title("Agglomerative Clustering with Lifecycle Stage", fontsize=20, fontweight='medium')
plt.xlabel("Log-Scaled Recency", fontsize=18, fontweight='medium')  # Larger x-axis label
plt.ylabel("Log-Scaled Monetary", fontsize=18, fontweight='medium')  # Larger y-axis label
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    title='Cluster_Lifecycle',
    loc='upper right',
    fontsize=18,          # Larger legend font size
    title_fontsize=20,    # Larger legend title font size
    frameon=True
)
plt.tight_layout()
plt.savefig("images/scatterplot_agglo_lifecycle.png", dpi=600, bbox_inches='tight')
plt.show()

# Pairplot with enhanced clarity and larger clusters
pairplot = sns.pairplot(
    filtered_rfm_df,
    vars=['Recency', 'Frequency', 'Monetary'],
    hue='Clusters',
    palette='Set1',
    markers=list(itertools.islice(itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>']), len(filtered_rfm_df['Clusters'].unique()))),
    height=3  # Increase pairplot marker size
)

# Suppress the legend and customize
pairplot.fig.suptitle("Agglomerative Clusters RFM", y=1.02, fontsize=20, fontweight='medium')  # Larger plot title
pairplot.fig.set_size_inches(14, 10)
legend = pairplot._legend
legend.set_title("Clusters")
legend.set_bbox_to_anchor((1, 0.5))  # Adjust legend position
legend.set_frame_on(True)  # Add a frame
for text in legend.get_texts():
    text.set_fontsize(18)  # Larger font size for labels
legend.get_title().set_fontsize(20)  # Larger legend title font size

# Make axis labels bigger for the pairplot
for ax in pairplot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)  # Update x-axis label size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)  # Update y-axis label size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Update tick label size

plt.tight_layout()
pairplot.fig.savefig("images/pairplot_agglo_clusters.png", dpi=600, bbox_inches='tight')
plt.show()

Z = linkage(scaled_rfm, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, color_threshold=0, no_labels=True)  # Remove x-axis labels
plt.title("Dendrogram for Agglomerative Clustering", fontsize=20, fontweight='medium')
plt.xlabel("CustomerID", fontsize=18, fontweight='medium')
plt.ylabel("Distance", fontsize=18, fontweight='medium')
plt.xticks(fontsize=14)  # Adjust x-axis tick font size
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("images/dendrogram_agglo_clusters_no_labels.png", dpi=600, bbox_inches='tight')
plt.show()