
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Adding the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.kmeans.KMeans import KMeans




def load_data(filename):
    df = pd.read_feather(filename)
    df.dropna(inplace=True)
    return df

def standardize_data(data):
    return (data - data.mean()) / data.std()


def elbow_method(X, max_k):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(X)
        wcss.append(kmeans.getCost())
    return wcss



def print_data_summary(data, title):
    print("\n{}".format(title))
    print("Shape:", data.shape)
    print("\nSummary statistics:")
    print(data.describe())
    print("\nFirst few rows:")
    print(data.head())

# Load the data
df = load_data('data/external/word-embeddings.feather')

# Convert 'vit' column to a DataFrame with separate columns
vit_df = pd.DataFrame(df['vit'].tolist(), index=df.index)

# Print summary of raw data
print_data_summary(vit_df, "Raw Data Summary")

# Process the data
X = vit_df.values
X_normalized = X

# # Print summary of processed data
# print_data_summary(X_normalized, "Processed Data Summary (After normalization)")

# Run elbow method
vit_df = pd.DataFrame(df['vit'].tolist(), index=df.index)

# Print summary of raw data
print_data_summary(vit_df, "Raw Data Summary")

# Process the data
X = vit_df.values
X_normalized = standardize_data(X)
max_k = 200
wcss_values = elbow_method(X_normalized, max_k)

# Print WCSS values
print("\nWCSS values for k=1 to k=20:")
for k, wcss_value in enumerate(wcss_values, start=1):
    print("k={}: {}".format(k, wcss_value))

# Find the elbow point
kkmeans1 = 4

print("\n optimal number of clusters (kkmeans1): {}".format(kkmeans1))

# Perform K-means clustering with the optimal k
optimal_kmeans = KMeans(k=kkmeans1, max_iters=100)
optimal_kmeans.fit(X_normalized)

print("Final WCSS: {}".format(optimal_kmeans.getCost()))

# Optional: Print cluster sizes
labels = optimal_kmeans.predict(X_normalized)
unique, counts = np.unique(labels, return_counts=True)
print("\nCluster sizes:")
for cluster, size in zip(unique, counts):
    print("Cluster {}: {} samples".format(cluster, size))

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss_values)
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.axvline(x=kkmeans1, color='r', linestyle='--', label='Elbow at k={}'.format(kkmeans1))
plt.legend()
plt.savefig('assignments/2/plots/elbow.png')
plt.show()

#Plot of first 20 values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20 + 1), wcss_values[:20])
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.axvline(x=kkmeans1, color='r', linestyle='--', label='Elbow at k={}'.format(kkmeans1))
plt.legend()
plt.savefig('assignments/2/plots/elbow2.png')
plt.show()


# K2 after PCA
k2 = 6

print("\n optimal number of clusters (kkmeans1): {}".format(k2))

# Perform K-means clustering with the optimal k
optimal_kmeans = KMeans(k=k2, max_iters=100)
optimal_kmeans.fit(X_normalized)

print("Final WCSS: {}".format(optimal_kmeans.getCost()))

# Optional: Print cluster sizes
labels = optimal_kmeans.predict(X_normalized)
unique, counts = np.unique(labels, return_counts=True)
print("\nCluster sizes:")
for cluster, size in zip(unique, counts):
    print("Cluster {}: {} samples".format(cluster, size))



