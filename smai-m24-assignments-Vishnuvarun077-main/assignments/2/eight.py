


# import sys
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as sch
# from scipy.cluster.hierarchy import fcluster

# # Adding the parent directory to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.append(parent_dir)
# from models.PCA.PCA import PCA

# def load_data(filename):
#     df = pd.read_feather(filename)
#     X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
#     return X

# # Function to perform and plot hierarchical clustering
# def hierarchical_clustering(X_reduced, linkage_method='ward', distance_metric='euclidean'):
#     # Compute the linkage matrix
#     linkage_matrix = sch.linkage(X_reduced, method=linkage_method, metric=distance_metric)

#     # Plot the dendrogram
#     plt.figure(figsize=(10, 6))
#     sch.dendrogram(linkage_matrix)
#     plt.title('Dendrogram using {} linkage and {} metric'.format(linkage_method, distance_metric))
#     plt.xlabel('Samples')
#     plt.ylabel('Distance')
#     plt.savefig('assignments/2/plots/q8/dendrogram_{}_{}.png'.format(linkage_method, distance_metric))
#     plt.show()
    
#     return linkage_matrix

# # Function to cut the dendrogram and form clusters
# def cut_dendrogram(linkage_matrix, k):
#     return fcluster(linkage_matrix, k, criterion='maxclust')

# # Main execution
# X = load_data('data/external/word-embeddings.feather')

# # Apply PCA for dimensionality reduction before hierarchical clustering
# pca = PCA(n_components=5)  # Choose a suitable number of components based on prior analysis
# pca.fit(X)
# X_reduced = pca.transform(X)

# # Experiment with different linkage methods and distance metrics
# linkage_methods = ['ward', 'complete', 'average', 'single']
# distance_metrics = ['euclidean', 'cosine']  # Euclidean is the default, but you can experiment with others

# # Store the clusters for comparison
# clusters_dict = {}

# for method in linkage_methods:
#     for metric in distance_metrics:
#         if method == 'ward' and metric != 'euclidean':  # Ward method requires Euclidean distance
#             continue
#         print("Performing hierarchical clustering with {} linkage and {} metric".format(method, metric))
#         linkage_matrix = hierarchical_clustering(X_reduced, linkage_method=method, distance_metric=metric)
        
#         # Cut the dendrogram to form clusters for kbest1 (from K-means) and kbest2 (from GMM)
#         kbest1 = 5  # Best k from K-means clustering
#         kbest2 = 3  # Best k from GMM clustering
        
#         clusters_kbest1 = cut_dendrogram(linkage_matrix, kbest1)
#         clusters_kbest2 = cut_dendrogram(linkage_matrix, kbest2)
        
#         clusters_dict[(method, metric, 'kbest1')] = clusters_kbest1
#         clusters_dict[(method, metric, 'kbest2')] = clusters_kbest2
        
#         print("Clusters for K-means (kbest1={}) with {} linkage and {} metric:".format(kbest1, method, metric), np.unique(clusters_kbest1))
#         print("Clusters for GMM (kbest2={}) with {} linkage and {} metric:".format(kbest2, method, metric), np.unique(clusters_kbest2))

# # Optional: Compare clusters from different linkage methods and metrics with the K-Means and GMM results
# for method_metric, clusters in clusters_dict.items():
#     method, metric, kbest = method_metric
#     print("Cluster comparison for {} linkage, {} metric, {}:".format(method, metric, kbest))
#     print("Unique clusters:", np.unique(clusters))


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

# Adding the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.PCA.PCA import PCA

def load_data(filename):
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    return X

# Function to perform and plot hierarchical clustering
def hierarchical_clustering(X_reduced, linkage_method='complete', distance_metric='euclidean'):
    # Compute the linkage matrix
    linkage_matrix = sch.linkage(X_reduced, method=linkage_method, metric=distance_metric)

    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    sch.dendrogram(linkage_matrix)
    plt.title('Dendrogram using {} linkage and {} metric'.format(linkage_method, distance_metric))
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig('assignments/2/plots/q8/dendrogram_{}_{}.png'.format(linkage_method, distance_metric))
    plt.show()
    
    return linkage_matrix

# Function to cut the dendrogram and form clusters
def cut_dendrogram(linkage_matrix, k):
    return fcluster(linkage_matrix, k, criterion='maxclust')

# Main execution
X = load_data('data/external/word-embeddings.feather')

# Apply PCA for dimensionality reduction before hierarchical clustering
pca = PCA(n_components=5)  # Choose a suitable number of components based on prior analysis
pca.fit(X)
X_reduced = pca.transform(X)

# Experiment with different linkage methods and distance metrics
linkage_methods = ['complete', 'average', 'single']
distance_metrics = ['euclidean', 'cosine']  # Experimenting with Euclidean and Cosine distance metrics

# Store the clusters for comparison
clusters_dict = {}

for method in linkage_methods:
    for metric in distance_metrics:
        print("Performing hierarchical clustering with {} linkage and {} metric".format(method, metric))
        linkage_matrix = hierarchical_clustering(X_reduced, linkage_method=method, distance_metric=metric)
        
        # Cut the dendrogram to form clusters for kbest1 (from K-means) and kbest2 (from GMM)
        kbest1 = 5  # Best k from K-means clustering
        kbest2 = 3  # Best k from GMM clustering
        
        clusters_kbest1 = cut_dendrogram(linkage_matrix, kbest1)
        clusters_kbest2 = cut_dendrogram(linkage_matrix, kbest2)
        
        clusters_dict[(method, metric, 'kbest1')] = clusters_kbest1
        clusters_dict[(method, metric, 'kbest2')] = clusters_kbest2
        
        print("Clusters for K-means (kbest1={}) with {} linkage and {} metric:".format(kbest1, method, metric), np.unique(clusters_kbest1))
        print("Clusters for GMM (kbest2={}) with {} linkage and {} metric:".format(kbest2, method, metric), np.unique(clusters_kbest2))

# Optional: Compare clusters from different linkage methods and metrics with the K-Means and GMM results
for method_metric, clusters in clusters_dict.items():
    method, metric, kbest = method_metric
    print("Cluster comparison for {} linkage, {} metric, {}:".format(method, metric, kbest))
    print("Unique clusters:", np.unique(clusters))

