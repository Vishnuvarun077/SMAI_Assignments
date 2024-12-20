
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from models.PCA.PCA import PCA
from models.kmeans.KMeans import KMeans  # Custom KMeans model
from models.GMM.GMM import GMM  # Custom GMM model

# Load the data from the dataset
def load_data(filename):
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    return X

# Evaluate the clustering results using multiple metrics
def evaluate_clusters(model, X, labels):
    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    if isinstance(model, KMeans):
        wcss = model.getCost()
        return silhouette, calinski_harabasz, wcss
    elif isinstance(model, GMM):
        log_likelihood = model.getLikelihood(X)
        return silhouette, calinski_harabasz, log_likelihood

# Analyze the clustering results (for both KMeans and GMM)
def analyze_clusters(model, X, method_name):
    if isinstance(model, GMM):
        membership = model.getMembership()
        labels = np.argmax(membership, axis=1)  # Convert responsibilities to labels
    else:
        labels = model.predict(X)
    
    metrics = evaluate_clusters(model, X, labels)
    
    print("\n{} Clustering Results:".format(method_name))
    print("Silhouette score: {:.4f}".format(metrics[0]))
    print("Calinski-Harabasz Index: {:.4f}".format(metrics[1]))
    if isinstance(model, KMeans):
        print("WCSS: {:.4f}".format(metrics[2]))
    # else:
        # print("Log-likelihood: {:.4f}".format(metrics[2]))
    
    return metrics, labels

# Load the data
X = load_data('data/external/word-embeddings.feather')

# Apply PCA for dimensionality reduction
def reduce_dimensions(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)  # First, fit the PCA model to the data
    X_reduced = pca.transform(X)  # Then, transform the data
    print("\nReduced data to {} dimensions using PCA".format(n_components))
    return X_reduced

# Reduce the data to 5 dimensions
X_reduced = reduce_dimensions(X, n_components=2)

# 7.1 K-Means Cluster Analysis
kkmeans1, k2, kkmeans3 = 4, 5, 3
kmeans_models = {
    'kkmeans1': KMeans(k=kkmeans1, max_iters=100),
    'k2': KMeans(k=k2, max_iters=100),
    'kkmeans3': KMeans(k=kkmeans3, max_iters=100)
}

# Analyze K-Means Clustering
kmeans_results = {}
for name, model in kmeans_models.items():
    model.fit(X_reduced)  # Use reduced data
    metrics, labels = analyze_clusters(model, X_reduced, "K-Means ({})".format(name))
    kmeans_results[name] = (metrics, labels)

# Identify the best k for K-Means based on silhouette score
kkmeans = max(kmeans_results, key=lambda k: kmeans_results[k][0][0])
print("\nBest K-Means clustering: {} with silhouette score {:.4f}".format(kkmeans, kmeans_results[kkmeans][0][0]))

# 7.2 GMM Cluster Analysis
kgmm1, kgmm3 = 3, 3
gmm_models = {
    'kgmm1': GMM(n_components=kgmm1),
    'k2': GMM(n_components=k2),
    'kgmm3': GMM(n_components=kgmm3)
}

# Analyze GMM Clustering
gmm_results = {}
for name, model in gmm_models.items():
    model.fit(X_reduced)  # Use reduced data
    metrics, labels = analyze_clusters(model, X_reduced, "GMM ({})".format(name))
    gmm_results[name] = (metrics, labels)

# Identify the best k for GMM based on silhouette score
kgmm = max(gmm_results, key=lambda k: gmm_results[k][0][0])
print("\nBest GMM clustering: {} with silhouette score {:.4f}".format(kgmm, gmm_results[kgmm][0][0]))

# 7.3 Compare K-Means and GMMs
print("\n7.3 Comparison of K-Means and GMM")
print("Best K-Means ({}):".format(kkmeans))
print("  Silhouette score: {:.4f}".format(kmeans_results[kkmeans][0][0]))
print("  Calinski-Harabasz Index: {:.4f}".format(kmeans_results[kkmeans][0][1]))
print("  WCSS: {:.4f}".format(kmeans_results[kkmeans][0][2]))

print("\nBest GMM ({}):".format(kgmm))
print("  Silhouette score: {:.4f}".format(gmm_results[kgmm][0][0]))
print("  Calinski-Harabasz Index: {:.4f}".format(gmm_results[kgmm][0][1]))

# Visualize the comparison between K-Means and GMM
def compare_and_plot(kmeans_results, gmm_results):
    plt.figure(figsize=(12, 6))
    
    # Silhouette Score
    plt.subplot(1, 2, 1)
    plt.bar(['K-Means', 'GMM'], [kmeans_results[kkmeans][0][0], gmm_results[kgmm][0][0]])
    plt.title('Comparison of Silhouette Scores')
    plt.ylabel('Silhouette Score')
    
    # Calinski-Harabasz Index
    plt.subplot(1, 2, 2)
    plt.bar(['K-Means', 'GMM'], [kmeans_results[kkmeans][0][1], gmm_results[kgmm][0][1]])
    plt.title('Comparison of Calinski-Harabasz Index')
    plt.ylabel('Calinski-Harabasz Index')
    
    plt.tight_layout()
    plt.savefig('assignments/2/figures/q7/kmeans_gmm_comparison.png', dpi=300)
    plt.close()

compare_and_plot(kmeans_results, gmm_results)
