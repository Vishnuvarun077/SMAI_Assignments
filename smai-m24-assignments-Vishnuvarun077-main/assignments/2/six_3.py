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
from models.GMM.GMM import GMM
from models.PCA.PCA import PCA

def load_data(filename):
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize the data
    return X

def plot_scree(pca,title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.title('Scree Plot')
    plt.savefig(title)
    plt.show()
    plt.close()

def elbow_method(X, max_k):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(X)
        wcss.append(kmeans.getCost())
    return wcss

def plot_elbow(wcss, max_k, optimal_k, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label='Elbow at k={}'.format(optimal_k))
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()

def compute_bic(X, gmm):
    n_samples, n_features = X.shape
    log_likelihood = gmm.getLikelihood(X)
    n_params = gmm.n_components * (n_features + 1 + n_features * (n_features + 1) / 2) - 1
    return -2 * log_likelihood + n_params * np.log(n_samples)

def compute_aic(X, gmm):
    log_likelihood = gmm.getLikelihood(X)
    n_params = gmm.n_components * (X.shape[1] + 1 + X.shape[1] * (X.shape[1] + 1) / 2) - 1
    return 2 * n_params - 2 * log_likelihood

def find_optimal_gmm(X, max_k):
    bic_values = []
    aic_values = []
    for k in range(1, max_k + 1):
        try:
            gmm = GMM(n_components=k)
            gmm.fit(X)
            bic = compute_bic(X, gmm)
            aic = compute_aic(X, gmm)
            bic_values.append(bic)
            aic_values.append(aic)
            print("k={}, BIC={:.2f}, AIC={:.2f}".format(k, bic, aic))
        except Exception as e:
            print("Error fitting GMM with k={}: {}".format(k, str(e)))
            break
    optimal_k_bic = np.argmin(bic_values) + 1 if bic_values else 1
    optimal_k_aic = np.argmin(aic_values) + 1 if aic_values else 1
    return optimal_k_bic, optimal_k_aic, bic_values, aic_values

def plot_bic_aic(bic_values, aic_values, optimal_k_bic, optimal_k_aic):
    plt.figure(figsize=(10, 6))
    k_values = range(1, len(bic_values) + 1)
    plt.plot(k_values, bic_values, marker='o', label='BIC')
    plt.plot(k_values, aic_values, marker='o', label='AIC')
    plt.axvline(x=optimal_k_bic, color='r', linestyle='--', label='Optimal k (BIC): {}'.format(optimal_k_bic))
    plt.axvline(x=optimal_k_aic, color='g', linestyle='--', label='Optimal k (AIC): {}'.format(optimal_k_aic))
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('BIC / AIC Score')
    plt.title('BIC and AIC Scores vs. Number of Clusters')
    plt.legend()
    plt.savefig('assignments/2/plots/q6/bic_aic_plot.png')
    plt.show()
    plt.close()

# Main execution
X = load_data('data/external/word-embeddings.feather')

# 6.1 K-means Clustering Based on 2D Visualization
pca_2d = PCA(n_components=2)
pca_2d.fit(X)
X_transformed_pca_2d = pca_2d.transform(X)

k2 = 5  # Estimated from 2D visualization
kmeans_2d = KMeans(k=k2, max_iters=100)
kmeans_2d.fit(X_transformed_pca_2d)
labels_2d = kmeans_2d.predict(X_transformed_pca_2d)

plt.figure(figsize=(8, 6))
plt.scatter(X_transformed_pca_2d[:, 0], X_transformed_pca_2d[:, 1], c=labels_2d, cmap='viridis', marker='o')
plt.title('2D PCA with K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('assignments/2/plots/q6/pca_2d_kmeans.png')
plt.show()
plt.close()

# 6.2 PCA + K-Means Clustering
pca = PCA(n_components=X.shape[1])  # Initialize PCA with all features
pca.fit(X)
plot_scree(pca,'assignments/2/plots/q6/scree_plot.png')

pca2 = PCA(n_components= 10)
pca2.fit(X)
plot_scree(pca2,'assignments/2/plots/q6/scree_plot_10.png')



# Choose optimal dimensions based on scree plot (e.g., elbow method or cumulative explained variance)
optimal_dims = 5  # This should be determined based on the scree plot
print("Optimal number of dimensions chosen: {}".format(optimal_dims))

pca_reduced = PCA(n_components=optimal_dims)
pca_reduced.fit(X)
X_reduced = pca_reduced.transform(X)

max_k =20
wcss_values = elbow_method(X_reduced, max_k)

kkmeans3 = 3  # This is determined based on the elbow plot
print("Optimal number of clusters for K-means (kkmeans3): {}".format(kkmeans3))
plot_elbow(wcss_values, max_k, kkmeans3, 'assignments/2/plots/q6/elbow_reduced.png')
kmeans_reduced = KMeans(k=kkmeans3, max_iters=100)
kmeans_reduced.fit(X_reduced)
labels_reduced = kmeans_reduced.predict(X_reduced)

# 6.3 GMM Clustering Based on 2D Visualization
gmm_2d = GMM(n_components=k2)
gmm_2d.fit(X_transformed_pca_2d)
labels_gmm_2d = gmm_2d.getMembership()

plt.figure(figsize=(8, 6))
plt.scatter(X_transformed_pca_2d[:, 0], X_transformed_pca_2d[:, 1], c=labels_gmm_2d, cmap='viridis', marker='o')
plt.title('2D PCA with GMM Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('assignments/2/plots/q6/pca_2d_gmm.png')
plt.show()
plt.close()

# 6.4 PCA + GMMs
optimal_k_bic, optimal_k_aic, bic_values, aic_values = find_optimal_gmm(X_reduced, max_k)
plot_bic_aic(bic_values, aic_values, optimal_k_bic, optimal_k_aic)

kgmm3 = optimal_k_bic
# kgmm3 = optimal_k_bic  # Use BIC for optimal number of clusters
print("Optimal number of clusters for GMM (kgmm3): {}".format(kgmm3))

gmm_reduced = GMM(n_components=kgmm3)
gmm_reduced.fit(X_reduced)
labels_gmm_reduced = gmm_reduced.getMembership()

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_gmm_reduced, cmap='viridis', marker='o')
plt.title('Reduced PCA with GMM Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('assignments/2/plots/q6/pca_reduced_gmm.png')
plt.show()
plt.close()


# # Choose optimal dimensions based on scree plot (e.g., elbow method or cumulative explained variance)
# optimal_dims = 5  # This should be determined based on the scree plot
# print("Optimal number of dimensions chosen: {}".format(optimal_dims))

# pca_reduced = PCA(n_components=optimal_dims)
# pca_reduced.fit(X)
# X_reduced = pca_reduced.transform(X)

# max_k =20
# wcss_values = elbow_method(X_reduced, max_k)

# kkmeans3 = 5  # This is determined based on the elbow plot
# print("Optimal number of clusters for K-means (kkmeans3): {}".format(kkmeans3))
# plot_elbow(wcss_values, max_k, kkmeans3, 'assignments/2/plots/q6/elbow_reduced.png')
# kmeans_reduced = KMeans(k=kkmeans3, max_iters=100)
# kmeans_reduced.fit(X_reduced)
# labels_reduced = kmeans_reduced.predict(X_reduced)

# # 6.3 GMM Clustering Based on 2D Visualization
# gmm_2d = GMM(n_components=k2)
# gmm_2d.fit(X_transformed_pca_2d)
# labels_gmm_2d = gmm_2d.getMembership()

# plt.figure(figsize=(8, 6))
# plt.scatter(X_transformed_pca_2d[:, 0], X_transformed_pca_2d[:, 1], c=labels_gmm_2d, cmap='viridis', marker='o')
# plt.title('2D PCA with GMM Clustering')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.savefig('assignments/2/plots/q6/pca_2d_gmm.png')
# plt.show()
# plt.close()

# # 6.4 PCA + GMMs
# optimal_k_bic, optimal_k_aic, bic_values, aic_values = find_optimal_gmm(X_reduced, max_k)
# plot_bic_aic(bic_values, aic_values, optimal_k_bic, optimal_k_aic)

# kgmm3 = optimal_k_bic
# # kgmm3 = optimal_k_bic  # Use BIC for optimal number of clusters
# print("Optimal number of clusters for GMM (kgmm3): {}".format(kgmm3))

# gmm_reduced = GMM(n_components=kgmm3)
# gmm_reduced.fit(X_reduced)
# labels_gmm_reduced = gmm_reduced.getMembership()

# plt.figure(figsize=(8, 6))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_gmm_reduced, cmap='viridis', marker='o')
# plt.title('Reduced PCA with GMM Clustering')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.savefig('assignments/2/plots/q6/pca_reduced_gmm.png')
# plt.show()
# plt.close()
