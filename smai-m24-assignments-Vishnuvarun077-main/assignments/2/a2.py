
import sys
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
# Adding the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.kmeans.KMeans import KMeans
from models.GMM.GMM import GMM
from models.PCA.PCA import PCA
from models.knn.knn import KNN
from performance_measures.performance_measures import PerformanceMeasures


# Q3 - KMEANS
def Q3_main():
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
    plt.savefig('assignments/2/figures/q3/elbow.png')
    plt.show()

    #Plot of first 20 values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 20 + 1), wcss_values[:20])
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.axvline(x=kkmeans1, color='r', linestyle='--', label='Elbow at k={}'.format(kkmeans1))
    plt.legend()
    plt.savefig('assignments/2/figures/q3/elbow2.png')
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


# Q4 - GMM
def load_data2(filename):
    # Load the dataset (word-embeddings.feather)
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def compute_bic(X, gmm):
    # Compute BIC for the GMM model
    n_samples, n_features = X.shape
    log_likelihood = gmm.getLikelihood(X)
    # n_params: Number of parameters (means, covariances, and mixing coefficients)
    n_params = gmm.n_components * (n_features + 1 + n_features * (n_features + 1) / 2) - 1
    return -2 * log_likelihood + n_params * np.log(n_samples)

def compute_aic(X, gmm):
    # Compute AIC for the GMM model
    log_likelihood = gmm.getLikelihood(X)
    n_params = gmm.n_components * (X.shape[1] + 1 + X.shape[1] * (X.shape[1] + 1) / 2) - 1
    return 2 * n_params - 2 * log_likelihood

def find_optimal_gmm(X, max_k):
    # Test different numbers of components and compute BIC and AIC scores
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
    
    if len(bic_values) > 0:
        # Find the optimal number of components according to BIC and AIC
        optimal_k_bic = np.argmin(bic_values) + 1
        optimal_k_aic = np.argmin(aic_values) + 1
    else:
        optimal_k_bic = optimal_k_aic = 2
    
    return optimal_k_bic, optimal_k_aic, bic_values, aic_values

def Q4_main():
    # Load and preprocess the data (word embeddings)
    X = load_data2('data/external/word-embeddings.feather')

    # Find the optimal number of clusters using BIC and AIC
    max_k = 20  # Max number of clusters to test
    kgmm1_bic, kgmm1_aic, bic_values, aic_values = find_optimal_gmm(X, max_k)

    print("Optimal number of clusters (BIC): {}".format(kgmm1_bic))
    print("Optimal number of clusters (AIC): {}".format(kgmm1_aic))

    # Plot the BIC and AIC values to visualize the optimal k
    if len(bic_values) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(bic_values) + 1), bic_values, marker='o', label='BIC')
        plt.plot(range(1, len(aic_values) + 1), aic_values, marker='o', label='AIC')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Score')
        plt.title('BIC and AIC Scores for GMM')
        plt.legend()
        plt.savefig('assignments/2/figures/q4/gmm_bic_aic.png')
        plt.show()
    else:
        print("No valid GMMs found. Unable to plot BIC and AIC curves.")

    # Fit GMM with the optimal number of clusters (using BIC)
    optimal_gmm = GMM(n_components=kgmm1_bic)
    optimal_gmm.fit(X)

    # Get predicted cluster memberships
    labels = optimal_gmm.getMembership()





##PCA PART - Q5
def load_data(filename):
    # Load the dataset (word-embeddings.feather)
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def Q5_main():
    # Load and preprocess the data (word embeddings)
    X = load_data('data/external/word-embeddings.feather')

    # Instantiate the PCA class for 2 components
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X)
    X_transformed_2d = pca_2d.transform(X)

    # Verify the functionality
    assert pca_2d.checkPCA(X), "PCA for 2 components failed."

    # Plot the 2D data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed_2d[:, 0], X_transformed_2d[:, 1], c='blue', marker='o')
    plt.title('2D PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('assignments/2/figures/q5/pca_2d.png')
    plt.show()

    # Instantiate the PCA class for 3 components
    pca_3d = PCA(n_components=3)
    pca_3d.fit(X)
    X_transformed_3d = pca_3d.transform(X)

    # Verify the functionality
    assert pca_3d.checkPCA(X), "PCA for 3 components failed."

    # Plot the 3D data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_transformed_3d[:, 0], X_transformed_3d[:, 1], X_transformed_3d[:, 2], c='blue', marker='o')
    ax.set_title('3D PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig('assignments/2/figures/q5/pca_3d.png')
    plt.show()


#Q6
def load_data(filename):
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize the data
    return X

def plot_scree(pca, title):
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
    plt.savefig('assignments/2/figures/q6/bic_aic_plot.png')
    plt.show()
    plt.close()

def Q6_main():
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
    plt.savefig('assignments/2/figures/q6/pca_2d_kmeans.png')
    plt.show()
    plt.close()

    # 6.2 PCA + K-Means Clustering
    pca = PCA(n_components=X.shape[1])  # Initialize PCA with all features
    pca.fit(X)
    plot_scree(pca, 'assignments/2/figures/q6/scree_plot.png')

    pca2 = PCA(n_components=10)
    pca2.fit(X)
    plot_scree(pca2, 'assignments/2/figures/q6/scree_plot_10.png')

    # Choose optimal dimensions based on scree plot (e.g., elbow method or cumulative explained variance)
    optimal_dims = 5  # This should be determined based on the scree plot
    print("Optimal number of dimensions chosen: {}".format(optimal_dims))

    pca_reduced = PCA(n_components=optimal_dims)
    pca_reduced.fit(X)
    X_reduced = pca_reduced.transform(X)

    max_k = 20
    wcss_values = elbow_method(X_reduced, max_k)

    kkmeans3 = 3  # This is determined based on the elbow plot
    print("Optimal number of clusters for K-means (kkmeans3): {}".format(kkmeans3))
    plot_elbow(wcss_values, max_k, kkmeans3, 'assignments/2/figures/q6/elbow_reduced.png')
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
    plt.savefig('assignments/2/figures/q6/pca_2d_gmm.png')
    plt.show()
    plt.close()

    # 6.4 PCA + GMMs
    optimal_k_bic, optimal_k_aic, bic_values, aic_values = find_optimal_gmm(X_reduced, max_k)
    plot_bic_aic(bic_values, aic_values, optimal_k_bic, optimal_k_aic)

    kgmm3 = optimal_k_bic
    print("Optimal number of clusters for GMM (kgmm3): {}".format(kgmm3))

    gmm_reduced = GMM(n_components=kgmm3)
    gmm_reduced.fit(X_reduced)
    labels_gmm_reduced = gmm_reduced.getMembership()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_gmm_reduced, cmap='viridis', marker='o')
    plt.title('Reduced PCA with GMM Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('assignments/2/figures/q6/pca_reduced_gmm.png')
    plt.show()
    plt.close()


#Q7
def Q7_main():
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

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
        return X_reduced

    # Reduce the data to 2 dimensions
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
    kgmm1, kgmm3 = 2, 3
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


#Q8
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
    plt.savefig('assignments/2/figures/q8/dendrogram_{}_{}.png'.format(linkage_method, distance_metric))
    plt.show()
    
    return linkage_matrix

# Function to cut the dendrogram and form clusters
def cut_dendrogram(linkage_matrix, k):
    return fcluster(linkage_matrix, k, criterion='maxclust')

def Q8_main():
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
            kbest1 = 3  # Best k from K-means clustering
            kbest2 = 2  # Best k from GMM clustering
            
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



#Q9
# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path, target_column):
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    # Convert non-numeric columns to numeric where possible
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that cannot be converted to numeric
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

    
    # Normalize the features (Min-Max scaling)
    X = (X - X.min()) / (X.max() - X.min())
    
    # Handle categorical labels
    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    
    return X.values, y.values

# Custom train-test split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(X.shape[0] * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Generate Scree Plot to find optimal number of components
def generate_scree_plot(X):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title("Scree Plot")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.savefig('assignments/2/figures/q9/scree_plot.png')
    plt.show()

    return pca

# KNN Classification
def knn_classification(X_train, X_test, y_train, y_test, k, distance_metric):
    knn = KNN(k=k, distance_metric=distance_metric)
    knn.fit(X_train, y_train)
    
    # Predict and evaluate performance
    start_time = time.time()
    y_pred = knn.predict(X_test)
    inference_time = time.time() - start_time
    
    pm = PerformanceMeasures()
    accuracy = pm.accuracy(y_test, y_pred)
    precision = pm.precision(y_test, y_pred, average='macro')
    recall = pm.recall(y_test, y_pred, average='macro')
    f1_macro = pm.f1_score_macro(y_test, y_pred)
    f1_micro = pm.f1_score_micro(y_test, y_pred)

    return accuracy, precision, recall, f1_macro, f1_micro, inference_time

# Main function for 9.1 PCA + KNN and 9.2 Evaluation
def Q9_main():
    # Load the dataset
    file_path = 'data/external/spotify.csv'  # Update with your dataset path
    target_column = 'track_genre'  # Replace with the actual target column
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, target_column)
    
    # Step 1: Generate Scree Plot and perform PCA
    print("\nGenerating Scree Plot to determine optimal dimensions...")
    pca = generate_scree_plot(X)
    
    # Let's assume we decide to keep 3 components based on the scree plot
    optimal_components = 3
    pca = PCA(n_components=optimal_components)
    pca.fit(X)
    X_reduced = pca.transform(X)
    
    # Split the data into training and test sets using custom function
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_reduced, X_test_reduced, _, _ = custom_train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # Step 2: Apply KNN with the best {k, distance metric} pair (k=19, distance='manhattan')
    k_best = 19
    distance_best = 'manhattan'
    
    print("\nPerforming KNN on full dataset...")
    full_accuracy, full_precision, full_recall, full_f1_macro, full_f1_micro, full_inference_time = knn_classification(
        X_train, X_test, y_train, y_test, k_best, distance_best)
    
    print("\nPerforming KNN on PCA-reduced dataset...")
    reduced_accuracy, reduced_precision, reduced_recall, reduced_f1_macro, reduced_f1_micro, reduced_inference_time = knn_classification(
        X_train_reduced, X_test_reduced, y_train, y_test, k_best, distance_best)
    
    # Step 3: Print evaluation metrics for full and reduced datasets
    print("\n---- KNN Performance on Full Dataset ----")
    print("Accuracy: {:.2f}".format(full_accuracy))
    print("Precision: {:.2f}".format(full_precision))
    print("Recall: {:.2f}".format(full_recall))
    print("F1 Score (Macro): {:.2f}".format(full_f1_macro))
    print("F1 Score (Micro): {:.2f}".format(full_f1_micro))
    print("Inference Time: {:.4f} seconds".format(full_inference_time))
    
    print("\n---- KNN Performance on PCA-Reduced Dataset ----")
    print("Accuracy: {:.2f}".format(reduced_accuracy))
    print("Precision: {:.2f}".format(reduced_precision))
    print("Recall: {:.2f}".format(reduced_recall))
    print("F1 Score (Macro): {:.2f}".format(reduced_f1_macro))
    print("F1 Score (Micro): {:.2f}".format(reduced_f1_micro))
    print("Inference Time: {:.4f} seconds".format(reduced_inference_time))
    
    # Step 4: Compare inference times
    inference_times = [full_inference_time, reduced_inference_time]
    dataset_labels = ['Full Dataset', 'PCA-Reduced Dataset']
    
    plt.figure(figsize=(10, 6))
    plt.bar(dataset_labels, inference_times)
    plt.title('Inference Time Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig('assignments/2/figures/q9/inference_time_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Q3_main()
    # Q4_main()
    # Q5_main()
    # Q6_main()
    # Q7_main()
    Q8_main()
    # Q9_main()



