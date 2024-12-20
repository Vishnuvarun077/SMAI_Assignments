# # import sys
# # import os
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Add the parent directory to the Python path
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # parent_dir = os.path.dirname(os.path.dirname(current_dir))
# # sys.path.append(parent_dir)

# # from models.GMM.GMM import GMM

# # def load_data(filename):
# #     # Load the dataset (word-embeddings.feather)
# #     df = pd.read_feather(filename)
# #     X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
# #     # Normalize the data
# #     X = (X - X.mean(axis=0)) / X.std(axis=0)
# #     return X

# # def compute_bic(X, gmm):
# #     # Compute BIC for the GMM model
# #     n_samples, n_features = X.shape
# #     log_likelihood = gmm.getLikelihood(X)
# #     # n_params: Number of parameters (means, covariances, and mixing coefficients)
# #     n_params = gmm.n_components * (n_features + 1 + n_features * (n_features + 1) / 2) - 1
# #     return -2 * log_likelihood + n_params * np.log(n_samples)

# # def compute_aic(X, gmm):
# #     # Compute AIC for the GMM model
# #     log_likelihood = gmm.getLikelihood(X)
# #     n_params = gmm.n_components * (X.shape[1] + 1 + X.shape[1] * (X.shape[1] + 1) / 2) - 1
# #     return 2 * n_params - 2 * log_likelihood

# # def find_optimal_gmm(X, max_k):
# #     # Test different numbers of components and compute BIC and AIC scores
# #     bic_values = []
# #     aic_values = []
    
# #     for k in range(1, max_k + 1):
# #         try:
# #             gmm = GMM(n_components=k)
# #             gmm.fit(X)
# #             bic = compute_bic(X, gmm)
# #             aic = compute_aic(X, gmm)
# #             bic_values.append(bic)
# #             aic_values.append(aic)
# #             print("k={}, BIC={:.2f}, AIC={:.2f}".format(k, bic, aic))
# #         except Exception as e:
# #             print("Error fitting GMM with k={}: {}".format(k, str(e)))
# #             break
    
# #     if len(bic_values) > 0:
# #         # Find the optimal number of components according to BIC and AIC
# #         optimal_k_bic = np.argmin(bic_values) + 1
# #         optimal_k_aic = np.argmin(aic_values) + 1
# #     else:
# #         optimal_k_bic = optimal_k_aic = 1
    
# #     return optimal_k_bic, optimal_k_aic, bic_values, aic_values

# # # Load and preprocess the data (word embeddings)
# # X = load_data('data/external/word-embeddings.feather')

# # # Find the optimal number of clusters using BIC and AIC
# # max_k = 20  # Max number of clusters to test
# # kgmm1_bic, kgmm1_aic, bic_values, aic_values = find_optimal_gmm(X, max_k)

# # print("Optimal number of clusters (BIC): {}".format(kgmm1_bic))
# # print("Optimal number of clusters (AIC): {}".format(kgmm1_aic))

# # # Plot the BIC and AIC values to visualize the optimal k
# # if len(bic_values) > 0:
# #     plt.figure(figsize=(12, 6))
# #     plt.plot(range(1, len(bic_values) + 1), bic_values, marker='o', label='BIC')
# #     plt.plot(range(1, len(aic_values) + 1), aic_values, marker='o', label='AIC')
# #     plt.xlabel('Number of clusters (k)')
# #     plt.ylabel('Score')
# #     plt.title('BIC and AIC Scores for GMM')
# #     plt.legend()
# #     plt.savefig('assignments/2/plots/gmm_bic_aic.png')
# #     plt.show()
# # else:
# #     print("No valid GMMs found. Unable to plot BIC and AIC curves.")

# # # Fit GMM with the optimal number of clusters (using BIC)
# #     optimal_gmm = GMM(n_components=kgmm1_bic)
# #     optimal_gmm.fit(X)

# #     # Get predicted cluster memberships
# #     labels = optimal_gmm.getMembership()

# #     # Print the size of each cluster
# #     unique, counts = np.unique(labels, return_counts=True)
# #     print("\nCluster sizes:")
# #     for cluster, size in zip(unique, counts):
# #         print("Cluster {}: {} samples".format(cluster, size))



import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from models.GMM.GMM import GMM

def load_data(filename):
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
    print("Log-likelihood for k={}: {}".format(gmm.n_components, log_likelihood))
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

# Load and preprocess the data (word embeddings)
X = load_data('data/external/word-embeddings.feather')

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
    plt.savefig('assignments/2/plots/gmm_bic_aic.png')
    plt.show()
else:
    print("No valid GMMs found. Unable to plot BIC and AIC curves.")

# Fit GMM with the optimal number of clusters (using BIC)
optimal_gmm = GMM(n_components=kgmm1_bic)
optimal_gmm.fit(X)

# Get predicted cluster memberships
labels = optimal_gmm.getMembership()

# Print the size of each cluster
unique, counts = np.unique(labels, return_counts=True)
print("\nCluster sizes:")
for cluster, size in zip(unique, counts):
    print("Cluster {}: {} samples".format(cluster, size))

