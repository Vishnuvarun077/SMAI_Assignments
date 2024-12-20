import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Check if mean has been computed
        if self.mean is None:
            raise ValueError("PCA has not been fitted. Call fit() before using transform().")
        
        # Center the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def check_pca(self, X, threshold=1e-5):
        # Transform the data
        X_transformed = self.transform(X)
        
        # Reconstruct the data
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        
        # Compute the reconstruction error
        error = np.mean((X - X_reconstructed) ** 2)
        
        # Check if the error is below the threshold
        return error < threshold


# import numpy as np

# class PCA:
#     def __init__(self, n_components):
#         self.n_components = n_components
#         self.components = None
#         self.mean = None

#     def fit(self, X):
#         # Center the data
#         self.mean = np.mean(X, axis=0)
#         X_centered = X - self.mean

#         # Compute the covariance matrix
#         cov = np.cov(X_centered, rowvar=False)

#         # Compute eigenvalues and eigenvectors
#         eigenvalues, eigenvectors = np.linalg.eigh(cov)

#         # Sort eigenvectors by decreasing eigenvalues
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvectors = eigenvectors[:, idx]

#         # Store the first n_components eigenvectors
#         self.components = eigenvectors[:, :self.n_components]

#     def transform(self, X):
#         # Check if mean has been computed
#         if self.mean is None:
#             raise ValueError("PCA has not been fitted. Call fit() before using transform().")
        
#         # Center the data
#         X_centered = X - self.mean

#         # Project the data onto the principal components
#         return np.dot(X_centered, self.components)

#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)
    
    
    
    