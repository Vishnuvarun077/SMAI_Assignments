import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from models.PCA.PCA import PCA

def load_data(filename):
    # Load the dataset (word-embeddings.feather)
    df = pd.read_feather(filename)
    X = np.array(df['vit'].tolist())  # The 512-dimensional embeddings
    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

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
plt.savefig('assignments/2/plots/pca_2d.png')
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
plt.savefig('assignments/2/plots/pca_3d.png')
plt.show()