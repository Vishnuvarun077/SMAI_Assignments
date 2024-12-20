import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def fit(self, X):
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape
        
        # Initialize centroids
        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = self.X[random_sample_indices]
        
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            # Calculate new centroids from the clusters
            centroids_old = self.centroids.copy()
            self.centroids = self._get_centroids(self.clusters)
            
            # Check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break
        
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._closest_centroid(sample, self.centroids) for sample in X])

    def getCost(self):
        wcss = sum(np.sum((self.X[idx] - self.centroids[label])**2)
                    for label, cluster in enumerate(self.clusters)
                    for idx in cluster)
        return wcss

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:  # Check if the cluster is not empty
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
            else:
                # If the cluster is empty, choose a random point
                centroids[cluster_idx] = self.X[np.random.choice(self.n_samples)]
        return centroids

    def _closest_centroid(self, sample, centroids):
        distances = np.sum((sample - centroids)**2, axis=1)
        return np.argmin(distances)

    def _is_converged(self, centroids_old, centroids):
        return np.all(centroids_old == centroids)













