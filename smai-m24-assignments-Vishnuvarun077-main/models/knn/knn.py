import numpy as np

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', batch_size=100):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.batch_size = batch_size
        self.distance_functions = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_distance
        }

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples, dtype=self.y_train.dtype)
        
        for i in range(0, num_samples, self.batch_size):
            batch = X[i:i+self.batch_size]
            distances = self.distance_functions[self.distance_metric](batch, self.X_train)
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_indices]
            batch_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=k_nearest_labels)
            predictions[i:i+self.batch_size] = batch_predictions
        
        return predictions

    @staticmethod
    def euclidean_distance(X1, X2):
        return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2) ** 2, axis=2))

    @staticmethod
    def manhattan_distance(X1, X2):
        return np.sum(np.abs(X1[:, np.newaxis, :] - X2), axis=2)

    @staticmethod
    def cosine_distance(X1, X2):
        dot_product = np.einsum('ijk,jk->ij', X1[:, np.newaxis, :], X2)
        norm_X1 = np.linalg.norm(X1, axis=1)
        norm_X2 = np.linalg.norm(X2, axis=1)
        return 1 - (dot_product / (norm_X1[:, np.newaxis] * norm_X2))


# import numpy as np

# class KNN:
class initial_KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.distance_functions = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_distance
        }

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self.prediction_helper(x) for x in X])

    def prediction_helper(self, x):
        distances = self.distance_functions[self.distance_metric](x, self.X_train)
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.bincount(k_nearest_labels).argmax()

    @staticmethod
    def euclidean_distance(x1, X):
        return np.sqrt(np.sum((X - x1) ** 2, axis=1))

    @staticmethod
    def manhattan_distance(x1, X):
        return np.sum(np.abs(X - x1), axis=1)

    @staticmethod
    def cosine_distance(x1, X):
        dot_product = np.dot(X, x1)
        norm_x1 = np.linalg.norm(x1)
        norm_X = np.linalg.norm(X, axis=1)
        return 1 - (dot_product / (norm_X * norm_x1))



# class initial_KNN:
#     def __init__(self, k=3, distance_metric='euclidean'):
#         self.k = k
#         self.distance_metric = distance_metric
#         self.X_train = None
#         self.y_train = None
#         self.distance_functions = {
#             'euclidean': self.euclidean_distance,
#             'manhattan': self.manhattan_distance,
#             'cosine': self.cosine_distance
#         }

#     def fit(self, X, y):
#             self.X_train = X
#             self.y_train = y

#     def predict(self, X):
#             predictions = np.empty(X.shape[0], dtype=self.y_train.dtype)
#             for i, x in enumerate(X):
#                 distances = self.distance_functions[self.distance_metric](x, self.X_train)
#                 k_indices = np.argpartition(distances, self.k)[:self.k]
#                 k_nearest_labels = self.y_train[k_indices]
#                 predictions[i] = np.bincount(k_nearest_labels).argmax()
#             return predictions

#     @staticmethod
#     def euclidean_distance(x1, X):
#             return np.sum((X - x1) ** 2, axis=1)

#     @staticmethod
#     def manhattan_distance(x1, X):
#             return np.sum(np.abs(X - x1), axis=1)

#     @staticmethod
#     def cosine_distance(x1, X):
#             dot_product = np.dot(X, x1)
#             norm_x1 = np.dot(x1, x1)
#             norm_X = np.sum(X ** 2, axis=1)
#             return 1 - dot_product / np.sqrt(norm_X * norm_x1)
