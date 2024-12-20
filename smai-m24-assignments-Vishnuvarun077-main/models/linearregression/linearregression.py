import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, reg_type=None, lambda_=0, degree=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.degree = degree
        self.coefficients = None
        self.mse_history = []
        self.std_dev_history = []
        self.variance_history = []

    def add_polynomial_features_for_animation(self, X):
        X_poly = X
        for power in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, np.power(X, power)))
        return X_poly
    def add_polynomial_features(self, X):
        from itertools import combinations_with_replacement
        n_samples, n_features = X.shape
        comb = combinations_with_replacement(range(n_features), self.degree)
        poly_features = np.ones((n_samples, 1))
        for indices in comb:
            new_feature = np.prod(X[:, indices], axis=1).reshape(-1, 1)
            poly_features = np.hstack((poly_features, new_feature))
        return poly_features

    def fit(self, X, y):
        if self.degree > 1:
            X = self.add_polynomial_features(X)
        else:
            X = np.c_[np.ones(X.shape[0]), X]
        m, n = X.shape
        self.coefficients = np.zeros(n)

        for _ in range(self.iterations):
            y_pred = X.dot(self.coefficients)
            error = y_pred - y
            gradient = X.T.dot(error) / m
            # Apply regularization if specified (excluding the intercept term)
            if self.reg_type == 'L1':
                gradient[1:] += self.lambda_ * np.sign(self.coefficients[1:])
            elif self.reg_type == 'L2':
                gradient[1:] += self.lambda_ * self.coefficients[1:]

            # Update coefficients
            self.coefficients -= self.learning_rate * gradient

            # Update history
            self.mse_history.append(np.mean(error ** 2))
            self.std_dev_history.append(np.std(error))
            self.variance_history.append(np.var(error))

    def predict(self, X):
        if self.degree > 1:
            X = self.add_polynomial_features(X)
        else:
            X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.coefficients)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def std_dev(self, X, y):
        y_pred = self.predict(X)
        return np.std(y - y_pred)

    def variance(self, X, y):
        y_pred = self.predict(X)
        return np.var(y - y_pred)