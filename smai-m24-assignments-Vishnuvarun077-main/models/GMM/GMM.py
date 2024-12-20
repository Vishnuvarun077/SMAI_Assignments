

import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.mixing_coefficients_ = None
        self.resp_ = None

    def initialize_params(self, X):
        n_samples, n_features = X.shape
        
        self.means_ = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(self.n_components, n_features))
        
        self.covariances_ = []
        for _ in range(self.n_components):
            A = np.random.randn(n_features, n_features)
            self.covariances_.append(np.dot(A.T, A) + np.eye(n_features) * 1e-6)  # Ensure positive definiteness
        
        self.covariances_ = np.array(self.covariances_)
        
        self.mixing_coefficients_ = np.random.dirichlet(np.ones(self.n_components))
        
    def e_step(self, X):
        n_samples, _ = X.shape
        self.resp_ = np.zeros((n_samples, self.n_components))
        
        for c in range(self.n_components):
            rv = multivariate_normal(self.means_[c], self.covariances_[c])
            self.resp_[:, c] = self.mixing_coefficients_[c] * rv.pdf(X)
        
        self.resp_ /= (np.sum(self.resp_, axis=1, keepdims=True) + 1e-1)
        
    def m_step(self, X):
        n_samples, _ = X.shape
        Nk = np.sum(self.resp_, axis=0)
        
        self.means_ = np.dot(self.resp_.T, X) / (Nk[:, np.newaxis] + 1e-1)
        
        for c in range(self.n_components):
            diff = X - self.means_[c]
            self.covariances_[c] = np.dot(self.resp_[:, c] * diff.T, diff) / (Nk[c] + 1e-1)
            self.covariances_[c] += np.eye(self.covariances_[c].shape[0]) * 1e-10  # Ensure positive definiteness

        self.mixing_coefficients_ = Nk / n_samples

    def fit(self, X):
        self.initialize_params(X)
        log_likelihood = []
        
        for _ in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
            log_likelihood.append(self.getLikelihood(X))
            
            if len(log_likelihood) > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < 1e-20:
                break
    
    def getParams(self):
        return {
            'means': self.means_,
            'covariances': self.covariances_,
            'mixing_coefficients': self.mixing_coefficients_
        }
    
    def getMembership(self):
        return self.resp_
    
    def getLikelihood(self, X):
        n_samples, _ = X.shape
        likelihood = np.zeros(n_samples)
        
        for c in range(self.n_components):
            rv = multivariate_normal(self.means_[c], self.covariances_[c])
            likelihood += self.mixing_coefficients_[c] * rv.pdf(X)
        
        return np.sum(np.log(likelihood + 1e-1))
    
    def compute_bic_aic(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.getLikelihood(X)
        n_params = self.n_components * (n_features + n_features * (n_features + 1) / 2) + self.n_components - 1
        
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        aic = -2 * log_likelihood + 2 * n_params
        return bic, aic

def determine_optimal_clusters(X, max_components, n_runs=5):
    bic_scores = []
    aic_scores = []
    
    for n_components in range(1, max_components + 1):
        bic_run = []
        aic_run = []
        
        for _ in range(n_runs):
            gmm = GMM(n_components)
            gmm.fit(X)
            bic, aic = gmm.compute_bic_aic(X)
            bic_run.append(bic)
            aic_run.append(aic)
        
        bic_scores.append(np.mean(bic_run))
        aic_scores.append(np.mean(aic_run))
    
    optimal_bic = np.argmin(bic_scores) + 1
    optimal_aic = np.argmin(aic_scores) + 1
    
    return optimal_bic, optimal_aic, bic_scores, aic_scores
