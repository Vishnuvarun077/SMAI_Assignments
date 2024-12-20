import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from models.PCA.PCA import PCA
from models.knn.knn import KNN
from performance_measures.performance_measures import PerformanceMeasures

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
def main():
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
    plt.show()

if __name__ == "__main__":
    main()