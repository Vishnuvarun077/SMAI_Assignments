import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
# Hyperparameter tuning
import itertools
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
import time
import matplotlib.animation as animation
# Adding the project directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# # Ensure the figures directory exists
# if not os.path.exists('figures'):
#     os.makedirs('figures')

from models.knn.knn import KNN
from performance_measures.performance_measures import PerformanceMeasures
from models.knn.knn import initial_KNN

def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    # Dropping rows with missing values
    df.dropna(inplace=True)
    return df

def normalize_data(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data

def encode_target(data, target_column):
    if data[target_column].dtype == 'object':
        data[target_column] = data[target_column].astype('category').cat.codes
    return data

def plot_feature_distribution(data, feature, prefix=''):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title('Distribution of {}'.format(feature))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Comments about observations
    skewness = data[feature].skew()
    if abs(skewness) > 1:
        plt.annotate('Skewed (skewness: {:.2f})'.format(skewness),
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     verticalalignment='top')
    # Check for outliers
    outlier_condition = data[feature] > data[feature].quantile(0.75) + 1.5 * (data[feature].quantile(0.75) - data[feature].quantile(0.25))
    if any(outlier_condition):
        plt.annotate('Potential outliers', xy=(0.05, 0.90),
                     xycoords='axes fraction', verticalalignment='top')
    plt.tight_layout()
    plt.savefig('assignments/1/figures/EDA/{}{}_distribution.png'.format(prefix, feature))
    plt.close()

def plot_correlation_heatmap(data, prefix=''):
    numeric_features = data.select_dtypes(include=[np.number])
    corr = numeric_features.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Features')
    plt.savefig('assignments/1/figures/EDA/{}correlation_heatmap.png'.format(prefix))
    plt.close()

def plot_feature_importance(data, target_column, prefix=''):
    numeric_features = data.select_dtypes(include=[np.number])
    correlations = numeric_features.corrwith(data[target_column]).abs().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    correlations.plot(kind='bar')
    plt.title('Feature Importance (Correlation with Target)')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('assignments/1/figures/EDA/{}feature_importance.png'.format(prefix))
    plt.close()

def plot_pairplot(data, target_column, prefix=''):
    sns.pairplot(data=data, hue=target_column)
    plt.savefig('assignments/1/figures/EDA/{}pairplot.png'.format(prefix))
    plt.close()

def remove_outliers_zscore(data, columns, threshold=3):
    cleaned_data = data.copy()
    for col in columns:
        z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
        cleaned_data = cleaned_data[z_scores < threshold]
    return cleaned_data

def exploratory_data_analysis(data, target_column):
    print("Dataset Information:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Encode target column
    data = encode_target(data, target_column)

    # Remove outliers using z-score method
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data_cleaned = remove_outliers_zscore(data, numeric_columns)

    # Plot distributions for each numeric feature before normalization
    for feature in numeric_columns:
        plot_feature_distribution(data_cleaned, feature, prefix='before_')

    plot_correlation_heatmap(data_cleaned, prefix='before_')

    data_cleaned[target_column] = data[target_column]
    plot_feature_importance(data_cleaned, target_column, prefix='before_')

    # Normalize data
    numeric_data = data_cleaned[numeric_columns]
    normalized_data = normalize_data(numeric_data)

    # Plot distributions for each numeric feature after normalization
    for feature in numeric_columns:
        plot_feature_distribution(normalized_data, feature, prefix='after_')

    

    normalized_data[target_column] = data_cleaned[target_column]
    plot_pairplot(normalized_data, target_column,prefix='before_')
    # Removing mode column,key colums from the normalized data if they exist
    coulums_to_drop = ['mode','key','time_signature']
    if any([col in normalized_data.columns for col in coulums_to_drop]):
        normalized_data = normalized_data.drop(columns=coulums_to_drop)
    plot_feature_importance(normalized_data, target_column, prefix='after_')
    plot_correlation_heatmap(normalized_data, prefix='after_')
    plot_pairplot(normalized_data, target_column,prefix='after_')
  
    # Write observations to a file
    with open('assignments/1/edaresults.txt', 'w') as f:
        f.write("# Exploratory Data Analysis Observations\n\n")
        f.write("## Dataset Overview\n")
        f.write("- Number of samples (original): {}\n".format(len(data)))
        f.write("- Number of samples (after outlier removal): {}\n".format(len(data_cleaned)))
        f.write("- Number of features: {}\n".format(len(data.columns) - 1))
        f.write("- Target variable: {}\n\n".format(target_column))
        f.write("## Feature Distributions\n")
        f.write("- See individual distribution plots saved in the 'figures' directory.\n")
        for column in numeric_columns:
            f.write("- {}:\n".format(column))
            f.write("  - Mean: {:.2f}\n".format(data_cleaned[column].mean()))
            f.write("  - Median: {:.2f}\n".format(data_cleaned[column].median()))
            f.write("  - Skewness: {:.2f}\n".format(data_cleaned[column].skew()))
            if abs(data_cleaned[column].skew()) > 1:
                f.write("  - Observation: Skewed distribution\n")
            f.write("\n")
        f.write("## Pairwise Relationships\n")
        f.write("- See 'pairplot.png' for pairwise relationship visualizations.\n")
        f.write("- Observations:\n")
        f.write("  - [Add your observations about pairwise relationships here]\n\n")
        f.write("## Feature Hierarchy for Classification\n")
        f.write("Based on the correlation with the target variable and the feature distributions, here's a proposed hierarchy of feature importance:\n")
        f.write("  - [Add your observations about feature importance here]\n")
    return normalized_data





def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X = X.iloc[indices]
    y = y.iloc[indices]
    
    # Split the data
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    return X_train, X_val, y_train, y_val

def prepare_data(data, target_column):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = data[numeric_columns]
    normalized_data = normalize_data(numeric_data)
    normalized_data[target_column] = data[target_column]
    coulums_to_drop = ['mode','key','time_signature']
    if any([col in normalized_data.columns for col in coulums_to_drop]):
        normalized_data = normalized_data.drop(columns=coulums_to_drop)
    X = normalized_data.drop(columns=[target_column])
    y = normalized_data[target_column]
      # Map string labels to integers if necessary
    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    
    return X, y
def knn_classification(data, target_column, k, distance_metric, test_size, random_state):
    X, y = prepare_data(data, target_column)
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize and fit KNN
    knn = KNN(k=k, distance_metric=distance_metric)
    knn.fit(X_train.values, y_train.values)
    
    # Make predictions
    y_pred = knn.predict(X_val.values)
    
    # Calculate performance metrics
    pm = PerformanceMeasures()
    accuracy = pm.accuracy(y_val, y_pred)
    precision_macro = pm.precision(y_val, y_pred, average='macro')
    recall_macro = pm.recall(y_val, y_pred, average='macro')
    f1_macro = pm.f1_score_macro(y_val, y_pred)
    f1_micro = pm.f1_score_micro(y_val, y_pred)
    
    # Print results
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision (macro): {:.2f}".format(precision_macro))
    print("Recall (macro): {:.2f}".format(recall_macro))
    print("F1 Score (macro): {:.2f}".format(f1_macro))
    print("F1 Score (micro): {:.2f}".format(f1_micro))
    return accuracy, precision_macro, recall_macro, f1_macro, f1_micro


def prepare_data_2(data, target_column):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = data[numeric_columns]
    normalized_data = normalize_data(numeric_data)
    normalized_data[target_column] = data[target_column]
    X = normalized_data.drop(columns=[target_column])
    y = normalized_data[target_column]
      # Map string labels to integers if necessary
    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    
    return X, y

def hyperparameter_tuning(data, target_column):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    X, y = prepare_data_2(data, target_column)
  
    # Split data into 80% training and 20% remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split remaining 20% into 50% validation and 50% test (10% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

    # Consider only odd k values
    k_values = [k for k in range(1, 21) if k % 2 != 0]
    distance_metrics = ['euclidean', 'manhattan', 'cosine']
    results = []

    for k, metric in itertools.product(k_values, distance_metrics):
        knn = KNN(k=k, distance_metric=metric)
        knn.fit(X_train.values, y_train.values)
        y_pred = knn.predict(X_val.values)
        accuracy = PerformanceMeasures().accuracy(y_val.values, y_pred)
        results.append((k, metric, accuracy))

    results.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 10 {k, distance_metric} pairs:")
    for k, metric, acc in results[:10]:
        print("k: {}, Distance Metric: {}, Accuracy: {:.4f}".format(k, metric, acc))

    best_k, best_metric, _ = results[0]
    
    # Plot k vs accuracy for the best distance metric
    k_accuracies = [acc for k, metric, acc in results if metric == best_metric]
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, k_accuracies)
    plt.title('k vs Accuracy (Best Distance Metric: {})'.format(best_metric))
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('assignments/1/figures/k_vs_accuracy.png')
    plt.close()
    
    # Task 4: Drop various columns and check for better accuracy
    best_columns = numeric_columns
    best_accuracy = 0
    for columns_to_drop in itertools.combinations(numeric_columns, len(numeric_columns) - 1):
        columns_to_keep = [col for col in numeric_columns if col not in columns_to_drop]
        X_temp = X[columns_to_keep]
        X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X_temp, y, test_size=0.2, random_state=42)
        knn = KNN(k=best_k, distance_metric=best_metric)
        knn.fit(X_train_temp.values, y_train_temp.values)
        y_pred_temp = knn.predict(X_val_temp.values)
        accuracy_temp = PerformanceMeasures().accuracy(y_val_temp.values, y_pred_temp)
        if accuracy_temp > best_accuracy:
            best_accuracy = accuracy_temp
            best_columns = X_temp.columns.tolist()

    print("Best columns combination: ", best_columns)
    print("Best accuracy with dropped columns: {:.4f}".format(best_accuracy))
    return best_k, best_metric

##Task 5: Feature Selection
def greedy_forward_selection(X, y, best_k, best_metric):
    remaining_features = list(X.columns)
    selected_features = []
    best_accuracy = 0

    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_temp = X[current_features]
            X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X_temp, y, test_size=0.2, random_state=42)
            knn = KNN(k=best_k, distance_metric=best_metric)
            knn.fit(X_train_temp.values, y_train_temp.values)
            y_pred_temp = knn.predict(X_val_temp.values)
            accuracy_temp = PerformanceMeasures().accuracy(y_val_temp.values, y_pred_temp)
            if accuracy_temp > best_accuracy:
                best_accuracy = accuracy_temp
                best_feature = feature

        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return selected_features, best_accuracy

def feature_selection(data, target_column, best_k, best_metric, use_greedy=False):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = data[numeric_columns]
    normalized_data = normalize_data(numeric_data)
    normalized_data[target_column] = data[target_column]
    data = normalized_data
    # coulums_to_drop = ['mode','key','time_signature']
    # if any([col in normalized_data.columns for col in coulums_to_drop]):
    #     normalized_data = normalized_data.drop(columns=coulums_to_drop)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    
    # Greedy Forward Selection
    if use_greedy:
        best_columns, best_accuracy = greedy_forward_selection(X, y, best_k, best_metric)
        print("Best features selected by Greedy Forward Selection: ", best_columns)
        print("Best accuracy with Greedy Forward Selection: {:.4f}".format(best_accuracy))
        return best_columns



def optimization_comparison(data, target_column, best_k, best_metric):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = data[numeric_columns]
    normalized_data = normalize_data(numeric_data)
    normalized_data[target_column] = data[target_column]
    data = normalized_data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        # Split data into 80% training and 20% remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split remaining 20% into 50% validation and 50% test (10% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)
    # Initial KNN model
    initial_knn = initial_KNN(k=19, distance_metric='euclidean')
    initial_knn.fit(X_train.values, y_train.values)

    # Best KNN model
    best_knn = initial_KNN(k=best_k, distance_metric=best_metric)
    best_knn.fit(X_train.values, y_train.values)

    # Optimized KNN model (using numpy vectorization)
    optimized_knn = KNN(k=best_k, distance_metric=best_metric)
    optimized_knn.fit(X_train.values, y_train.values)

    # Sklearn KNN model
    sklearn_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
    sklearn_knn.fit(X_train, y_train)

    models = [
        ("Initial KNN", initial_knn),
        ("Best KNN", best_knn),
        ("Optimized KNN", optimized_knn),
        ("Sklearn KNN", sklearn_knn)
    ]

    # Inference time comparison
    inference_times = []
    for name, model in models:
        start_time = time.time()
        if name == "Sklearn KNN":
            model.predict(X_test)
        else:
            model.predict(X_test.values)
        end_time = time.time()
        inference_times.append((name, end_time - start_time))

    # Plot inference time comparison
    plt.figure(figsize=(10, 6))
    names, times = zip(*inference_times)
    plt.bar(names, times)
    plt.title('Inference Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig('assignments/1/figures/inference_time_comparison.png')
    plt.close()

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(['track_genre'], axis=1)
    y = data['track_genre']
    
    # Convert non-numeric columns to numeric where possible
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that cannot be converted to numeric or are boolean
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')
    
    # Normalize the features
    X = (X - X.min()) / (X.max() - X.min())
    
    if y.dtype == 'object':
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = y.map(label_mapping)
    
    return X, y

def evaluate_knn(X_train, y_train, X_val, y_val, k, distance_metric):
    # Initialize and fit KNN
    knn = KNN(k=k, distance_metric=distance_metric)
    knn.fit(X_train.values, y_train.values)
    
    # Make predictions
    y_pred = knn.predict(X_val.values)
    
    # Calculate performance metrics
    pm = PerformanceMeasures()
    accuracy = pm.accuracy(y_val.values, y_pred)
    precision_macro = pm.precision(y_val.values, y_pred, average='macro')
    recall_macro = pm.recall(y_val.values, y_pred, average='macro')
    f1_macro = pm.f1_score_macro(y_val.values, y_pred)
    f1_micro = pm.f1_score_micro(y_val.values, y_pred)
    
    return accuracy, precision_macro, recall_macro, f1_macro, f1_micro



#######CODE FOR QUESTION 2##################
from models.linearregression.linearregression import LinearRegression
def load_and_split_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    # Split the data into 80% train, 10% validation, and 10% test
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    X_train = train_data.iloc[:, 0].values.reshape(-1, 1)
    y_train = train_data.iloc[:, 1].values
    X_val = val_data.iloc[:, 0].values.reshape(-1, 1)
    y_val = val_data.iloc[:, 1].values
    X_test = test_data.iloc[:, 0].values.reshape(-1, 1)
    y_test = test_data.iloc[:, 1].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def visualize_data(X_train,X_val,X_test,y_train,y_val,y_test,title = 'Data_Splits'):
   #Visulaize the splitted data
   plt.scatter(X_train, y_train, color='blue', label='Train')
   plt.scatter(X_val, y_val, color='green', label='Validation')
   plt.scatter(X_test, y_test, color='red', label='Test')
   plt.xlabel('X')
   plt.ylabel('y')
   plt.legend()
   plt.title('Data Splits')
   # Ensure the directory exists
   output_dir = 'figures/linearregression'
   os.makedirs(output_dir, exist_ok=True)
   # Save the figure
   plt.savefig('assignments/1/figures/linearregression/{}.png'.format(title))
   plt.show()
def simple_regression_degree1(X_train, X_val, X_test, y_train, y_val, y_test):
    # Train the model and select the best learning rate
    # learning_rates = [0.001, 0.01, 0.1, 1]
    learning_rates = np.linspace(0.001, 1, 1000)
    best_lr = None
    best_mse = float('inf')

    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr)
        model.fit(X_train, y_train)
        mse_val = model.mse(X_val, y_val)
        if mse_val < best_mse:
            best_mse = mse_val
            best_lr = lr

    # Train the final model with the best learning rate
    final_model = LinearRegression(learning_rate=best_lr)
    final_model.fit(X_train, y_train)

    # Report metrics
    mse_train = final_model.mse(X_train, y_train)
    std_train = final_model.std_dev(X_train, y_train)
    var_train = final_model.variance(X_train, y_train)

    mse_test = final_model.mse(X_test, y_test)
    std_test = final_model.std_dev(X_test, y_test)
    var_test = final_model.variance(X_test, y_test)
    print("Train MSE: {}, Std Dev: {}, Variance: {}".format(mse_train, std_train, var_train))
    print("Test MSE: {}, Std Dev: {}, Variance: {}".format(mse_test, std_test, var_test))
    print("Best learning rate: {}".format(best_lr))
    # Plot the training points with the fitted line
    plt.scatter(X_train, y_train, color='blue', label='Train')
    plt.plot(X_train, final_model.predict(X_train), color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Training Points with Fitted Line')
    # Ensure the directory exists
    output_dir = 'assignments/1/figures/linearregression'
    os.makedirs(output_dir, exist_ok=True)
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'best_fit_degree_1.png'))
    plt.show()

def simple_regression_higher_degree(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=10):
    results = []
    
    for degree in range(2, max_degree + 1):
        model = LinearRegression(degree=degree)
        model.fit(X_train, y_train)
        
        train_mse = model.mse(X_train, y_train)
        test_mse = model.mse(X_test, y_test)
        train_std = model.std_dev(X_train, y_train)
        test_std = model.std_dev(X_test, y_test)
        train_var = model.variance(X_train, y_train)
        test_var = model.variance(X_test, y_test)
        
        results.append((degree, train_mse, test_mse, train_std, test_std, train_var, test_var))
        print("Degree {} Polynomial Results:".format(degree))
        print("Train MSE: {:.4f}, Std Dev: {:.4f}, Variance: {:.4f}".format(train_mse, train_std, train_var))
        print("Test MSE: {:.4f}, Std Dev: {:.4f}, Variance: {:.4f}".format(test_mse, test_std, test_var))

    best_degree = min(results, key=lambda x: x[2])[0]
    print("Best degree (minimizing test MSE): {}".format(best_degree))
    
    # Save best model coefficients
    best_model = LinearRegression(degree=best_degree)
    best_model.fit(X_train, y_train)
    coefficients = best_model.coefficients
    np.savetxt('assignments/1/best_model_coefficients.txt', coefficients, header='Coefficients')
    # Plot the best degree polynomial fitting the data
    plt.scatter(X_train, y_train, color='blue', label='Train data')
    plt.scatter(X_test, y_test, color='red', label='Test data')
    X_plot = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
    y_plot = best_model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='green', label='Best fit (degree={})'.format(best_degree))
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial Regression (degree={})'.format(best_degree))
    # Ensure the directory exists
    output_dir = 'assignments/1/figures/linearregression'
    os.makedirs(output_dir, exist_ok=True)
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'best_fit_higher_degree.png'))
    plt.show()
    return best_degree





def create_animation(X_train, y_train, degree, learning_rate, iterations, filename):
    model = LinearRegression(degree=degree, learning_rate=learning_rate, iterations=iterations)
    X_poly = model.add_polynomial_features_for_animation(X_train)
    m, n = X_poly.shape
    coefficients = np.zeros(n)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Degree {} Polynomial Fit Animation'.format(degree))

    def animate(i):
        nonlocal coefficients
        y_pred = X_poly.dot(coefficients)
        error = y_pred - y_train
        gradient = X_poly.T.dot(error) / m
        coefficients -= learning_rate * gradient

        mse = np.mean(error ** 2)
        std_dev = np.std(error)
        variance = np.var(error)

        model.mse_history.append(mse)
        model.std_dev_history.append(std_dev)
        model.variance_history.append(variance)

        for ax in axs.flat:
            ax.clear()

        axs[0, 0].scatter(X_train, y_train, color='blue', label='Data')
        axs[0, 0].plot(X_train, y_pred, color='red', label='Fit')
        axs[0, 0].set_title('Data and Fit')
        axs[0, 0].legend()

        axs[0, 1].plot(range(i+1), model.mse_history[:i+1])
        axs[0, 1].set_title('MSE')

        axs[1, 0].plot(range(i+1), model.std_dev_history[:i+1])
        axs[1, 0].set_title('Standard Deviation')

        axs[1, 1].plot(range(i+1), model.variance_history[:i+1])
        axs[1, 1].set_title('Variance')

    anim = animation.FuncAnimation(fig, animate, frames=iterations, interval=50, repeat=False)
    anim.save('assignments/1/figures/linearregression/animations/{}.gif'.format(filename), writer='pillow')
    plt.close()


def regularization_tasks(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=20):
 # Create directory for figures if it doesn't exist
    dirs = [
        'figures/linearregression/no_reg',
        'figures/linearregression/reg_l1',
        'figures/linearregression/reg_l2'
    ]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Open a text file to write results
    with open('results.txt', 'w') as f:
        # Without regularization
        for degree in range(1, max_degree + 1):
            model = LinearRegression(degree=degree, learning_rate=0.01, iterations=1000)
            model.fit(X_train, y_train)

            plt.figure(figsize=(10, 6))
            plt.scatter(X_train, y_train, color='blue', label='Train')
            plt.plot(X_train, model.predict(X_train), color='red', label='Degree {}'.format(degree))
            plt.legend()
            plt.title('Degree {} Polynomial Fit (No Regularization)'.format(degree))
            plt.xlabel('X')
            plt.ylabel('y')

            # Ensure the directory exists
            output_dir = 'figures/linearregression/no_reg'
            os.makedirs(output_dir, exist_ok=True)

            save_path = 'assignments/1/figures/linearregression/no_reg/no_reg_degree_{}.png'.format((degree))
            plt.savefig(save_path)
            plt.close()
            # print("Saved plot: {}".format(save_path))

            f.write("Degree {} (No Regularization):\n".format(degree))
            f.write("Train MSE: {:.4f}\n".format(model.mse(X_train, y_train)))
            f.write("Test MSE: {:.4f}\n\n".format(model.mse(X_test, y_test)))

        # With L1 and L2 regularization
        for reg_type in ['L1', 'L2']:
            for degree in range(1, max_degree + 1):
                model = LinearRegression(degree=degree, learning_rate=0.01, iterations=1000, reg_type=reg_type, lambda_=0.1)
                model.fit(X_train, y_train)

                plt.figure(figsize=(10, 6))
                plt.scatter(X_train, y_train, color='blue', label='Train')
                plt.plot(X_train, model.predict(X_train), color='red', label='Degree {}'.format(degree))
                plt.legend()
                plt.title('Degree {} Polynomial Fit ({} Regularization)'.format(degree, reg_type))
                plt.xlabel('X')
                plt.ylabel('y')
                  # Ensure the directory exists
                output_dir = 'assignments/1/figures/linearregression/{}_reg'.format(reg_type.lower())
                os.makedirs(output_dir, exist_ok=True)

                save_path = '{}/{}_reg_degree_{}.png'.format(output_dir,reg_type.lower(), degree)
                plt.savefig(save_path)
                plt.close()
                # print("Saved plot: {}".format(save_path))

                f.write("Degree {} ({} Regularization):\n".format(degree, reg_type))
                f.write("Train MSE: {:.4f}\n".format(model.mse(X_train, y_train)))
                f.write("Test MSE: {:.4f}\n\n".format(model.mse(X_test, y_test)))



# def Implementall_1():
#      # Load data
#     data = load_data('data/external/spotify.csv')
#     # Perform exploratory data analysis
#     target_column = 'track_genre'
#     print("\nPerforming exploratory data analysis...\n")
#     data_cleaned= exploratory_data_analysis(data, target_column)
#     print("\nExploratory data analysis completed.\n")
#     # Perform KNN classification
#     print("\nPerforming KNN base case...\n")
#     knn_classification(data_cleaned, target_column, k=3, distance_metric='euclidean', test_size=0.2, random_state=42)
#     print("\nKNN base case completed.\n")
#     #Perform hyperparameter tuning
#     print("\nPerforming hyperparameter tuning...\n")
#     best_k, best_metric = hyperparameter_tuning(data_cleaned, target_column)
#     print("\nHyperparameter tuning completed.\n")
#     # best_k = 19
#     # best_metric = 'manhattan'
#     print("\nPerforming feature selection...\n")
#     feature_selection(data_cleaned, target_column, best_k, best_metric,use_greedy=True)
#     print("\nFeature selection completed.\n")
#     print("\nPerforming optimization comparison...\n")
#     optimization_comparison(data_cleaned, target_column, best_k, best_metric)
#     print("\nOptimization comparison completed.\n")
#     print("\nPerforming second data set analysis...\n")
#     test = load_data('data/external/spotify_2/test.csv')
#     train = load_data('data/external/spotify_2/train.csv')
#     validate = load_data('data/external/spotify_2/validate.csv')
#     # Load and preprocess the data
#     X_train, y_train = load_and_preprocess_data('data/external/spotify_2/train.csv')
#     X_val, y_val = load_and_preprocess_data('data/external/spotify_2/validate.csv')
#     X_test, y_test = load_and_preprocess_data('data/external/spotify_2/test.csv')
#     # # Use the best {k, distance_metric} pair found earlier
#     best_k = 19
#     best_distance_metric = 'manhattan'
#     # Evaluate on validation set
#     val_accuracy, val_precision, val_recall, val_f1_macro, val_f1_micro = evaluate_knn(
#         X_train, y_train, X_val, y_val, best_k, best_distance_metric
#     )
#     # Evaluate on test set
#     test_accuracy, test_precision, test_recall, test_f1_macro, test_f1_micro = evaluate_knn(
#         X_train, y_train, X_test, y_test, best_k, best_distance_metric
#     )
#     # Print results
#     print("\nValidation Set Results:")
#     print("Validation Accuracy: {:.2f}".format(val_accuracy))
#     print("Validation Precision (macro): {:.2f}".format(val_precision))
#     print("Validation Recall (macro): {:.2f}".format(val_recall))
#     print("Validation F1 Score (macro): {:.2f}".format(val_f1_macro))
#     print("Validation F1 Score (micro): {:.2f}".format(val_f1_micro))
    
#     print("\nTest Set Results:")
#     print("Test Accuracy: {:.2f}".format(test_accuracy))
#     print("Test Precision (macro): {:.2f}".format(test_precision))
#     print("Test Recall (macro): {:.2f}".format(test_recall))
#     print("Test F1 Score (macro): {:.2f}".format(test_f1_macro))
#     print("Test F1 Score (micro): {:.2f}".format(test_f1_micro)
#     )
#     print("\nSecond data set analysis completed.\n")

# def Implementall_2():
#     # Load and split data
#     X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/linreg.csv')
#     print("\nPerforming simpleregrerssion with degree 1...\n")
#     # Visualize the data splits
#     visualize_data(X_train, X_val, X_test, y_train, y_val, y_test,title = 'Data_Splits_1')
#     # Simple Regression with degree <1
#     simple_regression_degree1(X_train, X_val, X_test, y_train, y_val, y_test)
#     print("\nSimple regression with degree 1 completed.\n")
#     print("\nPerforming simple regression with degree greater than 1...\n")
#     # Simple Regression with higher degree
#     best_degree = simple_regression_higher_degree(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=10)
#     print("\nSimple regression with degree greater than 1 completed.\n")
#     print("\nCreating animations...\n")
#     # Create animations for degrees 1 to 5
#     for degree in range(1, 10):
#         create_animation(X_train, y_train, degree, 0.01, 100, 'animation_degree_{}'.format(degree))
#     print("\nAnimations created.\n")
#     print("\nPerforming regularization tasks...\n")
#     # Regularization
#     X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/regularisation.csv')
#     visualize_data(X_train, X_val, X_test, y_train, y_val, y_test,title = 'Data_Splits_2')
#     regularization_tasks(X_train, X_val, X_test, y_train, y_val, y_test)
#     print("\nRegularization tasks completed.\n")

# def knn_tasks():
#     print("KNN Tasks:")
#     print("1. Exploratory Data Analysis")
#     print("2. KNN Base Case")
#     print("3. Hyperparameter Tuning (without feature selection)")
#     print("4. Feature Selection")
#     print("5. Optimization")
#     print("6. Second Data Set")
#     print("7. Implement all")
    
#     choice = input("Enter your choice (1-7): ")
    
#     data = load_data('data/external/spotify.csv')
#     target_column = 'track_genre'
    
#     if choice == '1':
#         print("\nPerforming exploratory data analysis...\n")
#         exploratory_data_analysis(data, target_column)
#         print("\nExploratory data analysis completed.\n")
#     elif choice == '2':
#         print("\nPerforming KNN base case...\n")
#         knn_classification(data, target_column, k=3, distance_metric='euclidean', test_size=0.6, random_state=42)
#         print("\nKNN base case completed.\n")
#     elif choice == '3':
#         print("\nPerforming hyperparameter tuning...\n")
#         best_k, best_metric = hyperparameter_tuning(data, target_column)
#         print("\nHyperparameter tuning completed.\n")
#     elif choice == '4':
#         print("\nPerforming feature selection...\n")
#         best_k, best_metric = 19, 'manhattan'  # Assuming these are the best values
#         feature_selection(data, target_column, best_k, best_metric, use_greedy=True)
#         print("\nFeature selection completed.\n")
#     elif choice == '5':
#         print("\nPerforming optimization comparison...\n")
#         best_k, best_metric = 19, 'manhattan'  # Assuming these are the best values
#         optimization_comparison(data, target_column, best_k, best_metric)
#         print("Optimization comparison completed.")
#     elif choice == '6':
#         print("\nPerforming tasks on the second data set...\n")
#         # Second data set tasks
#         X_train, y_train = load_and_preprocess_data('data/external/spotify_2/train.csv')
#         X_val, y_val = load_and_preprocess_data('data/external/spotify_2/validate.csv')
#         X_test, y_test = load_and_preprocess_data('data/external/spotify_2/test.csv')
#         # Use the best {k, distance_metric} pair found earlier
#         best_k, best_distance_metric = 19, 'manhattan'
        
#         val_accuracy, val_precision, val_recall, val_f1_macro, val_f1_micro = evaluate_knn(
#             X_train, y_train, X_val, y_val, best_k, best_distance_metric
#         )
        
#         test_accuracy, test_precision, test_recall, test_f1_macro, test_f1_micro = evaluate_knn(
#             X_train, y_train, X_test, y_test, best_k, best_distance_metric
#         )
        
#         print("\nValidation Set Results:")
#         print("Validation Accuracy: {:.2f}".format(val_accuracy))
#         print("Validation Precision (macro): {:.2f}".format(val_precision))
#         print("Validation Recall (macro): {:.2f}".format(val_recall))
#         print("Validation F1 Score (macro): {:.2f}".format(val_f1_macro))
#         print("Validation F1 Score (micro): {:.2f}".format(val_f1_micro))
        
#         print("\nTest Set Results:")
#         print("Test Accuracy: {:.2f}".format(test_accuracy))
#         print("Test Precision (macro): {:.2f}".format(test_precision))
#         print("Test Recall (macro): {:.2f}".format(test_recall))
#         print("Test F1 Score (macro): {:.2f}".format(test_f1_macro))
#         print("Test F1 Score (micro): {:.2f}".format(test_f1_micro))
        
#         print("\nSecond data set tasks completed.\n")
#     elif choice == '7':
#         Implementall_1()
#     else:
#         print("Invalid choice. Please enter a number between 1 and 7.")

# def linear_regression_tasks():
#     print("Linear Regression Tasks:")
#     print("1. Simple Regression with Degree 1")
#     print("2. Simple Regression with Degree Greater than 1")
#     print("3. Animation")
#     print("4. Regularization")
#     print("5. Implement all")
#     choice = input("Enter your choice (1-5): ")
    
#     X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/linreg.csv')
    
#     if choice == '1':
#         print("\nPerforming simple regression with degree 1...\n")
#         simple_regression_degree1(X_train, X_val, X_test, y_train, y_val, y_test)
#         print("\nSimple regression with degree 1 completed.\n")
#     elif choice == '2':
#         print("\nPerforming simple regression with degree greater than 1...\n")
#         best_degree = simple_regression_higher_degree(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=10)
#         print("\nSimple regression with degree greater than 1 completed.\n")
#     elif choice == '3':
#         print("\nCreating animations...\n")
#         for degree in range(1, 10):
#             create_animation(X_train, y_train, degree, 0.01, 100, 'animation_degree_{}'.format(degree))
#         print("\nAnimations created.\n")
#     elif choice == '4':
#         print("\nPerforming regularization tasks...\n")
#         X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/regularisation.csv')
#         visualize_data(X_train, X_val, X_test, y_train, y_val, y_test)
#         regularization_tasks(X_train, X_val, X_test, y_train, y_val, y_test)
#         print("\nRegularization tasks completed.\n")
#     elif choice == '5':
#         Implementall_2()
#     else:
#         print("Invalid choice. Please enter a number between 1 and 5.")

# def main():
#     print("Welcome to the Assignment 1 Solutions!")
#     print("1. KNN Tasks")
#     print("2. Linear Regression Tasks")
    
#     choice = input("Enter your choice (1 or 2): ")
    
#     if choice == '1':
#         knn_tasks()
#     elif choice == '2':
#         linear_regression_tasks()
#     else:
#         print("Invalid choice. Please enter 1 or 2.")

# if __name__ == "__main__":
#     main()

def Implementall_1():
    # Load data
    data = load_data('data/external/spotify.csv')
    # Perform exploratory data analysis
    target_column = 'track_genre'
    print("\nPerforming exploratory data analysis...\n")
    data_cleaned = exploratory_data_analysis(data, target_column)
    print("\nExploratory data analysis completed.\n")
    # Perform KNN classification
    print("\nPerforming KNN base case...\n")
    knn_classification(data_cleaned, target_column, k=3, distance_metric='euclidean', test_size=0.2, random_state=42)
    print("\nKNN base case completed.\n")
    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning...\n")
    best_k, best_metric = hyperparameter_tuning(data_cleaned, target_column)
    print("\nHyperparameter tuning completed.\n")
    # best_k = 19
    # best_metric = 'manhattan'
    print("\nPerforming feature selection...\n")
    feature_selection(data_cleaned, target_column, best_k, best_metric, use_greedy=True)
    print("\nFeature selection completed.\n")
    print("\nPerforming optimization comparison...\n")
    optimization_comparison(data_cleaned, target_column, best_k, best_metric)
    print("\nOptimization comparison completed.\n")
    print("\nPerforming second data set analysis...\n")
    test = load_data('data/external/spotify_2/test.csv')
    train = load_data('data/external/spotify_2/train.csv')
    validate = load_data('data/external/spotify_2/validate.csv')
    # Load and preprocess the data
    X_train, y_train = load_and_preprocess_data('data/external/spotify_2/train.csv')
    X_val, y_val = load_and_preprocess_data('data/external/spotify_2/validate.csv')
    X_test, y_test = load_and_preprocess_data('data/external/spotify_2/test.csv')
    # # Use the best {k, distance_metric} pair found earlier
    best_k = 19
    best_distance_metric = 'manhattan'
    # Evaluate on validation set
    val_accuracy, val_precision, val_recall, val_f1_macro, val_f1_micro = evaluate_knn(
        X_train, y_train, X_val, y_val, best_k, best_distance_metric
    )
    # Evaluate on test set
    test_accuracy, test_precision, test_recall, test_f1_macro, test_f1_micro = evaluate_knn(
        X_train, y_train, X_test, y_test, best_k, best_distance_metric
    )
    # Print results
    print("\nValidation Set Results:")
    print("Validation Accuracy: {:.2f}".format(val_accuracy))
    print("Validation Precision (macro): {:.2f}".format(val_precision))
    print("Validation Recall (macro): {:.2f}".format(val_recall))
    print("Validation F1 Score (macro): {:.2f}".format(val_f1_macro))
    print("Validation F1 Score (micro): {:.2f}".format(val_f1_micro))
    
    print("\nTest Set Results:")
    print("Test Accuracy: {:.2f}".format(test_accuracy))
    print("Test Precision (macro): {:.2f}".format(test_precision))
    print("Test Recall (macro): {:.2f}".format(test_recall))
    print("Test F1 Score (macro): {:.2f}".format(test_f1_macro))
    print("Test F1 Score (micro): {:.2f}".format(test_f1_micro))
    print("\nSecond data set analysis completed.\n")

def Implementall_2():
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/linreg.csv')
    print("\nPerforming simpleregrerssion with degree 1...\n")
    # Visualize the data splits
    visualize_data(X_train, X_val, X_test, y_train, y_val, y_test, title='Data_Splits_1')
    # Simple Regression with degree <1
    simple_regression_degree1(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\nSimple regression with degree 1 completed.\n")
    print("\nPerforming simple regression with degree greater than 1...\n")
    # Simple Regression with higher degree
    best_degree = simple_regression_higher_degree(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=10)
    print("\nSimple regression with degree greater than 1 completed.\n")
    print("\nCreating animations...\n")
    # Create animations for degrees 1 to 5
    for degree in range(1, 10):
        create_animation(X_train, y_train, degree, 0.01, 100, 'animation_degree_{}'.format(degree))
    print("\nAnimations created.\n")
    print("\nPerforming regularization tasks...\n")
    # Regularization
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/regularisation.csv')
    visualize_data(X_train, X_val, X_test, y_train, y_val, y_test, title='Data_Splits_2')
    regularization_tasks(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\nRegularization tasks completed.\n")

def knn_tasks():
    while True:
        print("KNN Tasks:")
        print("1. Exploratory Data Analysis")
        print("2. KNN Base Case")
        print("3. Hyperparameter Tuning (without feature selection)")
        print("4. Feature Selection")
        print("5. Optimization")
        print("6. Second Data Set")
        print("7. Implement all")
        print("8. Return to Main Menu")
        print("9. Exit")

        choice = input("Enter your choice (1-9): ")

        data = load_data('data/external/spotify.csv')
        target_column = 'track_genre'

        if choice == '1':
            print("\nPerforming exploratory data analysis...\n")
            exploratory_data_analysis(data, target_column)
            print("\nExploratory data analysis completed.\n")
        elif choice == '2':
            print("\nPerforming KNN base case...\n")
            knn_classification(data, target_column, k=19, distance_metric='manhattan', test_size=0.2, random_state=42)
            print("\nKNN base case completed.\n")
        elif choice == '3':
            print("\nPerforming hyperparameter tuning...\n")
            best_k, best_metric = hyperparameter_tuning(data, target_column)
            print("\nHyperparameter tuning completed.\n")
        elif choice == '4':
            print("\nPerforming feature selection...\n")
            best_k, best_metric = 19, 'manhattan'  # Assuming these are the best values
            feature_selection(data, target_column, best_k, best_metric, use_greedy=True)
            print("\nFeature selection completed.\n")
        elif choice == '5':
            print("\nPerforming optimization comparison...\n")
            best_k, best_metric = 19, 'manhattan'  # Assuming these are the best values
            optimization_comparison(data, target_column, best_k, best_metric)
            print("Optimization comparison completed.")
        elif choice == '6':
            print("\nPerforming tasks on the second data set...\n")
            # Second data set tasks
            X_train, y_train = load_and_preprocess_data('data/external/spotify_2/train.csv')
            X_val, y_val = load_and_preprocess_data('data/external/spotify_2/validate.csv')
            X_test, y_test = load_and_preprocess_data('data/external/spotify_2/test.csv')
            # Use the best {k, distance_metric} pair found earlier
            best_k, best_distance_metric = 19, 'manhattan'

            val_accuracy, val_precision, val_recall, val_f1_macro, val_f1_micro = evaluate_knn(
                X_train, y_train, X_val, y_val, best_k, best_distance_metric
            )

            test_accuracy, test_precision, test_recall, test_f1_macro, test_f1_micro = evaluate_knn(
                X_train, y_train, X_test, y_test, best_k, best_distance_metric
            )

            print("\nValidation Set Results:")
            print("Validation Accuracy: {:.2f}".format(val_accuracy))
            print("Validation Precision (macro): {:.2f}".format(val_precision))
            print("Validation Recall (macro): {:.2f}".format(val_recall))
            print("Validation F1 Score (macro): {:.2f}".format(val_f1_macro))
            print("Validation F1 Score (micro): {:.2f}".format(val_f1_micro))

            print("\nTest Set Results:")
            print("Test Accuracy: {:.2f}".format(test_accuracy))
            print("Test Precision (macro): {:.2f}".format(test_precision))
            print("Test Recall (macro): {:.2f}".format(test_recall))
            print("Test F1 Score (macro): {:.2f}".format(test_f1_macro))
            print("Test F1 Score (micro): {:.2f}".format(test_f1_micro))

            print("\nSecond data set tasks completed.\n")
        elif choice == '7':
            Implementall_1()
        elif choice == '8':
            return
        elif choice == '9':
            exit()
        else:
            print("Invalid choice. Please enter a number between 1 and 9.")

def linear_regression_tasks():
    while True:
        print("Linear Regression Tasks:")
        print("1. Simple Regression with Degree 1")
        print("2. Simple Regression with Degree Greater than 1")
        print("3. Animation")
        print("4. Regularization")
        print("5. Implement all")
        print("6. Return to Main Menu")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/linreg.csv')

        if choice == '1':
            print("\nPerforming simple regression with degree 1...\n")
            simple_regression_degree1(X_train, X_val, X_test, y_train, y_val, y_test)
            print("\nSimple regression with degree 1 completed.\n")
        elif choice == '2':
            print("\nPerforming simple regression with degree greater than 1...\n")
            best_degree = simple_regression_higher_degree(X_train, X_val, X_test, y_train, y_val, y_test, max_degree=10)
            print("\nSimple regression with degree greater than 1 completed.\n")
        elif choice == '3':
            print("\nCreating animations...\n")
            for degree in range(1, 10):
                create_animation(X_train, y_train, degree, 0.01, 100, 'animation_degree_{}'.format(degree))
            print("\nAnimations created.\n")
        elif choice == '4':
            print("\nPerforming regularization tasks...\n")
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/external/regularisation.csv')
            visualize_data(X_train, X_val, X_test, y_train, y_val, y_test)
            regularization_tasks(X_train, X_val, X_test, y_train, y_val, y_test)
            print("\nRegularization tasks completed.\n")
        elif choice == '5':
            Implementall_2()
        elif choice == '6':
            return
        elif choice == '7':
            exit()
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

def main():
    while True:
        print("Welcome to the Assignment 1 Solutions!")
        print("1. KNN Tasks")
        print("2. Linear Regression Tasks")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            knn_tasks()
        elif choice == '2':
            linear_regression_tasks()
        elif choice == '3':
            exit()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

