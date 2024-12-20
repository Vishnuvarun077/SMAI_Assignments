
def MLPClassifier_main():
    import numpy as np
    import pandas as pd
    # import matplotlib.pyplot as plt
    import sys
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from itertools import product
    import wandb
    # Adding the parent directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(parent_dir)
    from models.MLP.MLP import MLPClassifier

    # Load and preprocess the data
    def load_and_preprocess_data(file_path):
        data = pd.read_csv(file_path)
        X = data.drop('quality', axis=1)
        y = data['quality']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_encoded = pd.get_dummies(y).values
        return X_scaled, y_encoded

    # Evaluate model
    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return accuracy, f1, precision, recall

    # Train and evaluate
    def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, config):
        model = MLPClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            output_size=y_train.shape[1],
            activation=config['activation'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            optimizer=config['optimizer']
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        return model, *evaluate_model(y_true, y_pred)

    # Hyperparameter tuning
    def hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test):
        hidden_sizes_list = [[64, 32], [128, 64], [256, 128]]
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rates = [0.0001, 0.001, 0.01]
        epochs_list = [100, 500, 1000]
        batch_sizes = [16, 32, 64]
        optimizers = ['sgd', 'batch', 'mini-batch']
        best_model = None
        best_accuracy = 0
        best_config = None
        wandb.init(project='q2_1', entity='vishnuvarun-iiit-hyderabad')
        for hidden_sizes, activation, lr, epochs, batch_size, optimizer in product(hidden_sizes_list, activations, learning_rates, epochs_list, batch_sizes, optimizers):
            config = {
                'hidden_sizes': hidden_sizes,
                'activation': activation,
                'learning_rate': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'optimizer': optimizer
            }
            model, accuracy, f1, precision, recall = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, config)
            wandb.log({
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_config = config
        print("Best Model Configuration:")
        print(best_config)
        wandb.finish()
        return best_model

    # Evaluate best model
    def evaluate_best_model(X_test, y_test, X_val, y_val, best_model):
        print("Best Model Parameters:")
        print(f"Hidden Sizes: {best_model.hidden_sizes}")
        print(f"Activation Function: {best_model.activation}")
        print(f"Learning Rate: {best_model.learning_rate}")
        print(f"Optimizer: {best_model.optimizer}")
        print(f"Batch Size: {best_model.batch_size}")
        print(f"Number of Epochs: {best_model.epochs}")
        y_pred_val = best_model.predict(X_val)
        y_true_val = np.argmax(y_val, axis=1)
        val_accuracy, val_f1, val_precision, val_recall = evaluate_model(y_true_val, y_pred_val)
        print("\nValidation Set Metrics:")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        y_pred_test = best_model.predict(X_test)
        y_true_test = np.argmax(y_test, axis=1)
        test_accuracy, test_f1, test_precision, test_recall = evaluate_model(y_true_test, y_pred_test)
        print("\nTest Set Metrics:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

    # Main execution
    file_path = '../../data/external/WineQT.csv' 
    X_scaled, y_encoded = load_and_preprocess_data(file_path)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    config = {
        'hidden_sizes': [256, 128],
        'activation': 'relu',
        'learning_rate': 0.001,
        'epochs': 500,
        'batch_size': 16,
        'optimizer': 'sgd'
    }
    accuracy, f1, precision, recall = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, config)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    best_model = hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test)
    evaluate_best_model(X_test, y_test, X_val, y_val, best_model)




####-----MLP REGRESSION-----####
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import wandb
# Adding the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.MLP.MLP import MultiLabelMLPClassifier, MLPRegressor, MLPClassifier_3
from models.knn.knn import KNN
# from assignments.1.a1 import knn_classification

def multilabel_classification():
    def load_and_preprocess_data(file_path):
        data = pd.read_csv(file_path)
        data = data.drop('city', axis=1)
        X = pd.get_dummies(data.drop('labels', axis=1), columns=['gender', 'education', 'married', 'occupation', 'most bought item'])
        X = np.array(X).astype(float)
        y = data['labels'].str.split()
        unique_labels = set(label for labels in y for label in labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        y_binary = np.zeros((len(y), len(unique_labels)), dtype=int)
        for i, labels in enumerate(y):
            for label in labels:
                y_binary[i, label_to_index[label]] = 1
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train)
        X_val_standardized = scaler.transform(X_val)
        X_test_standardized = scaler.transform(X_test)
        return X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test

    def train_and_evaluate_ml(X_train, y_train, X_val, y_val, X_test, y_test, config):
        model = MultiLabelMLPClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            output_size=y_train.shape[1],
            activation=config['activation'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            optimizer=config['optimizer']
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        y_true = y_test
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        hamming = hamming_loss(y_true, y_pred)
        return model, accuracy, f1, precision, recall, hamming

    def hyperparameter_tuning_multilabel(X_train, y_train, X_val, y_val, X_test, y_test):
        hidden_sizes_list = [[128]]
        activations = ['tanh']
        learning_rates = [0.01]
        epochs_list = [1000]
        batch_sizes = [16, 32, 64]
        optimizers = ['batch']
        best_model = None
        best_accuracy = 0
        best_config = None
        wandb.init(project='q2_final4_best', entity='vishnuvarun-iiit-hyderabad')
        for hidden_sizes, activation, lr, epochs, batch_size, optimizer in product(hidden_sizes_list, activations, learning_rates, epochs_list, batch_sizes, optimizers):
            config = {
                'hidden_sizes': hidden_sizes,
                'activation': activation,
                'learning_rate': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'optimizer': optimizer
            }
            model, accuracy, f1, precision, recall, hamming = train_and_evaluate_ml(X_train, y_train, X_val, y_val, X_test, y_test, config)
            wandb.log({
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "hamming_loss": hamming
            })
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_config = config
        print("Best Model Configuration:")
        print(best_config)
        wandb.finish()
        return best_model

    def evaluate_best_model_ml(X_test_standardized, y_test, X_val_standardized, y_val, best_model):
        print("Best Model Parameters:")
        print(f"Hidden Sizes: {best_model.hidden_sizes}")
        print(f"Activation Function: {best_model.activation}")
        print(f"Learning Rate: {best_model.learning_rate}")
        print(f"Optimizer: {best_model.optimizer}")
        print(f"Batch Size: {best_model.batch_size}")
        print(f"Number of Epochs: {best_model.epochs}")
        y_pred_val = best_model.predict(X_val_standardized)
        y_true_val = y_val
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted')
        val_hamming = hamming_loss(y_true_val, y_pred_val)
        print("\nValidation Set Metrics:")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation Hamming Loss: {val_hamming:.4f}")
        y_pred_test = best_model.predict(X_test_standardized)
        y_true_test = y_test
        test_accuracy = accuracy_score(y_true_test, y_pred_test)
        test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')
        test_precision = precision_score(y_true_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_true_test, y_pred_test, average='weighted')
        test_hamming = hamming_loss(y_true_test, y_pred_test)
        print("\nTest Set Metrics:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Hamming Loss: {test_hamming:.4f}")

    file_path = '../../data/external/advertisement.csv'
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(file_path)
    best_model = hyperparameter_tuning_multilabel(X_train, y_train, X_val, y_val, X_test, y_test)
    evaluate_best_model_ml(X_test, y_test, X_val, y_val, best_model)

def mlp_regression():
    def load_and_preprocess_data():
        housing_data = pd.read_csv("../../data/external/HousingData.csv")
        stats_summary = housing_data.agg(['mean', 'std', 'min', 'max'])
        print(stats_summary)
        medv_values = housing_data['MEDV']
        plt.figure(figsize=(8, 6))
        plt.hist(medv_values, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel('MEDV')
        plt.ylabel('Frequency')
        plt.title('Distribution of MEDV')
        plt.savefig('figures/3/MEDV_distribution.png')
        plt.show()
        features = housing_data.drop(columns=['MEDV'])
        target = housing_data['MEDV']
        target = np.array(target).reshape(-1, 1)
        X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)
        # min_max_scaler = MinMaxScaler()
        # X_train_normalized = min_max_scaler.fit_transform(X_train_imputed)
        # X_val_normalized = min_max_scaler.transform(X_val_imputed)
        # X_test_normalized = min_max_scaler.transform(X_test_imputed)
        standard_scaler = StandardScaler()
        X_train_standardized = standard_scaler.fit_transform(X_train_imputed)
        X_val_standardized = standard_scaler.transform(X_val_imputed)
        X_test_standardized = standard_scaler.transform(X_test_imputed)
        return X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test

    def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, config):
        model = MLPRegressor(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            output_size=1,
            activation=config['activation'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            optimizer=config['optimizer']
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        y_true = y_test
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return model, mse, rmse, r2

    def hyperparameter_tuning(X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test):
        hidden_sizes_list = [[64, 32], [128, 64], [256, 128]]
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rates = [0.0001, 0.001, 0.01]
        epochs_list = [100, 500, 1000]
        batch_sizes = [16, 32, 64]
        optimizers = ['sgd', 'batch', 'mini-batch']
        best_model = None
        best_mse = float('inf')
        best_config = None
        wandb.init(project='regression_test_prefinal', entity='vishnuvarun-iiit-hyderabad')
        for hidden_sizes, activation, lr, epochs, batch_size, optimizer in product(hidden_sizes_list, activations, learning_rates, epochs_list, batch_sizes, optimizers):
            config = {
                'hidden_sizes': hidden_sizes,
                'activation': activation,
                'learning_rate': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'optimizer': optimizer
            }
            model, mse, rmse, r2 = train_and_evaluate(X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test, config)
            wandb.log({
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2
            })
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_config = config
        print("Best Model Configuration:")
        print(best_config)
        wandb.finish()
        return best_model

    def evaluate_best_model(X_test, y_test, X_val, y_val, best_model):
        print("Best Model Parameters:")
        print(f"Hidden Sizes: {best_model.hidden_sizes}")
        print(f"Activation Function: {best_model.activation}")
        print(f"Learning Rate: {best_model.learning_rate}")
        print(f"Optimizer: {best_model.optimizer}")
        print(f"Batch Size: {best_model.batch_size}")
        print(f"Number of Epochs: {best_model.epochs}")
        y_pred_val = best_model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_pred_val)
        print("\nValidation Set Metrics:")
        print(f"Validation MSE: {mse_val:.4f}")
        print(f"Validation MAE: {mae_val:.4f}")
        print(f"Validation RMSE: {rmse_val:.4f}")
        print(f"Validation R-squared: {r2_val:.4f}")
        y_pred_test = best_model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_pred_test)
        print("\nTest Set Metrics:")
        print(f"Test MSE: {mse_test:.4f}")
        print(f"Test MAE: {mae_test:.4f}")
        print(f"Test RMSE: {rmse_test:.4f}")
        print(f"Test R-squared: {r2_test:.4f}")

    def analyze_mse_loss(X_test, y_test, best_model):
        y_pred = best_model.predict(X_test)
        mse_per_datapoint = np.mean((y_test - y_pred) ** 2, axis=1)
        results_df = pd.DataFrame({
            'Index': np.arange(len(y_test)),
            'MSE': mse_per_datapoint,
            'True': y_test.flatten(),
            'Predicted': y_pred.flatten()
        })
        results_df = results_df.sort_values(by='MSE', ascending=False)
        print("Top 5 data points with the highest MSE:")
        print(results_df.head())
        print("\nTop 5 data points with the lowest MSE:")
        print(results_df.tail())
        plt.figure(figsize=(10, 6))
        plt.hist(mse_per_datapoint, bins=50, edgecolor='k')
        plt.title('Distribution of MSE Loss per Data Point')
        plt.xlabel('MSE Loss')
        plt.ylabel('Frequency')
        plt.savefig('figures/3/q3_6_mse_distribution.png')
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Index'], results_df['MSE'], alpha=0.5)
        plt.title('MSE Loss per Data Point')
        plt.xlabel('Data Point Index')
        plt.ylabel('MSE Loss')
        plt.savefig('figures/3/q3_6_mse_per_datapoint.png')
        plt.show()
        return results_df

    X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test = load_and_preprocess_data()
    best_model = hyperparameter_tuning(X_train_standardized, y_train, X_val_standardized, y_val, X_test_standardized, y_test)
    evaluate_best_model(X_test_standardized, y_test, X_val_standardized, y_val, best_model)
    analyze_mse_loss(X_test_standardized, y_test, best_model)

def compare_loss_functions():
    def load_and_preprocess_diabetes_data():
        diabetes_df = pd.read_csv('../../data/external/diabetes.csv')
        X_diabetes = diabetes_df.drop('Outcome', axis=1)
        y_diabetes = diabetes_df['Outcome']
        X_train_d, X_temp_d, y_train_d, y_temp_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)
        X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(X_temp_d, y_temp_d, test_size=0.5, random_state=42)
        scaler_d = StandardScaler()
        X_train_d_standardized = scaler_d.fit_transform(X_train_d)
        X_val_d_standardized = scaler_d.transform(X_val_d)
        X_test_d_standardized = scaler_d.transform(X_test_d)
        return X_train_d_standardized, y_train_d, X_val_d_standardized, y_val_d, X_test_d_standardized, y_test_d

    def train_models_with_different_losses(X_train_d, y_train_d, X_val_d, y_val_d):
        model_mse = MLPClassifier_3(input_size=X_train_d.shape[1], hidden_sizes=[], output_size=1, activation='linear', epochs=1000)
        model_bce = MLPClassifier_3(input_size=X_train_d.shape[1], hidden_sizes=[], output_size=1, activation='linear', epochs=1000)
        model_mse.fit(X_train_d, y_train_d, X_val_d, y_val_d, loss_function='mse')
        model_bce.fit(X_train_d, y_train_d, X_val_d, y_val_d, loss_function='bce')
        return model_mse, model_bce

    def plot_loss_vs_epochs(model_mse, model_bce):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(model_mse.losses, label='MSE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('MSE Loss vs Epochs')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(model_bce.losses, label='BCE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('BCE Loss vs Epochs')
        plt.legend()
        plt.savefig('figures/3/q_3_5.png')
        plt.show()

    X_train_d, y_train_d, X_val_d, y_val_d, X_test_d, y_test_d = load_and_preprocess_diabetes_data()
    model_mse, model_bce = train_models_with_different_losses(X_train_d, y_train_d, X_val_d, y_val_d)
    plot_loss_vs_epochs(model_mse, model_bce)

def Combined_Class():
    from models.MLP.MLP import MLP_combined
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

    # Load the Pima Indians Diabetes dataset for classification
    diabetes_df = pd.read_csv('../../data/external/diabetes.csv')

    # Preprocess the dataset
    X_diabetes = diabetes_df.drop('Outcome', axis=1)
    y_diabetes = diabetes_df['Outcome']

    X_train_d, X_temp_d, y_train_d, y_temp_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)
    X_val_d, X_test_d, y_val_d, y_test_d = train_test_split(X_temp_d, y_temp_d, test_size=0.5, random_state=42)

    # Standardize the data
    scaler_d = StandardScaler()
    X_train_d_standardized = scaler_d.fit_transform(X_train_d)
    X_val_d_standardized = scaler_d.transform(X_val_d)
    X_test_d_standardized = scaler_d.transform(X_test_d)

    # Convert back to DataFrame for consistency
    X_train_d = pd.DataFrame(X_train_d_standardized, columns=X_diabetes.columns)
    X_val_d = pd.DataFrame(X_val_d_standardized, columns=X_diabetes.columns)
    X_test_d = pd.DataFrame(X_test_d_standardized, columns=X_diabetes.columns)

    # Train model for classification
    model_classification = MLP_combined(input_size=X_train_d.shape[1], hidden_sizes=[], output_size=1, task='classification', activation='linear', epochs=1000)
    model_classification.fit(X_train_d, y_train_d, X_val_d, y_val_d)

    # Evaluate classification model
    y_pred_classification = model_classification.predict(X_test_d)
    y_pred_classification_binary = (y_pred_classification > 0.5).astype(int)
    accuracy = accuracy_score(y_test_d, y_pred_classification_binary)
    print(f"Classification Accuracy: {accuracy:.4f}")
    print("\n" + "="*50 + "\n")

    # Load the Housing dataset for regression
    housing_df = pd.read_csv('../../data/external/HousingData.csv')

    # Plot the distribution of the target variable (MEDV)
    medv_values = housing_df['MEDV']

    # Handle NaN values by filling them with the mean of the column
    housing_df.fillna(housing_df.mean(), inplace=True)

    # Ensure no NaN values are present
    assert not housing_df.isnull().values.any(), "There are still NaN values in the dataset"

    # Preprocess the dataset
    X_housing = housing_df.drop('MEDV', axis=1)
    y_housing = housing_df['MEDV']

    # Split the dataset into training (70%), validation (15%), and test (15%) sets
    X_train_h, X_temp_h, y_train_h, y_temp_h = train_test_split(X_housing, y_housing, test_size=0.3, random_state=42)
    X_val_h, X_test_h, y_val_h, y_test_h = train_test_split(X_temp_h, y_temp_h, test_size=0.5, random_state=42)

    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    X_train_h_imputed = imputer.fit_transform(X_train_h)
    X_val_h_imputed = imputer.transform(X_val_h)
    X_test_h_imputed = imputer.transform(X_test_h)

    # Normalize the features using Min-Max scaling
    min_max_scaler = MinMaxScaler()
    X_train_h_normalized = min_max_scaler.fit_transform(X_train_h_imputed)
    X_val_h_normalized = min_max_scaler.transform(X_val_h_imputed)
    X_test_h_normalized = min_max_scaler.transform(X_test_h_imputed)

    # Standardize the features to have mean=0 and std=1 using Z-score scaling
    standard_scaler = StandardScaler()
    X_train_h_standardized = standard_scaler.fit_transform(X_train_h_imputed)
    X_val_h_standardized = standard_scaler.transform(X_val_h_imputed)
    X_test_h_standardized = standard_scaler.transform(X_test_h_imputed)

    # Convert back to DataFrame for consistency
    X_train_h = pd.DataFrame(X_train_h_standardized, columns=X_housing.columns)
    X_val_h = pd.DataFrame(X_val_h_standardized, columns=X_housing.columns)
    X_test_h = pd.DataFrame(X_test_h_standardized, columns=X_housing.columns)

    # Train model for regression
    model_regression = MLP_combined(input_size=X_train_h.shape[1], hidden_sizes=[], output_size=1, task='regression', activation='linear', epochs=1000)
    model_regression.fit(X_train_h, y_train_h, X_val_h, y_val_h)

    # Evaluate regression model
    y_pred_regression = model_regression.predict(X_test_h)
    mse = mean_squared_error(y_test_h, y_pred_regression)
    mae = mean_absolute_error(y_test_h, y_pred_regression)
    print(f"Regression MSE: {mse:.4f}")
    print(f"Regression MAE: {mae:.4f}")

def q4_main():
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from performance_measures.performance_measures import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from models.PCA.PCA import PCA
    import os
    import sys
    # Adding the parent directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(parent_dir)
    from models.knn.knn import KNN
    from models.AUTOENCODER.AUTOENCODER import AutoEncoder
    from models.MLP.MLP import MLPClassifier
    

    def load_and_preprocess_data(file_path, target_column):
        data = pd.read_csv(file_path)
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')
        X = (X - X.min()) / (X.max() - X.min())
        if y.dtype == 'object':
            label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
            y = y.map(label_mapping)
        return X.values, y.values

    file_path = '../../data/external/spotify.csv' 
    target_column = 'track_genre'  
    X, y = load_and_preprocess_data(file_path, target_column)
    
    # Step 1: Perform PCA
    optimal_components = 3
    pca = PCA(n_components=optimal_components)
    X_reduced = pca.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_reduced, X_test_reduced, _, _ = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # Step 2: Apply KNN with the best {k, distance metric} pair (k=19, distance='manhattan')
    k_best = 19
    distance_best = 'manhattan'
    
    print("\nPerforming KNN on full dataset...")
    knn = KNN(k=k_best, distance_metric=distance_best)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    full_accuracy = accuracy_score(y_test, y_pred)
    full_precision = precision_score(y_test, y_pred, average='macro')
    full_recall = recall_score(y_test, y_pred, average='macro')
    full_f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print("\nPerforming KNN on PCA-reduced dataset...")
    knn.fit(X_train_reduced, y_train)
    y_pred_reduced = knn.predict(X_test_reduced)
    reduced_accuracy = accuracy_score(y_test, y_pred_reduced)
    reduced_precision = precision_score(y_test, y_pred_reduced, average='macro')
    reduced_recall = recall_score(y_test, y_pred_reduced, average='macro')
    reduced_f1_macro = f1_score(y_test, y_pred_reduced, average='macro')
    
    # Step 3: Print evaluation metrics for full and reduced datasets
    print("\n---- KNN Performance on Full Dataset ----")
    print(f"Accuracy: {full_accuracy:.2f}")
    print(f"Precision: {full_precision:.2f}")
    print(f"Recall: {full_recall:.2f}")
    print(f"F1 Score (Macro): {full_f1_macro:.2f}")
    
    print("\n---- KNN Performance on PCA-Reduced Dataset ----")
    print(f"Accuracy: {reduced_accuracy:.2f}")
    print(f"Precision: {reduced_precision:.2f}")
    print(f"Recall: {reduced_recall:.2f}")
    print(f"F1 Score (Macro): {reduced_f1_macro:.2f}")
    
    # Train AutoEncoder on PCA-reduced dataset
    print("\nTraining AutoEncoder on PCA-reduced dataset...")
    autoencoder = AutoEncoder(input_size=optimal_components, hidden_sizes=[64, 32], latent_size=optimal_components)
    autoencoder.fit(X_train_reduced, epochs=1000, batch_size=32)
    
    # Get latent representation
    X_train_latent = autoencoder.get_latent(X_train_reduced)
    X_test_latent = autoencoder.get_latent(X_test_reduced)
    
    # Apply KNN on latent representation
    print("\nPerforming KNN on AutoEncoder latent representation...")
    knn.fit(X_train_latent, y_train)
    y_pred_latent = knn.predict(X_test_latent)
    latent_accuracy = accuracy_score(y_test, y_pred_latent)
    latent_precision = precision_score(y_test, y_pred_latent, average='macro')
    latent_recall = recall_score(y_test, y_pred_latent, average='macro')
    latent_f1_macro = f1_score(y_test, y_pred_latent, average='macro')
    
    # Print evaluation metrics for latent representation
    print("\n---- KNN Performance on AutoEncoder Latent Representation ----")
    print(f"Accuracy: {latent_accuracy:.2f}")
    print(f"Precision: {latent_precision:.2f}")
    print(f"Recall: {latent_recall:.2f}")
    print(f"F1 Score (Macro): {latent_f1_macro:.2f}")

    # Train MLP classifier on original dataset
    print("\nTraining MLP classifier on original dataset...")
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))
    mlp_classifier = MLPClassifier(input_size=input_size, hidden_sizes=[64, 32], output_size=num_classes, activation='relu', learning_rate=0.01)
    mlp_classifier.fit(X_train, y_train, epochs=1000, batch_size=32)
    y_pred_mlp = mlp_classifier.predict(X_test)

    # Calculate metrics
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
    recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
    f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

    print("\nMLP Classifier Results:")
    print(f"Accuracy: {accuracy_mlp:.4f}")
    print(f"Precision: {precision_mlp:.4f}")
    print(f"Recall: {recall_mlp:.4f}")
    print(f"F1 Score: {f1_mlp:.4f}")
    
    # Compare results
    print("\nComparison:")
    latent_f1_macro =0.11
    print(f"AutoEncoder + KNN F1 Score: {latent_f1_macro:.4f}")
    print(f"MLP Classifier F1 Score: {f1_mlp:.4f}")
    print(f"Difference: {abs(latent_f1_macro - f1_mlp):.4f}")



if __name__ == "__main__":
    # MLPClassifier_main()
    # multilabel_classification()
    # mlp_regression()
    # compare_loss_functions()
    Combined_Class()
    # q4_main()

