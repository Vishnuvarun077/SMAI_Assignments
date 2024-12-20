# import numpy as np

# class performance_measures:
#     def __init__(self):
#         pass

#     # Accuracy is the ratio of correctly predicted observation to the total observations 
#     def accuracy(self, y_true, y_pred):
#         return np.sum(y_true == y_pred) / len(y_true)

#     # Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
#     def precision(self, y_true, y_pred, average='macro'):
#         classes = np.unique(y_true)
#         precisions = []
#         for cls in classes:
#             true_positives = np.sum((y_true == cls) & (y_pred == cls))
#             false_positives = np.sum((y_true != cls) & (y_pred == cls))
#             precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#             precisions.append(precision)
#         if average == 'macro':
#             return np.mean(precisions)
#         elif average == 'micro':
#             true_positives = np.sum((y_true == y_pred) & (y_pred != 0))
#             false_positives = np.sum((y_true != y_pred) & (y_pred != 0))
#             return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

#     # Recall is the ratio of correctly predicted positive observations to the all observations in actual class
#     def recall(self, y_true, y_pred, average='macro'):
#         classes = np.unique(y_true)
#         recalls = []
#         for cls in classes:
#             true_positives = np.sum((y_true == cls) & (y_pred == cls))
#             false_negatives = np.sum((y_true == cls) & (y_pred != cls))
#             recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
#             recalls.append(recall)
#         if average == 'macro':
#             return np.mean(recalls)
#         elif average == 'micro':
#             true_positives = np.sum((y_true == y_pred) & (y_pred != 0))
#             false_negatives = np.sum((y_true != y_pred) & (y_true != 0))
#             return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

#     # F1 score macro is the harmonic mean of precision and recall and it is macro so it is calculated for each class and then averaged 
#     def f1_score_macro(self, y_true, y_pred):
#         precision = self.precision(y_true, y_pred, average='macro')
#         recall = self.recall(y_true, y_pred, average='macro')
#         return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     # F1 score micro is the harmonic mean of precision and recall and it is micro so it is calculated for each class and then averaged
#     def f1_score_micro(self, y_true, y_pred):
#         precision = self.precision(y_true, y_pred, average='micro')
#         recall = self.recall(y_true, y_pred, average='micro')
#         return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     # Confusion matrix is a table used to describe the performance of a classification model 
#     def confusion_matrix(self, y_true, y_pred):
#         classes = np.unique(y_true)
#         matrix = np.zeros((len(classes), len(classes)), dtype=int)
#         for i, cls_true in enumerate(classes):
#             for j, cls_pred in enumerate(classes):
#                 matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))
#         return matrix
    
#     # Mean Squared Error
#     def mean_squared_error(self, y_true, y_pred):
#         return np.mean((y_true - y_pred) ** 2)

#     # Mean Absolute Error
#     def mean_absolute_error(self, y_true, y_pred):
#         return np.mean(np.abs(y_true - y_pred))

#     # Root Mean Squared Error
#     def root_mean_squared_error(self, y_true, y_pred):
#         return np.sqrt(self.mean_squared_error(y_true, y_pred))

#     # R-squared
#     def r_squared(self, y_true, y_pred):
#         ss_res = np.sum((y_true - y_pred) ** 2)
#         ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#         return 1 - (ss_res / ss_tot)


import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def precision_score(y_true, y_pred, average='weighted'):
    unique_classes = np.unique(y_true)
    precisions = []
    for cls in unique_classes:
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        predicted_positive = np.sum(y_pred == cls)
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        precisions.append(precision)
    if average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in unique_classes]
        return np.average(precisions, weights=weights)
    return np.mean(precisions)

def recall_score(y_true, y_pred, average='weighted'):
    unique_classes = np.unique(y_true)
    recalls = []
    for cls in unique_classes:
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        actual_positive = np.sum(y_true == cls)
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        recalls.append(recall)
    if average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in unique_classes]
        return np.average(recalls, weights=weights)
    return np.mean(recalls)

def f1_score(y_true, y_pred, average='weighted'):
    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)
    f1s = []
    for p, r in zip(precisions, recalls):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    if average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in np.unique(y_true)]
        return np.average(f1s, weights=weights)
    return np.mean(f1s)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1, precision, recall