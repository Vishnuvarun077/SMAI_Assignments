When you run  the 'a1.py' file first it will ask the user :
# Assignment 1 solutions:
- Welcome to the Assignment 1 Solutions!
 - 1. KNN Tasks
 - 2. Linear Regression Tasks
 - 3. Exit
 Enter your choice (1-3): 
- After each process completion same menu will be prompted.

- If user enters 1 then it will prompt:
# KNN Tasks:
 - 1. Exploratory Data Analysis
 - 2. KNN Base Case
 - 3. Hyperparameter Tuning (without feature selection)
 - 4. Feature Selection
 - 5. Optimization
 - 6. Second Data Set
 - 7. Implement all
 - 8. Return to Main Menu
 - 9. Exit
 - Enter your choice (1-9): 
- After each process completion same menu will be prompted.
If user enters the number one then it will performs the Exploratory Data Analysis part
## Explaination for the EDA part: 
In this section : I performed a comprehensive analysis of the dataset to understand its structure, identify patterns, and detect anomalies. The steps involved in the EDA process are as follows:

## Loading Data: 
The load_data function reads the dataset from a CSV file and drops any rows with missing values to ensure data quality.

## Encoding Target Variable:
The encode_target function converts categorical target variables into numerical codes, making them suitable for analysis.

## Removing Outliers: 
The remove_outliers_zscore function removes outliers from the dataset using the z-score method, which helps in reducing the impact of extreme values on the analysis.

## Plotting Feature Distributions:
 The plot_feature_distribution function generates histograms for each numeric feature to visualize their distributions. It also annotates skewness and potential outliers.

## Plotting Correlation Heatmap:
 The plot_correlation_heatmap function creates a heatmap to show the correlation between numeric features, helping to identify relationships between variables.

## Plotting Feature Importance: 
The plot_feature_importance function plots the absolute correlation of each feature with the target variable, indicating their importance.

## Normalizing Data: 
The normalize_data function scales numeric features to a range between 0 and 1, ensuring that all features contribute equally to the analysis.

## Writing Observations:
 The results and observations from the EDA are written to a file (edaresults.txt). This includes dataset overview, feature distributions, pairwise relationships, and a proposed hierarchy of feature importance.

## By following these steps, I got the following results:

## Dataset Overview
- Number of samples (original): 113999
- Number of samples (after outlier removal): 104363
- Number of features: 19
- Target variable: track_genre

## Feature Distributions
-Note these are the metrics after outliners removal
- See individual distribution plots saved in the 'figures/EDA' directory.
- popularity:
  - Mean: 33.35
  - Median: 35.00
  - Skewness: 0.04

- duration_ms:
  - Mean: 224203.24
  - Median: 213089.00
  - Skewness: 0.97

- danceability:
  - Mean: 0.58
  - Median: 0.59
  - Skewness: -0.36

- energy:
  - Mean: 0.65
  - Median: 0.69
  - Skewness: -0.54

- key:
  - Mean: 5.33
  - Median: 5.00
  - Skewness: -0.02

- loudness:
  - Mean: -7.73
  - Median: -6.84
  - Skewness: -1.16
  - Observation: Skewed distribution

- mode:
  - Mean: 0.63
  - Median: 1.00
  - Skewness: -0.56

- speechiness:
  - Mean: 0.07
  - Median: 0.05
  - Skewness: 2.42
  - Observation: Skewed distribution

- acousticness:
  - Mean: 0.29
  - Median: 0.15
  - Skewness: 0.81

- instrumentalness:
  - Mean: 0.14
  - Median: 0.00
  - Skewness: 1.88
  - Observation: Skewed distribution

- liveness:
  - Mean: 0.19
  - Median: 0.13
  - Skewness: 1.72
  - Observation: Skewed distribution

- valence:
  - Mean: 0.48
  - Median: 0.47
  - Skewness: 0.11

- tempo:
  - Mean: 122.98
  - Median: 122.87
  - Skewness: 0.33

- time_signature:
  - Mean: 3.94
  - Median: 4.00
  - Skewness: -1.89
  - Observation: Skewed distribution

- track_genre:
  - Mean: 56.56
  - Median: 56.00
  - Skewness: 0.00

## Pairwise Relationships
- See 'before_pairplot.png' and 'after_pairplot.png for pairwise relationship visualizations.
- Observations:
  -From the before_pairplot we can see that  mode ,key and time_signature shows no effect on remaining parameters
  -So I have removed these three columns and plottedd the pair plot again which is after_pairplot
  -In the after_pairplot all the values show significant imapact on track_genere.

## Feature Hierarchy for Classification
Based on the correlation with the target variable and the feature distributions, here's a proposed hierarchy of feature importance:
  - acoustness 
  - instrumentalness
  - energy
  - valence
  - loudness
  - liveness
  - speechiness
  - duration_ms
  - tempo
  - popularity
  - danceability
- If user enters two base knn model is implemented:
## KNN model: 
- In my implementation of the K-Nearest Neighbors (KNN) algorithm, I initially created a simple version called       initial_KNN and then optimized it using a vectorization approach.

- Initial Implementation (initial_KNN)
## Initialization:
- I defined the initial_KNN class with parameters k (number of neighbors) and distance_metric (type of distance calculation).I stored the training data (X_train and y_train) and defined a dictionary to map distance metric names to their corresponding functions.

## Training:
 The fit method simply stored the training data.

## Prediction:
The predict method iterated over each sample in the test set and used the prediction_helper method to predict the label.The prediction_helper method calculated distances between the test sample and all training samples, selected the k nearest neighbors, and determined the most common label among them.

## Distance Calculations:
 I implemented three distance metrics: Euclidean, Manhattan, and Cosine.


## Optimized Implementation (KNN)
To improve the efficiency of the initial implementation, I optimized it using vectorization:

## Initialization:
I added a batch_size parameter to handle predictions in batches, which helps in managing memory usage and computational efficiency.

## Training:
The fit method remained the same, storing the training data.

## Prediction:
The predict method was modified to process the test samples in batches.
For each batch, I calculated distances between the batch samples and all training samples using the selected distance metric.
I used np.argpartition to efficiently find the indices of the k nearest neighbors.
I applied np.apply_along_axis to determine the most common label among the nearest neighbors for each sample in the batch.

## Distance Calculations:
I implemented three distance metrics: Euclidean, Manhattan, and Cosine.
I vectorized the distance calculation methods to handle multiple samples at once, significantly improving the computational efficiency.By adopting a vectorized approach, I was able to optimize the KNN algorithm, making it more efficient and scalable for larger datasets.

- Then I have tested the baseknn implementation by the function  knn_classification under base knn classification section with k = 3 and distance_metric = euclidean and got the following results:
Accuracy: 0.18
Precision (macro): 0.20
Recall (macro): 0.18
F1 Score (macro): 0.19
F1 Score (micro): 0.18
- If user enters 3 then Hyperparameter Tuning will be implemented
 ## 3. Hyperparameter Tuning:
 - ## Data Preparation:
 - I started by loading and preprocessing the data using the load_and_preprocess_data function. This function handles the conversion of non-numeric columns to numeric, drops columns that cannot be converted, and normalizes the features.
 - I then split the data into 80% training and 20% remaining, and further split the remaining 20% into 50% validation and 50% test sets.
 - I considered only odd values of k ranging from 1 to 20 and three distance metrics: 'euclidean', 'manhattan', and 'cosine'.For each combination of k and distance metric, I trained a KNN model and evaluated its accuracy on the validation set.I stored the results and sorted them to find the top 10 {k, distance metric} pairs with the highest validation accuracy.
- Results:
- - ## Top 10 {k, distance_metric} pairs:
- k: 19, Distance Metric: manhattan, Accuracy: 0.2145
- k: 17, Distance Metric: manhattan, Accuracy: 0.2135
- k: 15, Distance Metric: manhattan, Accuracy: 0.2115
- k: 11, Distance Metric: manhattan, Accuracy: 0.2111
- k: 13, Distance Metric: manhattan, Accuracy: 0.2093
- k: 9, Distance Metric: manhattan, Accuracy: 0.2054
- k: 7, Distance Metric: manhattan, Accuracy: 0.2024
- k: 1, Distance Metric: manhattan, Accuracy: 0.2011
- k: 5, Distance Metric: manhattan, Accuracy: 0.1961
- k: 15, Distance Metric: euclidean, Accuracy: 0.1918
- ## Dropping Columns:
- I experimented with dropping various columns to see if it improved accuracy. I iterated over all combinations of columns and evaluated the accuracy for each combination.I documented the combination of columns that gave the best results.
- I got the following results: 
  - Best columns combination:  ['tempo']
  - Best accuracy with dropped columns: 0.0640
- If we user enters 4 Feature Selection will be implemented.
## Feature Selection:
- In this task, I implemented a greedy forward selection algorithm to select the best combination of features. This algorithm iteratively adds the feature that improves accuracy the most until no further improvement is possible.
I documented the best features selected and the corresponding accuracy.
- I got the following results:
 - Best features selected by Greedy Forward Selection:  ['tempo', 'popularity', 'acousticness', 'danceability', 'instrumentalness', 'loudness', 'speechiness', 'duration_ms', 'valence', 'energy']
 - Best accuracy with Greedy Forward Selection: 0.2497
- If user enters 5 Optimization task will be implemented.
## Optimization
Task 1: Improving Execution Time
- ## Vectorization:
- I optimized the KNN algorithm using vectorization to improve its execution time. This involved modifying the distance calculation methods to handle multiple samples at once, significantly improving computational efficiency.
- # Inference Time Comparison:
- I compared the inference time of four models: the initial KNN model(intialknn,k=3,euclidian), the best KNN model(initalknn, k= 19, manhatten), the optimized KNN model(knn,k = 19, manhatten), and the default sklearn KNN model.I plotted the inference time for each model and saved the plot:
- In the inference_time_comparision.png we can clearly see that sklearn has the lowest time and intial knn has second best time due to low k value.
- We can clearly see optimised knn has less time than best knn  due to vecotrization.
- Inference Time vs Train Dataset Size:

I plotted the inference time vs train dataset size for the four models mentioned above.
  - Here I have observed that intial knn model which is used in intial knn and best knn have same metrics as they both are same but with different k and distance metric.
  - On the other hand optimised knn which is optimised version of intial knn through vectorization has better times at higher train sizes 
  - Overall skitlearn inbuilt model has lower times at all stages.
- If user enters 6 operations on second data set will be implemented.
## A Second Dataset
- Application on a New Dataset:
 - I applied the best {k, distance metric} pair obtained from the previous hyperparameter tuning on a new dataset provided in data/external/spotify-2.
 - I used the pre-split train, validate, and test sets from this dataset.
 - I documented my observations on the data and the performance of the KNN model on this new dataset:
 - Validation Set Results:
   - Validation Accuracy: 0.28
   - Validation Precision (macro): 0.28
   - Validation Recall (macro): 0.28
   - Validation F1 Score (macro): 0.28
   - Validation F1 Score (micro): 0.28

  - Test Set Results:
    - Test Accuracy: 0.28
    - Test Precision (macro): 0.27
    - Test Recall (macro): 0.28
    - Test F1 Score (macro): 0.28
    - Test F1 Score (micro): 0.28


- By following these steps, I was able to systematically tune the hyperparameters, optimize the KNN algorithm, and evaluate its performance on a new dataset. This comprehensive approach ensured that I achieved the best possible results while maintaining efficiency.
- If user enters  7 all the above tasks will be implemented sequentially.
## Linear Regression : 
-- After selecting 2 in the first prompt  the code will again prompt  the following :
 1. Simple Regression with Degree 1
 2. Simple Regression with Degree Greater than 1
 3. Animation
 4. Regularization
 5. Implement all
 6. Return to Main Menu
 7. Exit
- After each process completion same menu will be prompted.
- Here I have implemented the linear regression class in the following way:
I defined a LinearRegression class with the following attributes and methods:

- # Attributes:
 - learning_rate: The learning rate for gradient descent.
 - iterations: The number of iterations for gradient descent.
 - reg_type: The type of regularization ('L1' for Lasso, 'L2' for Ridge).
 - lambda_: The regularization parameter.
 - degree: The degree of the polynomial for polynomial regression.
 - coefficients: The coefficients of the regression model.
 - mse_history: A list to store the Mean Squared Error (MSE) over iterations.
 - std_dev_history: A list to store the standard deviation of errors over iterations.
 - variance_history: A list to store the variance of errors over iterations.
- # Methods:
 - add_polynomial_features_for_animation(X): Adds polynomial features to the input data for animation purposes.
 - add_polynomial_features(X): Adds polynomial features to the input data.
  - fit(X, y): Fits the linear regression model to the training data.
  - predict(X): Predicts the output for the given input data.
  - mse(X, y): Calculates the Mean Squared Error for the given data.
  - std_dev(X, y): Calculates the standard deviation of errors for the given data.
  - variance(X, y): Calculates the variance of errors for the given data.
 - Polynomial Features
  - To handle polynomial regression, I implemented methods to add polynomial features to the input data. The add_polynomial_features_for_animation method adds polynomial features up to the specified degree for animation purposes, while the add_polynomial_features method uses combinations with replacement to generate polynomial features.

- #  Model Fitting
 - The fit method fits the linear regression model to the training data using gradient descent. If the degree is greater than 1, polynomial features are added to the input data. The method also supports L1 and L2 regularization to prevent overfitting. The coefficients are updated iteratively, and the MSE, standard deviation, and variance of errors are recorded over iterations.

- # Predictions and Error Metrics
 - The predict method predicts the output for the given input data by applying the learned coefficients. The mse,    std_dev, and variance methods calculate the Mean Squared Error, standard deviation, and variance of errors, respectively, for the given data.

- By implementing the LinearRegression class, I systematically addressed the tasks of fitting linear and polynomial regression models, incorporating regularization techniques, and calculating error metrics. This implementation provided a comprehensive understanding of linear regression, polynomial regression, and regularization techniques.

# Data Preparation:
 - I started by loading the data from linreg.csv, which contains 400 points sampled from a polynomial. The data was shuffled and split into 80% training, 10% validation, and 10% test sets. The splits were visualized using different colors in one graph and saved in the figures folder.
- If user enters 1 Simple Regression with Degree 1 will be implemented.
# Simple Regression with Degree 1:
- For degree 1 regression, I fit a linear model to the data. I experimented with different learning rates and selected the best one based on the validation set's Mean Squared Error (MSE). The model was then trained with the best learning rate, and metrics such as MSE, standard deviation, and variance were reported for both the training and test sets. The training points and the fitted line were plotted.
- Results obtained when tested for learning rates from 0.001 to 1 with 1000 points in between are:
 - Train MSE: 0.3859926099458966, Std Dev: 0.6211960092811467, Variance: 0.3858844819468225
 - Test MSE: 0.28054655485614066, Std Dev: 0.529661854662208, Variance: 0.28054168028420995
 - Best learning rate: 0.007
 - The plot containing the best fit line is saved in figures/linearregression.
- If user enters 2 Simple Regression with Degree Greater than 1  will be implemented.
# Simple Regression with Degree Greater than 1:
- For higher degree regression, I fit polynomials of varying degrees (up to 10) to the data. I reported the MSE, standard deviation, and variance metrics for each degree on both the training and test sets. The degree that minimized the test MSE was identified as the best degree. The coefficients of the best model were saved to a file, and the best degree polynomial fitting the data was plotted.
- The Results obtained are:
 - Degree 2 Polynomial Results:
   - Train MSE: 1.1864, Std Dev: 1.0892, Variance: 1.1864
   - Test MSE: 1.1698, Std Dev: 1.0770, Variance: 1.1599
 - Degree 3 Polynomial Results:
   - Train MSE: 0.4496, Std Dev: 0.6705, Variance: 0.4495
   - Test MSE: 0.3457, Std Dev: 0.5880, Variance: 0.3457
 - Degree 4 Polynomial Results:
   - Train MSE: 1.1779, Std Dev: 1.0853, Variance: 1.1778
   - Test MSE: 1.0990, Std Dev: 1.0454, Variance: 1.0928
 - Degree 5 Polynomial Results:
   - Train MSE: 0.6169, Std Dev: 0.7854, Variance: 0.6168
   - Test MSE: 0.4908, Std Dev: 0.7006, Variance: 0.4908
 - Degree 6 Polynomial Results:
   - Train MSE: 1.1393, Std Dev: 1.0674, Variance: 1.1393
   - Test MSE: 1.0047, Std Dev: 1.0001, Variance: 1.0002
 - Degree 7 Polynomial Results:
   - Train MSE: 0.7336, Std Dev: 0.8564, Variance: 0.7335
   - Test MSE: 0.5935, Std Dev: 0.7703, Variance: 0.5934
 - Degree 8 Polynomial Results:
    - Train MSE: 1.0933, Std Dev: 1.0456, Variance: 1.0933
    - Test MSE: 0.9209, Std Dev: 0.9578, Variance: 0.9174
 - Degree 9 Polynomial Results:
   - Train MSE: 0.8085, Std Dev: 0.8991, Variance: 0.8084
   - Test MSE: 0.6578, Std Dev: 0.8108, Variance: 0.6575
 - Degree 10 Polynomial Results:
   - Train MSE: 1.0527, Std Dev: 1.0260, Variance: 1.0526
   - Test MSE: 0.8586, Std Dev: 0.9251, Variance: 0.8558
- Best degree (minimizing test MSE): 3
-  If user enters 3 Animation tasks will be implemented.
# Animation:
- I created animations for degrees 1 to 5, visualizing the original data along with the line being fitted to it, as well as the MSE, standard deviation, and variance over iterations. The animations were saved as GIFs.
- If user enters 4 Regularization tasks will be implemented.
- If user enters 5 all the above linearregression tasks will be implemented sequentially.
# Regularization:
- # Data Preparation:
 - I loaded the data from regularisation.csv, which contains 300 points sampled from a 5-degree polynomial. The data was shuffled and split into 80% training, 10% validation, and 10% test sets. The splits were visualized.
- # Regularization Tasks
 - I fit higher degree polynomials (up to 20) to the data, which resulted in overfitting. I plotted the data points and the resulting curves, and reported the MSE, standard deviation, and variance for each degree. To reduce overfitting, I applied L1 and L2 regularization and compared the results. The regularization type was specified as a parameter in the regression class.
- # Overview
  -I conducted an analysis of polynomial regression models with degrees ranging from 1 to 20, using no regularization, L1 regularization, and L2 regularization. The dataset consists of 300 points sampled from a 5-degree polynomial, split into 80:10:10 for training, validation, and testing.
# Key Observations
- # No Regularization
  - The best performing model without regularization is the 4th-degree polynomial, with a test MSE of 0.0551.
  - I observe clear signs of overfitting as the polynomial degree increases beyond 4:
  - The training MSE continues to decrease, while the test MSE starts to increase.
  - This divergence between training and test MSE is a classic indicator of overfitting.
  - The model's performance becomes increasingly unstable at higher degrees, with alternating periods of better and worse performance.

- # L1 Regularization (Lasso)
  - L1 regularization significantly reduces overfitting, especially for higher-degree polynomials.
  - The best performing model with L1 regularization is the 2nd-degree polynomial, with a test MSE of 0.1682.
  - L1 regularization results in more stable performance across different polynomial degrees:
  - The difference between training and test MSE is consistently smaller compared to the non-regularized models.
  - This suggests that L1 regularization is effective in preventing the model from fitting noise in the training data.



- # L2 Regularization (Ridge)
 - L2 regularization also helps in reducing overfitting, but to a lesser extent than L1 regularization.
 - The best performing model with L2 regularization is the 4th-degree polynomial, with a test MSE of 0.1028.
 - L2 regularization allows for slightly higher degree polynomials to perform well compared to L1 regularization:
 - This suggests that L2 regularization is less aggressive in feature selection compared to L1.

- # Comparative Analysis

 - # No Regularization:
   - Pros: Achieves the lowest overall MSE (0.0551 with 4th-degree polynomial).
   - Cons: Highly susceptible to overfitting, especially with higher-degree polynomials.

 - # L1 Regularization:
   - Pros: Most effective at preventing overfitting, provides the most stable performance across degrees.
   - Cons: Slightly higher MSE compared to no regularization and L2 regularization.

 - # L2 Regularization:
   - Pros: Balances between model complexity and overfitting prevention.
   - Cons: Not as effective as L1 in preventing overfitting for very high-degree polynomials.

- # Conclusions:

 - The true underlying model is a 5th-degree polynomial, but I found that a 4th-degree polynomial often performs best. This suggests that the noise in the data makes it challenging to perfectly recover the true model.
 - Regularization is crucial for maintaining model performance and generalization, especially when the polynomial degree is high or when the amount of training data is limited.
 -  L1 regularization appears to be the most effective in this case for preventing overfitting across a wide range of polynomial degrees. However, if one can accurately estimate the true degree of the underlying polynomial, L2 regularization or even no regularization might yield slightly better results.
 - The choice between L1 and L2 regularization depends on the specific goals of the modeling task:
 - If the priority is to have a simple, interpretable model with few features, L1 regularization would be preferred.
 - If the goal is to maintain all features but reduce their impact, L2 regularization would be more appropriate.
 - These results highlight the importance of careful model selection and validation. Simply increasing model complexity (in this case, polynomial degree) does not guarantee better performance, and can often lead to worse results due to overfitting.

- By following the above steps, I systematically addressed the tasks of fitting linear and polynomial regression   models, creating animations to visualize the fitting process, and applying regularization techniques to mitigate overfitting. The results were documented, and the best models were identified based on the test set performance. The implementation provided a comprehensive understanding of linear regression, polynomial regression, and regularization techniques.

# Resources used for completing the assignment
- For writing knn class I have used the "https://www.youtube.com/watch?v=rTEtEy5o3X0&t=2s" and For writing Linearregression class I have used the  "https://www.youtube.com/watch?v=ltXSoduiVwY"
- Later in knn class for writing optimising the code and  I have used the chatgpt with the prompt
"How to optimise the intial_knn using vectorization" then I got  a code which caused a memory error
Then asked to do operations in batches in order to avoid memory errors.
- In Linear regression class  after writing intial code and  I have asked the chatgpt and gave the base code 
along with polynomial features and animation question to modify the code.
- In EDA part I asked the chatgpt write comments on the plot about potential outliners and the asked how to 
print the data summaries.
- In the feature selection part intially I tried to do in brute force method but later asked chatgpt to 
do it in less time  then It gave a method called greedy forward selection function using which I have implemented 
featureselction funcion.
- Where greedy forward selection is a heuristic method that incrementally builds a feature set by adding the feature that provides the most significant improvement in model performance at each step. This approach is "greedy" because it makes the locally optimal choice at each step with the hope of finding a global optimum.

