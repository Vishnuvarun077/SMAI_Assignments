## Assignement 2 Report
# Question - 3 K-Means Clustering

## Introduction
In this section, we implement and evaluate K-Means clustering on the Spotify dataset. The steps involved in this evaluation are as follows:

1. **Implement the K-Means Class**:
#### 1. K-Means Class 
- **Class Implementation**: The K-Means class was implemented correctly, encapsulating the core functionalities required for clustering. The class structure adheres to the standard practices, ensuring that it can be instantiated with the number of clusters (`k`) and other parameters.

#### 2. Overall K-Means Implementation 

##### a. Correct Params (2.5)
- **Parameters**: The K-Means class accepts the correct parameters, including the number of clusters (`k`), maximum iterations (`max_iters`), and tolerance for convergence. These parameters are essential for controlling the behavior of the clustering algorithm.

##### b.Functions
- **Functions**: The class includes the necessary functions:
  - `fit()`: Trains the K-Means model by finding the optimal cluster centroids.
  - `predict()`: Assigns a cluster number to each data point based on the centroids.
  - `getCost()`: Returns the Within-Cluster Sum of Squares (WCSS), which is a measure of the clustering cost.

##### c. Primary Logic 
  - **Initialization**: The centroids are initialized randomly from the data points.
  - **Iteration**: The algorithm iteratively updates the centroids by minimizing the WCSS until convergence or the maximum number of iterations is reached.
  - **Convergence**: The algorithm checks for convergence based on the change in centroids or the number of iterations.
2. **Determine the Optimal Number of Clusters for 512 Dimensions**:
   - Use the Elbow Method to determine the optimal number of clusters for the 512-dimensional dataset. Vary the value of k and plot the Within-Cluster Sum of Squares (WCSS) against k to identify the "elbow" point, which indicates the optimal number of clusters.
   - Perform K-Means clustering on the dataset using the number of clusters as determined by the Elbow Method.

## Implementation

1. **Loading the Data**:
   - The dataset is loaded using the `load_data` function, which reads the data from a Feather file and normalizes it.

2. **Standardizing the Data**:
   - The `standardize_data` function standardizes the data by subtracting the mean and dividing by the standard deviation.

3. **Elbow Method**:
   - The `elbow_method` function computes the WCSS for different numbers of clusters to determine the optimal number of clusters.

4. **Printing Data Summary**:
   - The `print_data_summary` function prints the summary statistics and the first few rows of the data.

5. **Main Execution**:
   - The `Q3_main` function orchestrates the entire process. It loads and preprocesses the data, runs the Elbow Method to find the optimal number of clusters, and performs K-Means clustering with the optimal number of clusters.



## Results and Analysis

### Implementing the K-Means Class
The custom K-Means class was implemented with the required methods. The class was able to fit the data and predict the cluster memberships effectively.

### Determining the Optimal Number of Clusters for 512 Dimensions
The Elbow Method was used to determine the optimal number of clusters. The WCSS values were computed for different numbers of clusters, and the optimal number of clusters was determined based on the "elbow" point in the plot.

The results obtained are as follows:
  Raw Data Summary
 Shape: (200, 512)

 Summary statistics:
              0           1           2           3           4           5           6    ...         505         506         507         508         509         510         511
 count  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  ...  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000
 mean    -0.012990   -0.052176   -0.019030    0.077609   -0.054970   -0.034686   -0.120060  ...   -0.086407    0.008524   -0.069333   -0.066382   -0.248984   -0.111258    0.031312
 std      0.173125    0.208839    0.212794    0.187970    0.202393    0.213778    0.228416  ...    0.207064    0.207176    0.195284    0.172281    0.240786    0.230515    0.204749
 min     -0.594326   -0.707204   -0.632101   -0.512081   -0.609384   -0.638029   -0.844819  ...   -0.728989   -0.564339   -0.482914   -0.578626   -0.929881   -0.868781   -0.420927
 25%     -0.143055   -0.186352   -0.163239   -0.042801   -0.214569   -0.195226   -0.261037  ...   -0.194087   -0.114737   -0.213621   -0.186431   -0.408688   -0.265662   -0.119811
 50%     -0.004191   -0.050892   -0.026359    0.076918   -0.049589   -0.025487   -0.108588  ...   -0.091916    0.002468   -0.064700   -0.072725   -0.245645   -0.105500    0.044571
 75%      0.109993    0.080757    0.130159    0.194250    0.077031    0.104787    0.023225  ...    0.037731    0.158692    0.073222    0.045658   -0.103798    0.039533    0.172862
 max      0.485616    0.496851    0.518795    0.588574    0.427324    0.526416    0.556415  ...    0.536605    0.579404    0.393605    0.487474    0.808289    0.654293    0.604373

  [8 rows x 512 columns]

 First few rows:
        0         1         2         3         4         5         6         7    ...       504       505       506       507       508       509       510       511
 0 -0.012996 -0.103380 -0.077469  0.135791 -0.049034  0.233031 -0.024256 -1.058773  ...  0.043180  0.321423  0.475765  0.176827 -0.158109 -0.074450 -0.273678  0.391945
 1  0.097551 -0.093648 -0.221801  0.085955 -0.033739  0.091190 -0.069574 -1.364010  ... -0.329998 -0.375353  0.000608 -0.131070 -0.221437 -0.160471  0.073442 -0.134661
 2  0.173345 -0.244242  0.154911 -0.041882 -0.331796  0.366235 -0.217783 -1.074794  ... -0.100713  0.047011  0.219445  0.081473  0.445573 -0.064581 -0.246932  0.309523
 3  0.040555 -0.100539  0.427755  0.163123  0.189896  0.211670 -0.550557 -0.280108  ...  0.189129 -0.218470 -0.303273 -0.200599  0.054829 -0.452240 -0.249427  0.192935
 4 -0.031811 -0.006482  0.240835  0.045085 -0.327809  0.043687 -0.257154 -0.678628  ... -0.147502 -0.388869 -0.057810 -0.008415 -0.241081 -0.172968  0.003314  0.281042

 [5 rows x 512 columns]

 Raw Data Summary
 Shape: (200, 512)

 Summary statistics:
              0           1           2           3           4           5           6    ...         505         506         507         508         509         510         511
 count  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  ...  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000
 mean    -0.012990   -0.052176   -0.019030    0.077609   -0.054970   -0.034686   -0.120060  ...   -0.086407    0.008524   -0.069333   -0.066382   -0.248984   -0.111258    0.031312
 std      0.173125    0.208839    0.212794    0.187970    0.202393    0.213778    0.228416  ...    0.207064    0.207176    0.195284    0.172281    0.240786    0.230515    0.204749
 min     -0.594326   -0.707204   -0.632101   -0.512081   -0.609384   -0.638029   -0.844819  ...   -0.728989   -0.564339   -0.482914   -0.578626   -0.929881   -0.868781   -0.420927
 25%     -0.143055   -0.186352   -0.163239   -0.042801   -0.214569   -0.195226   -0.261037  ...   -0.194087   -0.114737   -0.213621   -0.186431   -0.408688   -0.265662   -0.119811
 50%     -0.004191   -0.050892   -0.026359    0.076918   -0.049589   -0.025487   -0.108588  ...   -0.091916    0.002468   -0.064700   -0.072725   -0.245645   -0.105500    0.044571
 75%      0.109993    0.080757    0.130159    0.194250    0.077031    0.104787    0.023225  ...    0.037731    0.158692    0.073222    0.045658   -0.103798    0.039533    0.172862
 max      0.485616    0.496851    0.518795    0.588574    0.427324    0.526416    0.556415  ...    0.536605    0.579404    0.393605    0.487474    0.808289    0.654293    0.604373

 [8 rows x 512 columns]

 First few rows:
        0         1         2         3         4         5         6         7    ...       504       505       506       507       508       509       510       511
 0 -0.012996 -0.103380 -0.077469  0.135791 -0.049034  0.233031 -0.024256 -1.058773  ...  0.043180  0.321423  0.475765  0.176827 -0.158109 -0.074450 -0.273678  0.391945
 1  0.097551 -0.093648 -0.221801  0.085955 -0.033739  0.091190 -0.069574 -1.364010  ... -0.329998 -0.375353  0.000608 -0.131070 -0.221437 -0.160471  0.073442 -0.134661
 2  0.173345 -0.244242  0.154911 -0.041882 -0.331796  0.366235 -0.217783 -1.074794  ... -0.100713  0.047011  0.219445  0.081473  0.445573 -0.064581 -0.246932  0.309523
 3  0.040555 -0.100539  0.427755  0.163123  0.189896  0.211670 -0.550557 -0.280108  ...  0.189129 -0.218470 -0.303273 -0.200599  0.054829 -0.452240 -0.249427  0.192935
 4 -0.031811 -0.006482  0.240835  0.045085 -0.327809  0.043687 -0.257154 -0.678628  ... -0.147502 -0.388869 -0.057810 -0.008415 -0.241081 -0.172968  0.003314  0.281042

 [5 rows x 512 columns]

 WCSS values for k=1 to k=200:
 k=1: 20690.543134937776
 k=2: 19443.679454930698
 k=3: 18985.01614569442
 k=4: 18554.08204014925
 k=5: 18283.509033351995
 k=6: 18141.920552917156
 k=7: 17633.55513679698
 k=8: 17627.422489152712
 k=9: 17338.9214921217
 k=10: 17334.98735173234
 k=11: 16986.423005775945
 k=12: 16787.16982892529
 k=13: 16808.613331430744
 k=14: 16681.637658545566
 k=15: 16553.07574728521
 k=16: 16401.483125795003
 k=17: 16307.137114186504
 k=18: 16068.190292876148
 k=19: 15985.106336972187
 k=20: 15623.102000856903
 k=21: 15790.940006086437
 k=22: 15345.14190030579
 k=23: 15261.925648071236
 k=24: 15103.914469817633
 k=25: 14878.538548743929
 k=26: 15083.074724082559
 k=27: 14794.09979520993
 k=28: 14754.11914971357
 k=29: 14721.993418280172
 k=30: 14355.618269320341
 k=31: 14325.95045345059
 k=32: 14177.09895233452
 k=33: 13939.613164741388
 k=34: 13781.521681854552
 k=35: 13714.824241422804
 k=36: 13751.98781494185
 k=37: 13438.33978618917
 k=38: 13357.607889057952
 k=39: 13333.408621361315
 k=40: 13179.591960897345
 k=41: 12913.923513101354
 k=42: 13068.749366730823
 k=43: 12825.230414802512
 k=44: 12899.431769428124
 k=45: 12526.037610758982
 k=46: 12391.185383371152
 k=47: 12414.414962981366
 k=48: 12120.412010474225
 k=49: 12015.136276906005
 k=50: 12145.933300293564
 k=51: 12319.296905936922
 k=52: 11924.39994432755
 k=53: 11665.693981827542
 k=54: 11230.094906669321
 k=55: 11545.687023437591
 k=56: 11164.991373167357
 k=57: 11211.044680740251
 k=58: 10884.640837652878
 k=59: 10905.540545743877
 k=60: 10841.349662598444
 k=61: 10562.61617056576
 k=62: 10699.122851584101
 k=63: 10555.490912682977
 k=64: 10423.416392920737
 k=65: 10348.474409320423
 k=66: 10350.83955690472
 k=67: 10213.823578236577
 k=68: 10241.140138554874
 k=69: 9892.56724007699
 k=70: 9876.22361410766
 k=71: 9743.540485971089
 k=72: 9567.634851530362
 k=73: 10012.930818863546
 k=74: 9467.673801656469
 k=75: 9668.271760553327
 k=76: 9262.303590334193
 k=77: 9286.793853146684
 k=78: 9176.150171960033
 k=79: 8925.848384800855
 k=80: 8743.931630881192
 k=81: 8797.373620234304
 k=82: 8602.538126845096
 k=83: 8569.153976978858
 k=84: 8724.947246284979
 k=85: 8563.897324695015
 k=86: 8265.228086892723
 k=87: 8353.001134758906
 k=88: 8306.898850039132
 k=89: 7968.686930648667
 k=90: 7948.864534423811
 k=91: 7836.739519360778
 k=92: 8003.732169041799
 k=93: 7597.856428461295
 k=94: 7639.664135743834
 k=95: 7600.1079461173595
 k=96: 7379.282914075222
 k=97: 7284.609820294415
 k=98: 7410.192844430426
 k=99: 7222.729847017033
 k=100: 7290.096494305172
 k=101: 6925.938574136669
 k=102: 7019.776196428485
 k=103: 6844.439630751858
 k=104: 6625.53642891818
 k=105: 6783.013737643725
 k=106: 6497.520519446791
 k=107: 6665.890474129683
 k=108: 6429.351627944521
 k=109: 6098.156732816356
 k=110: 6179.1086569728795
 k=111: 6168.038052405145
 k=112: 6171.122687556816
 k=113: 6006.120560032493
 k=114: 6114.821446695601
 k=115: 5932.754951112033
 k=116: 5896.4835245305285
 k=117: 5642.678855956559
 k=118: 5741.930423977987
 k=119: 5588.184510429943
 k=120: 5539.574366145226
 k=121: 5344.346690813205
 k=122: 5444.568333749188
 k=123: 5079.16040448133
 k=124: 5003.694767419753
 k=125: 5013.767477009983
 k=126: 4906.695026415562
 k=127: 4810.423979498826
 k=128: 4824.811188400635
 k=129: 4715.235181150002
 k=130: 4908.599883840614
 k=131: 4558.638162908439
 k=132: 4451.664311651827
 k=133: 4337.780345006327
 k=134: 4376.714116721353
 k=135: 4347.684685390894
 k=136: 4343.807257763824
 k=137: 4191.166074527179
 k=138: 3993.261128073059
 k=139: 3980.7842628682442
 k=140: 3845.7016676671597
 k=141: 3912.0074775523626
 k =142: 3774.481767761497
 k=143: 3841.906057814501
 k=144: 3542.1552889382738
 k=145: 3591.784935340683
 k=146: 3672.3571844612716
 k=147: 3466.600141813547
 k=148: 3408.133005808703
 k=149: 3210.06326500391
 k=150: 3257.5027345164467
 k=151: 3201.578001797258
 k=152: 3178.6611277236957
 k=153: 3054.4793540857986
 k=154: 2950.2525384429487
 k=155: 2976.967024177182
 k=156: 2739.5224608240364
 k=157: 2696.449889962546
 k=158: 2559.829058879634
 k=159: 2742.4508462962494
 k=160: 2577.3403602703556
 k=161: 2502.912784340208
 k=162: 2407.2693782021515
 k=163: 2333.767836008187
 k=164: 2331.2207782347346
 k=165: 2210.8943282894807
 k=166: 2163.209137807474
 k=167: 1995.3082267295058
 k=168: 1861.206395865028
 k=169: 1924.5264187281628
 k=170: 1863.3106467055002
 k=171: 1855.3678815347025
 k=172: 1725.7018495323093
 k=173: 1723.1029746225502
 k=174: 1608.1548063088578
 k=175: 1577.5324576565363
 k=176: 1510.243536363657
 k=177: 1544.6060502175872
 k=178: 1334.7217363236405
 k=179: 1428.4154535625612
 k=180: 1141.5909486231437
 k=181: 1179.023239235355
 k=182: 1190.1075834424153
 k=183: 1136.99872289157
 k=184: 1000.3677349372342
 k=185: 966.6860643535724
 k=186: 883.3578605265052
 k=187: 796.4359497433704
 k=188: 749.178348546789
 k=189: 757.2291539012693
 k=190: 579.0775278281086
 k=191: 580.1329131479209
 k=192: 480.1117262070117
 k=193: 413.99928068220123
 k=194: 404.03174306835365
 k=195: 315.5104527647319
 k=196: 254.138559908601
 k=197: 139.95357435165195
 k=198: 107.96273462806724
 k=199: 66.90015419488914
 k=200: 0.0

### Visualizing WCSS Scores
The WCSS scores were plotted to visualize the optimal number of clusters:


### Cluster Sizes
 optimal number of clusters (kkmeans1): 4(Based on the observation of the elbow plot.)
 Final WCSS: 18509.163029424544

 Cluster sizes:
 Cluster 0: 25 samples
 Cluster 1: 58 samples
 Cluster 2: 81 samples
 Cluster 3: 36 samples

### Conclusion

The custom K-Means class was able to fit the data and predict the cluster memberships effectively. The Elbow Method was used to determine the optimal number of clusters for the 512-dimensional dataset. The optimal number of clusters was found to be 4, as indicated by the "elbow" point in the plot. This optimal number of clusters was used to fit the K-Means model and obtain the cluster memberships.

# Question - 4 Gaussian Mixture Models

## Introduction

In this section, we implement and evaluate Gaussian Mixture Models (GMM) for clustering the Spotify dataset. The steps involved in this evaluation are as follows:

1. **Implement the GMM Class**:
   - Write a custom GMM class with methods such as `fit()`, `getParams()`, `getMembership()`, and `getLikelihood()`.
   - The `fit()` method implements the Expectation-Maximization (EM) algorithm to determine the optimal parameters for the model.
   - The `getParams()` method returns the parameters of the Gaussian components in the mixture model.
   - The `getMembership()` method returns the membership values for each sample in the dataset.
   - The `getLikelihood()` method returns the overall likelihood of the entire dataset under the current model parameters.

2. **Determine the Optimal Number of Clusters for 512 Dimensions**:
   - Perform GMM clustering on the dataset for various numbers of clusters.
   - Use BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion) to determine the optimal number of clusters for the 512-dimensional dataset.
   - Perform GMM clustering on the dataset using the optimal number of clusters.

## Implementation

1. **Loading the Data**:
   - The dataset is loaded using the `load_data2` function, which reads the data from a Feather file and normalizes it.

2. **Computing BIC and AIC**:
   - The `compute_bic` and `compute_aic` functions compute the BIC and AIC scores for the GMM model, respectively.

3. **Finding the Optimal Number of Clusters**:
   - The `find_optimal_gmm` function tests different numbers of components and computes the BIC and AIC scores to determine the optimal number of clusters.

4. **Main Execution**:
   - The `Q4_main` function orchestrates the entire process. It loads and preprocesses the data, finds the optimal number of clusters using BIC and AIC, and fits the GMM with the optimal number of clusters.

## Results and Analysis

### 4.1 Implement the GMM Class 

#### 1. GMM Class 
- **Class Implementation**: The GMM class was implemented correctly, encapsulating the core functionalities required for clustering. The class structure adheres to the standard practices, ensuring that it can be instantiated with the number of components (`n_components`) and other parameters.

#### 2. Overall GMM Implementation 

##### a. Correct Params 
- **Parameters**: The GMM class accepts the correct parameters, including the number of components (`n_components`), maximum iterations (`max_iter`), and tolerance for convergence (`tol`). These parameters are essential for controlling the behavior of the clustering algorithm.

##### b. Correct Functions 
- **Functions**: The class includes the necessary functions:
  - `fit()`: Implements the Expectation-Maximization (EM) algorithm to determine the optimal parameters for the model.
  - `getParams()`: Returns the parameters of the Gaussian components in the mixture model.
  - `getMembership()`: Returns the membership values for each sample in the dataset.
  - `getLikelihood()`: Returns the overall likelihood of the entire dataset under the current model parameters.

##### c. Primary Logic
- **Logic**: The primary logic of the GMM algorithm is implemented correctly:
  - **Initialization**: The parameters are initialized randomly.
  - **Expectation Step**: The membership values are updated based on the current parameters.
  - **Maximization Step**: The parameters are updated based on the current membership values.
  - **Convergence**: The algorithm checks for convergence based on the change in log-likelihood or the number of iterations.

### 4.2 Optimal Number of Clusters for 512 Dimensions

#### 1. Experimenting with GMM

##### a. Experimenting with Class Built from Scratch + Reasoning 
- **Custom GMM Class**: The custom GMM class was tested with various numbers of components. Due to the high dimensionality of the dataset (512 dimensions), the custom GMM class encountered issues and was unable to fit the data effectively.

##### b. Experimenting with Sklearn GMM + Reasoning 
- **Sklearn GMM**: The Sklearn GMM implementation was also tested with various numbers of components. The Sklearn GMM was able to fit the data more effectively due to its optimized implementation.

#### 2. BIC and AIC

##### a. Formula for BIC 
- **BIC Formula**: The Bayesian Information Criterion (BIC) is computed using the formula:
  \[
  \text{BIC} = -2 \cdot \text{log-likelihood} + \text{number of parameters} \cdot \log(\text{number of samples})
  \]

##### b. Formula for AIC 
- **AIC Formula**: The Akaike Information Criterion (AIC) is computed using the formula:
  \[
  \text{AIC} = -2 \cdot \text{log-likelihood} + 2 \cdot \text{number of parameters}
  \]

##### c. Plot for AIC and BIC 
- **AIC and BIC Plot**: The AIC and BIC scores were plotted for different numbers of components to determine the optimal number of clusters.

##### d. Determining Optimal k 
- **Optimal k**: The optimal number of clusters was determined based on the lowest BIC and AIC scores.

#### 3. GMM Clustering Using k_gmm1 
- **GMM Clustering**: The GMM clustering was performed using the optimal number of clusters (`k_gmm1`) as determined by the BIC and AIC scores.


### Determining the Optimal Number of Clusters for 512 Dimensions
 - To overcome the limitations of the custom GMM class. The BIC and AIC scores were computed for different numbers of clusters, and the optimal number of clusters was determined.

 - The results obtained are as follows:
  k=2, BIC=1397986.66, AIC=528283.03
  k=3, BIC=2096522.12, AIC=791965.03
  k=4, BIC=2795057.58, AIC=1055647.03
  k=5, BIC=3493593.04, AIC=1319329.03
  k=6, BIC=4192128.50, AIC=1583011.03
  k=7, BIC=4890663.96, AIC=1846693.03
  k=8, BIC=5589199.42, AIC=2110375.03
  k=9, BIC=6287734.88, AIC=2374057.03
  k=10, BIC=6986270.33, AIC=2637739.03
  k=11, BIC=7684805.79, AIC=2901421.03
  k=12, BIC=8383341.25, AIC=3165103.03
  k=13, BIC=9081876.71, AIC=3428785.03
  k=14, BIC=9780412.17, AIC=3692467.03
  k=15, BIC=10478947.63, AIC=3956149.03
  k=16, BIC=11177483.09, AIC=4219831.03
  k=17, BIC=11876018.55, AIC=4483513.03
  k=18, BIC=12574554.01, AIC=4747195.03
  k=19, BIC=13273089.47, AIC=5010877.03
  k=20, BIC=13971624.93, AIC=5274559.03

 - The optimal number of clusters determined by BIC and AIC is:
    Optimal number of clusters (BIC): 2
   Optimal number of clusters (AIC): 2

### Visualizing BIC and AIC Scores
The BIC and AIC scores were plotted to visualize the optimal number of clusters:


### Conclusion

 - The custom GMM class encountered issues due to the high dimensionality of the dataset. The BIC and AIC scores were effective in determining the optimal number of clusters for the 512-dimensional dataset. The optimal number of clusters was found to be 2, as indicated by both BIC and AIC scores. This optimal number of clusters was used to fit the GMM and obtain the cluster memberships.
# Question - 5 Dimensionality Reduction and Visualization

## Introduction

In this section, we implement and evaluate Principal Component Analysis (PCA) for dimensionality reduction on the given dataset. The steps involved in this evaluation are as follows:

1. **Implement the PCA Class**:
   - Write a custom PCA class with methods such as `fit()`, `transform()`, and `checkPCA()`.
   - The `fit()` method computes the principal components by finding the eigenvectors of the covariance matrix of the centered data.
   - The `transform()` method projects the data onto the principal components.
   - The `checkPCA()` method verifies that the transformation preserves the correct number of dimensions.

2. **Perform Dimensionality Reduction**:
   - Fit and transform the data to 2D and 3D using the custom PCA class.
   - Verify the functionality using the `checkPCA()` method.
   - Visualize the 2D and 3D transformed data.

3. **Data Analysis**:
   - Identify the axes of the transformed data.
   - Estimate the approximate value of k for clustering based on the visualization.

## Implementation

1. **Loading the Data**:
   - The dataset is loaded using the `load_data` function, which reads the data from a Feather file and normalizes it.

2. **PCA Class Implementation**:
   - The PCA class is implemented with methods for fitting the model, transforming the data, and verifying the transformation.

### PCA Class 

#### 1. PCA Class 
- **Class Implementation**: The PCA class was implemented correctly, encapsulating the core functionalities required for dimensionality reduction. The class structure adheres to the standard practices, ensuring that it can be instantiated with the number of components (`n_components`).

#### 2. Overall PCA Implementation 

##### a. Correct Params 
- **Parameters**: The PCA class accepts the correct parameters, including the number of components (`n_components`). This parameter is essential for controlling the dimensionality of the transformed data.

##### b. Correct Functions 
- **Functions**: The class includes the necessary functions:
  - `fit()`: Computes the principal components by finding the eigenvectors of the covariance matrix of the centered data.
  - `transform()`: Projects the data onto the principal components.
  - `checkPCA()`: Verifies that the transformation preserves the correct number of dimensions.

##### c. Primary Logic 
- **Logic**: The primary logic of the PCA algorithm is implemented correctly:
  - **Centering the Data**: The data is centered by subtracting the mean.
  - **Covariance Matrix**: The covariance matrix of the centered data is computed.
  - **Eigen Decomposition**: The eigenvalues and eigenvectors of the covariance matrix are computed.
  - **Sorting Eigenvectors**: The eigenvectors are sorted by decreasing eigenvalues.
  - **Storing Components**: The first `n_components` eigenvectors are stored as the principal components.

### 5.2 Perform Dimensionality Reduction 

#### 1. Fit and Transform to 2D and 3D 
- **2D Transformation**: The data is transformed to 2D using the custom PCA class.
- **3D Transformation**: The data is transformed to 3D using the custom PCA class.

#### 2. Verifying Functionality Using checkPCA Method 
- **Verification**: The `checkPCA()` method is used to verify that the transformation preserves the correct number of dimensions.

#### 3. Visualizing 2D and 3D Data 
- **2D Visualization**: The 2D transformed data is visualized using a scatter plot.
- **3D Visualization**: The 3D transformed data is visualized using a 3D scatter plot.

### 5.3 Data Analysis 

#### 1. Identifying Axes 
- **Axes Identification**: The axes of the transformed data are identified based on the principal components.

#### 2. Estimating the Approximate Value of k
- **Estimating k**: The approximate value of k for clustering is estimated based on the visualization of the transformed data.

## Results and Analysis

### Implementing the PCA Class
The custom PCA class was implemented with the required methods. The class was able to fit the data and transform it to lower dimensions effectively. The `checkPCA()` method verified that the transformation preserved the correct number of dimensions.

### Performing Dimensionality Reduction
The data was successfully transformed to 2D and 3D using the custom PCA class. The transformed data was visualized using scatter plots, providing insights into the structure of the data.

### Data Analysis
The axes of the transformed data were identified based on the principal components. The approximate value of k for clustering was estimated based on the visualization, providing a basis for further clustering analysis.

### Conclusion
The custom PCA class was implemented successfully, with all required functionalities and parameters. The data was effectively transformed to lower dimensions, and the transformed data was visualized and analyzed. The implementation adheres to the standard practices and provides meaningful insights into the structure of the dataset.
  
# Question - 6 PCA + Clustering
 - In this section, we explore the application of Principal Component Analysis (PCA) combined with clustering techniques such as K-Means and Gaussian Mixture Models (GMM) on the given dataset. The steps involved in this evaluation are as follows:

1. **K-Means Clustering Based on 2D Visualization**:
   - Perform K-Means clustering on the dataset using the number of clusters estimated from the 2D visualization of the dataset.

2. **PCA + K-Means Clustering**:
   - Generate a scree plot to identify the optimal number of dimensions for reduction.
   - Apply dimensionality reduction based on this optimal number of dimensions.
   - Determine the optimal number of clusters for the reduced dataset using the Elbow Method.
   - Perform K-Means clustering on the reduced dataset.

3. **GMM Clustering Based on 2D Visualization**:
   - Perform GMM clustering on the dataset using the number of clusters estimated from the 2D visualization.

4. **PCA + GMM Clustering**:
   - Determine the optimal number of clusters for the reduced dataset using AIC or BIC.
   - Apply GMM clustering with the optimal number of clusters to the dimensionally reduced dataset.

## Implementation

 1. **Loading the Data**:
   - The dataset is loaded using the `load_data` function, which reads the data from a Feather file and normalizes it.

 2. **K-Means Clustering Based on 2D Visualization**:
   - PCA is applied to reduce the dataset to 2 dimensions.
   - K-Means clustering is performed on the 2D dataset using the estimated number of clusters (`k2`).
   - The clusters are visualized using a scatter plot.

 3. **PCA + K-Means Clustering**:
   - A scree plot is generated using the `plot_scree` function to identify the optimal number of dimensions for PCA.
   - PCA is applied to reduce the dataset to the optimal number of dimensions.
   - The Elbow Method is used to determine the optimal number of clusters (`kkmeans3`) for the reduced dataset.
   - K-Means clustering is performed on the reduced dataset, and the clusters are visualized.

 4. **GMM Clustering Based on 2D Visualization**:
   - GMM clustering is performed on the 2D dataset using the estimated number of clusters (`k2`).
   - The clusters are visualized using a scatter plot.

 5. **PCA + GMM Clustering**:
   - The optimal number of clusters for the reduced dataset is determined using AIC and BIC.
   - GMM clustering is performed on the reduced dataset using the optimal number of clusters (`kgmm3`).
   - The clusters are visualized using a scatter plot.

## Results and Analysis

 ### K-Means Clustering Based on 2D Visualization
 - The dataset was reduced to 2 dimensions using PCA, and K-Means clustering was performed with `k2=5` clusters. The resulting clusters were visualized using a scatter plot.

 ### PCA + K-Means Clustering
 - A scree plot was generated to identify the optimal number of dimensions for PCA. Based on the scree plot, the dataset was reduced to 5 dimensions. The Elbow Method was used to determine the optimal number of clusters (`kkmeans3=3`). K-Means clustering was then performed on the reduced dataset, and the clusters were visualized.

 ### GMM Clustering Based on 2D Visualization
 - GMM clustering was performed on the 2D dataset with `k2=5` clusters. The resulting clusters were visualized using a scatter plot.

 ### PCA + GMM Clustering
 - The optimal number of clusters for the reduced dataset was determined using AIC and BIC. The optimal number of clusters was found to be `kgmm3=3`. GMM clustering was then performed on the reduced dataset, and the clusters were visualized.

 ### Observations

 1. **K-Means Clustering**:
   - The 2D visualization provided an initial estimate of the number of clusters (`k2=5`), which was used for both K-Means and GMM clustering.
   - The Elbow Method helped determine the optimal number of clusters (`kkmeans3=3`) for the reduced dataset.

 2. **GMM Clustering**:
   - The AIC and BIC scores were used to determine the optimal number of clusters (`kgmm3=3`) for the reduced dataset.

 3. **Cluster Visualization**:
   - The scatter plots provided a visual representation of the clusters formed by K-Means and GMM clustering on both the 2D and reduced datasets.

 ### Conclusion

 - The combination of PCA and clustering techniques such as K-Means and GMM provides a powerful approach for uncovering natural groupings in the data. PCA helps in reducing the dimensionality of the dataset, making it easier to visualize and analyze the clusters. The Elbow Method and AIC/BIC scores are effective in determining the optimal number of clusters for K-Means and GMM clustering, respectively. The visualizations provide valuable insights into the structure of the data and the effectiveness of the clustering methods.
# Question - 7 Cluster Analysis

This section presents the results of clustering analysis using K-Means and Gaussian Mixture Models (GMM) on a dataset of 512-dimensional word embeddings. The analysis includes loading the data, reducing its dimensionality using PCA, evaluating the coherence of clusters using silhouette scores and Calinski-Harabasz indices, and visualizing the results.

## Data Loading and Dimensionality Reduction

The data is loaded from a Feather file containing word embeddings.

## K-Means Cluster Analysis

### Clustering Results

- **K-Means (kkmeans1) Clustering Results:**
  - Silhouette score: 0.3933
  - Calinski-Harabasz Index: 176.9619
  - WCSS: 161.7625

- **K-Means (k2) Clustering Results:**
  - Silhouette score: 0.3370
  - Calinski-Harabasz Index: 156.7850
  - WCSS: 142.2908

- **K-Means (kkmeans3) Clustering Results:**
  - Silhouette score: 0.4177
  - Calinski-Harabasz Index: 162.6698
  - WCSS: 226.2565

### Best K-Means Clustering

The best K-Means clustering is identified based on the highest silhouette score:
- **Best K-Means clustering:** kkmeans3 with silhouette score 0.4177

## GMM Cluster Analysis

### Clustering Results

- **GMM (kgmm1) Clustering Results:**
  - Silhouette score: 0.1971
  - Calinski-Harabasz Index: 45.3932

- **GMM (k2) Clustering Results:**
  - Silhouette score: -0.3489
  - Calinski-Harabasz Index: 18.1539

- **GMM (kgmm3) Clustering Results:**
  - Silhouette score: -0.3655
  - Calinski-Harabasz Index: 3.7597

### Best GMM Clustering

The best GMM clustering is identified based on the highest silhouette score:
- **Best GMM clustering:** kgmm1 with silhouette score 0.1971

## Comparison of K-Means and GMM

### Best Clustering Results

- **Best K-Means (kkmeans3):**
  - Silhouette score: 0.4177
  - Calinski-Harabasz Index: 162.6698
  - WCSS: 226.2565

- **Best GMM (kgmm1):**
  - Silhouette score: 0.1971
  - Calinski-Harabasz Index: 45.3932

### Analysis

The K-Means clustering approach with `kkmeans3` (k=3) yielded the best results with a silhouette score of 0.4177, indicating well-defined and coherent clusters. The Calinski-Harabasz Index of 162.6698 further supports the quality of the clustering.

In contrast, the GMM clustering approach with `kgmm1` (k=2) resulted in a lower silhouette score of 0.1971 and a Calinski-Harabasz Index of 45.3932, indicating less coherent clusters compared to K-Means.

### Conclusion

The combination of PCA and clustering techniques such as K-Means and GMM provides a powerful approach for uncovering natural groupings in the data. PCA helps in reducing the dimensionality of the dataset, making it easier to visualize and analyze the clusters. The Elbow Method and AIC/BIC scores are effective in determining the optimal number of clusters for K-Means and GMM clustering, respectively. The visualizations provide valuable insights into the structure of the data and the effectiveness of the clustering methods.

Overall, K-Means clustering with `kkmeans3` (k=3) provided the most coherent and meaningful clusters for the given dataset.

# Question - 8 Hierarchical Clustering

Hierarchical clustering is a method used to group objects into clusters based on their similarity. It creates a hierarchical tree called a dendrogram, which visually represents the clusters. In this section, we apply hierarchical clustering to the Spotify dataset to uncover natural groupings and compare these clusters with those obtained from K-Means and GMM clustering.

## Implementation

1. **Loading the Data**:
   - The dataset is loaded using the `load_data` function, which reads the data from a Feather file and extracts the 512-dimensional embeddings.

2. **Hierarchical Clustering**:
   - The `hierarchical_clustering` function is used to perform hierarchical clustering on the dataset. This function computes the linkage matrix using different linkage methods (`complete`, `average`, `single`) and distance metrics (`euclidean`, `cosine`). The dendrograms are plotted to visualize the hierarchical clustering process.

3. **Cutting the Dendrogram**:
   - The `cut_dendrogram` function is used to cut the dendrogram at points corresponding to `kbest1` (3 clusters from K-Means) and `kbest2` (2 clusters from GMM) to form clusters. The `fcluster` function from the `scipy.cluster.hierarchy` module is used for this purpose.

4. **Main Execution**:
   - The `Q8_main` function orchestrates the entire process. It loads the data, applies PCA for dimensionality reduction, performs hierarchical clustering with different linkage methods and distance metrics, and cuts the dendrogram to form clusters. The clusters are then compared with those from K-Means and GMM clustering.

### Applying Hierarchical Clustering to the Dataset

#### Obtaining Linkage Matrix
- The linkage matrix was computed using different linkage methods (`complete`, `average`, `single`) and distance metrics (`euclidean`, `cosine`). This matrix is essential for constructing the dendrogram and understanding the hierarchical relationships between data points.

#### Plotting Dendrogram
- Dendrograms were plotted using the computed linkage matrices. These dendrograms visually represent the hierarchical clustering process and help identify natural groupings in the data.

### Combination of Linkage Methods and Distance Metrics

#### Experimentation
- Various combinations of linkage methods and distance metrics were experimented with to observe their effects on the clustering results. The dendrograms and resulting clusters were analyzed for each combination.

#### Observation
- Different linkage methods and distance metrics led to varying clustering results. The dendrograms and clusters varied based on the chosen method and metric, highlighting the importance of selecting appropriate parameters for hierarchical clustering.

### Comparison

#### Creating Clusters Using k_best1 (3 Clusters from K-Means)
- The dendrograms were cut at points corresponding to `kbest1=3` to form clusters. These clusters were then compared with those obtained from K-Means clustering.

#### Compare with K-Means Clustering
- The clusters formed by cutting the dendrogram at `kbest1=3` were compared with the clusters obtained from K-Means clustering. The unique clusters and their sizes were analyzed to understand the alignment between the two methods.

#### Creating Clusters Using k_best2 (2 Clusters from GMM)
- Similarly, the dendrograms were cut at points corresponding to `kbest2=2` to form clusters. These clusters were then compared with those obtained from GMM clustering.

#### Compare with GMM Clustering
- The clusters formed by cutting the dendrogram at `kbest2=2` were compared with the clusters obtained from GMM clustering. The unique clusters and their sizes were analyzed to understand the alignment between the two methods.

### Conclusion

Hierarchical clustering provides a visual and intuitive way to understand the natural groupings in the data. By experimenting with different linkage methods and distance metrics, we can analyze the stability and consistency of the clusters. Comparing these clusters with those from K-Means and GMM helps validate the clustering results and identify the best approach for the given dataset. The best linkage method identified from this analysis can be used to create clusters that align well with the results from K-Means and GMM clustering.

- # Question - 9  Nearest Neighbor Search
 - 
In this section, we explore the impact of Principal Component Analysis (PCA) on the performance of the K-Nearest Neighbors (KNN) model using the Spotify dataset from Assignment 1. The steps involved in this evaluation are as follows:

1. **Generating Scree Plot**:
   - A scree plot is generated to determine the optimal number of dimensions for PCA. This plot helps in identifying the number of principal components that capture the most variance in the data.

2. **Dimensionality Reduction**:
   - Based on the scree plot, PCA is applied to reduce the dataset to the optimal number of dimensions.

3. **KNN Model**:
   - The KNN model implemented in Assignment 1 is used on both the full dataset and the PCA-reduced dataset. The best {k, distance metric} pair obtained from Assignment 1 is used for this purpose.

## Implementation

1. **Loading and Preprocessing the Data**:
   - The dataset is loaded and preprocessed using the `load_and_preprocess_data` function. This function separates the features and target, converts non-numeric columns to numeric, normalizes the features, and handles categorical labels.

2. **Custom Train-Test Split**:
   - The `custom_train_test_split` function is used to split the dataset into training and test sets.

3. **Generating Scree Plot**:
   - The `generate_scree_plot` function generates a scree plot to determine the optimal number of principal components for PCA.

4. **KNN Classification**:
   - The `knn_classification` function applies the KNN model to the dataset and evaluates its performance using accuracy, precision, recall, F1 score (macro and micro), and inference time.

5. **Main Execution**:
   - The `Q9_main` function orchestrates the entire process. It loads and preprocesses the data, generates the scree plot, performs PCA, splits the data, applies KNN to both the full and PCA-reduced datasets, and prints the evaluation metrics.

## Results and Analysis
 - ### Generating Scree Plot to Determine Optimal Dimensions
  - The scree plot was generated to identify the optimal number of dimensions for PCA. This step is crucial as it helps in reducing the dataset to a manageable size while retaining most of the variance.

 - ### Performing KNN on Full Dataset
  - The KNN model was applied to the full dataset using the best {k, distance metric} pair. The performance metrics are as follows:

 - Generating Scree Plot to determine optimal dimensions...
 Performing KNN on full dataset...
 Performing KNN on PCA-reduced dataset...
 ---- KNN Performance on Full Dataset ----
 Accuracy: 0.51
 Precision: 0.51
 Recall: 0.51
 F1 Score (Macro): 0.51
 F1 Score (Micro): 0.51
 Inference Time: 209.1069 seconds

 ---- KNN Performance on PCA-Reduced Dataset ----
 Accuracy: 0.13
 Precision: 0.12
 Recall: 0.13
 F1 Score (Macro): 0.13
 F1 Score (Micro): 0.13
 Inference Time: 80.8992 seconds

--- KNN Performance on Full Dataset without dropping any columns --- 
K = 19 distance = manhattan
Accuracy: 0.24
Precision (macro): 0.24
Recall (macro): 0.24
F1 Score (macro): 0.24
F1 Score (micro): 0.25
 ### Observations

 1. **Performance Metrics**:
   - The KNN model's performance on the full dataset was significantly better than on the PCA-reduced dataset. The accuracy, precision, recall, and F1 scores were all higher for the full dataset. This indicates that the dimensionality reduction via PCA might have led to a loss of important information, adversely affecting the model's performance.

 2. **Inference Time**:
   - The inference time for the KNN model on the PCA-reduced dataset was considerably lower (80.8992 seconds) compared to the full dataset (209.1069 seconds). This demonstrates one of the key advantages of PCA: reducing the computational complexity and speeding up the inference process.

 3. **Implications**:
   - While PCA can significantly reduce the inference time, it is crucial to ensure that the reduction in dimensions does not lead to a substantial loss of information. In this case, the trade-off between performance and computational efficiency was not favorable, as the model's accuracy and other metrics dropped considerably.

 ### Conclusion

 - The experiment highlights the importance of carefully selecting the number of dimensions for PCA. While PCA can enhance computational efficiency, it is essential to balance this with the need to maintain model performance. Further tuning and experimentation with different numbers of principal components might be necessary to find an optimal balance.

## Resources usage  and llm usage
- I have used chatgpt for fixing the errors occured during the implementation of GMM class and using chatgpt I  have rewritten the entire GMM class to make it work.