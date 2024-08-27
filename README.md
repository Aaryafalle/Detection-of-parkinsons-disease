# Detection-of-parkinsons-disease

## **README**

**Project Title:** Parkinson's Disease Detection using SVM

**Purpose:**
This project aims to develop a machine learning model capable of predicting the presence of Parkinson's disease based on a given dataset. The model utilizes Support Vector Machines (SVM) for classification.

**Dataset:**
* **Name:** parkinsons.csv
* **Description:** Contains various features related to voice recordings of individuals, including:
    - MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz): Measures of voice fundamental frequency.
    - MDVP:Jitter, MDVP:Shimmer, MDVP:RAP, MDVP:PPQ, Jitter:DDP, Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA: Measures of voice variability.
    - NHR, HNR: Measures of noise and harmonic-to-noise ratio.
    - RPDE, DFA, spread1, spread2, D2, PPE: Measures of voice complexity and variability.
    - status: Target variable indicating the presence (1) or absence (0) of Parkinson's disease.

**Methodology:**

1. **Data Preprocessing:**
   - Load the dataset.
   - Explore the data for understanding its structure and characteristics.
   - Handle missing values if necessary.
   - Split the data into features (X) and target variable (Y).

2. **Feature Scaling:**
   - Standardize the features to ensure they have a similar scale, improving model performance.

3. **Data Splitting:**
   - Divide the data into training and testing sets to evaluate the model's generalization ability.

4. **Model Training:**
   - Create an SVM model with a linear kernel.
   - Train the model on the training data.

5. **Model Evaluation:**
   - Evaluate the model's performance on both training and testing sets using accuracy score.

6. **Prediction:**
   - Create a function to predict the presence or absence of Parkinson's disease for new input data.

**Algorithm: Support Vector Machines (SVM)**

SVM is a supervised machine learning algorithm that is particularly effective for classification tasks with complex decision boundaries. It works by finding the optimal hyperplane that separates data points of different classes. The hyperplane maximizes the margin between the two classes, leading to better generalization performance.

In this project, a linear kernel is used for SVM. This means the decision boundary is a linear hyperplane. Other kernels, such as radial basis function (RBF) or polynomial, can also be used for non-linear decision boundaries.

**Hyperparameter Tuning:**
SVM models have hyperparameters that can be tuned to improve performance. In this project, the default linear kernel is used without any hyperparameter tuning. However, you can experiment with different kernels and hyperparameters like `C` (regularization parameter) and `gamma` (kernel coefficient) to potentially achieve better results.

**Usage:**
1. Clone the repository or download the Python script.
2. Ensure the `parkinsons.csv` dataset is in the same directory as the script.
3. Run the script.
4. The model will be trained, evaluated, and ready to make predictions for new input data.

**Dependencies:**
* pandas
* numpy
* scikit-learn

**Note:**
* This project serves as a basic demonstration of using SVM for Parkinson's disease detection. For real-world applications, consider exploring other algorithms, feature engineering techniques, and model evaluation metrics.
* Medical diagnosis is a complex task, and this model should not be used as a substitute for professional medical advice.
**OUTPUTS**
  1] first five rows of the data
	name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	...	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
0	phon_R01_S01_1	119.992	157.302	74.997	0.00784	0.00007	0.00370	0.00554	0.01109	0.04374	...	0.06545	0.02211	21.033	1	0.414783	0.815285	-4.813031	0.266482	2.301442	0.284654
1	phon_R01_S01_2	122.400	148.650	113.819	0.00968	0.00008	0.00465	0.00696	0.01394	0.06134	...	0.09403	0.01929	19.085	1	0.458359	0.819521	-4.075192	0.335590	2.486855	0.368674
2	phon_R01_S01_3	116.682	131.111	111.555	0.01050	0.00009	0.00544	0.00781	0.01633	0.05233	...	0.08270	0.01309	20.651	1	0.429895	0.825288	-4.443179	0.311173	2.342259	0.332634
3	phon_R01_S01_4	116.676	137.871	111.366	0.00997	0.00009	0.00502	0.00698	0.01505	0.05492	...	0.08771	0.01353	20.644	1	0.434969	0.819235	-4.117501	0.334147	2.405554	0.368975
4	phon_R01_S01_5	116.014	141.781	110.655	0.01284	0.00011	0.00655	0.00908	0.01966	0.06425	...	0.10470	0.01767	19.649	1	0.417356	0.823484	-3.747787	0.234513	2.332180	0.410335
5 rows × 24 columns
  2]info about the dataset
  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 195 entries, 0 to 194
Data columns (total 24 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   name              195 non-null    object 
 1   MDVP:Fo(Hz)       195 non-null    float64
 2   MDVP:Fhi(Hz)      195 non-null    float64
 3   MDVP:Flo(Hz)      195 non-null    float64
 4   MDVP:Jitter(%)    195 non-null    float64
 5   MDVP:Jitter(Abs)  195 non-null    float64
 6   MDVP:RAP          195 non-null    float64
 7   MDVP:PPQ          195 non-null    float64
 8   Jitter:DDP        195 non-null    float64
 9   MDVP:Shimmer      195 non-null    float64
 10  MDVP:Shimmer(dB)  195 non-null    float64
 11  Shimmer:APQ3      195 non-null    float64
 12  Shimmer:APQ5      195 non-null    float64
 13  MDVP:APQ          195 non-null    float64
 14  Shimmer:DDA       195 non-null    float64
 15  NHR               195 non-null    float64
 16  HNR               195 non-null    float64
 17  status            195 non-null    int64  
 18  RPDE              195 non-null    float64
 19  DFA               195 non-null    float64
 20  spread1           195 non-null    float64
 21  spread2           195 non-null    float64
 22  D2                195 non-null    float64
 23  PPE               195 non-null    float64
dtypes: float64(22), int64(1), object(1)
memory usage: 36.7+ KB
3]null values
	0
name	0
MDVP:Fo(Hz)	0
MDVP:Fhi(Hz)	0
MDVP:Flo(Hz)	0
MDVP:Jitter(%)	0
MDVP:Jitter(Abs)	0
MDVP:RAP	0
MDVP:PPQ	0
Jitter:DDP	0
MDVP:Shimmer	0
MDVP:Shimmer(dB)	0
Shimmer:APQ3	0
Shimmer:APQ5	0
MDVP:APQ	0
Shimmer:DDA	0
NHR	0
HNR	0
status	0
RPDE	0
DFA	0
spread1	0
spread2	0
D2	0
PPE	0
dtype: int64

4] statistical measures
	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	...	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
count	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	...	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000	195.000000
mean	154.228641	197.104918	116.324631	0.006220	0.000044	0.003306	0.003446	0.009920	0.029709	0.282251	...	0.046993	0.024847	21.885974	0.753846	0.498536	0.718099	-5.684397	0.226510	2.381826	0.206552
std	41.390065	91.491548	43.521413	0.004848	0.000035	0.002968	0.002759	0.008903	0.018857	0.194877	...	0.030459	0.040418	4.425764	0.431878	0.103942	0.055336	1.090208	0.083406	0.382799	0.090119
min	88.333000	102.145000	65.476000	0.001680	0.000007	0.000680	0.000920	0.002040	0.009540	0.085000	...	0.013640	0.000650	8.441000	0.000000	0.256570	0.574282	-7.964984	0.006274	1.423287	0.044539
25%	117.572000	134.862500	84.291000	0.003460	0.000020	0.001660	0.001860	0.004985	0.016505	0.148500	...	0.024735	0.005925	19.198000	1.000000	0.421306	0.674758	-6.450096	0.174351	2.099125	0.137451
50%	148.790000	175.829000	104.315000	0.004940	0.000030	0.002500	0.002690	0.007490	0.022970	0.221000	...	0.038360	0.011660	22.085000	1.000000	0.495954	0.722254	-5.720868	0.218885	2.361532	0.194052
75%	182.769000	224.205500	140.018500	0.007365	0.000060	0.003835	0.003955	0.011505	0.037885	0.350000	...	0.060795	0.025640	25.075500	1.000000	0.587562	0.761881	-5.046192	0.279234	2.636456	0.252980
max	260.105000	592.030000	239.170000	0.033160	0.000260	0.021440	0.019580	0.064330	0.119080	1.302000	...	0.169420	0.314820	33.047000	1.000000	0.685151	0.825288	-2.434031	0.450493	3.671155	0.527367
8 rows × 23 columns

5]separating target and features of data
for x
  MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)  \
0        119.992       157.302        74.997         0.00784   
1        122.400       148.650       113.819         0.00968   
2        116.682       131.111       111.555         0.01050   
3        116.676       137.871       111.366         0.00997   
4        116.014       141.781       110.655         0.01284   
..           ...           ...           ...             ...   
190      174.188       230.978        94.261         0.00459   
191      209.516       253.017        89.488         0.00564   
192      174.688       240.005        74.287         0.01360   
193      198.764       396.961        74.904         0.00740   
194      214.289       260.277        77.973         0.00567   

     MDVP:Jitter(Abs)  MDVP:RAP  MDVP:PPQ  Jitter:DDP  MDVP:Shimmer  \
0             0.00007   0.00370   0.00554     0.01109       0.04374   
1             0.00008   0.00465   0.00696     0.01394       0.06134   
2             0.00009   0.00544   0.00781     0.01633       0.05233   
3             0.00009   0.00502   0.00698     0.01505       0.05492   
4             0.00011   0.00655   0.00908     0.01966       0.06425   
..                ...       ...       ...         ...           ...   
190           0.00003   0.00263   0.00259     0.00790       0.04087   
191           0.00003   0.00331   0.00292     0.00994       0.02751   
192           0.00008   0.00624   0.00564     0.01873       0.02308   
193           0.00004   0.00370   0.00390     0.01109       0.02296   
194           0.00003   0.00295   0.00317     0.00885       0.01884   

     MDVP:Shimmer(dB)  ...  MDVP:APQ  Shimmer:DDA      NHR     HNR      RPDE  \
0               0.426  ...   0.02971      0.06545  0.02211  21.033  0.414783   
1               0.626  ...   0.04368      0.09403  0.01929  19.085  0.458359   
2               0.482  ...   0.03590      0.08270  0.01309  20.651  0.429895   
3               0.517  ...   0.03772      0.08771  0.01353  20.644  0.434969   
4               0.584  ...   0.04465      0.10470  0.01767  19.649  0.417356   
..                ...  ...       ...          ...      ...     ...       ...   
190             0.405  ...   0.02745      0.07008  0.02764  19.517  0.448439   
191             0.263  ...   0.01879      0.04812  0.01810  19.147  0.431674   
192             0.256  ...   0.01667      0.03804  0.10715  17.883  0.407567   
193             0.241  ...   0.01588      0.03794  0.07223  19.020  0.451221   
194             0.190  ...   0.01373      0.03078  0.04398  21.209  0.462803   

          DFA   spread1   spread2        D2       PPE  
0    0.815285 -4.813031  0.266482  2.301442  0.284654  
1    0.819521 -4.075192  0.335590  2.486855  0.368674  
2    0.825288 -4.443179  0.311173  2.342259  0.332634  
3    0.819235 -4.117501  0.334147  2.405554  0.368975  
4    0.823484 -3.747787  0.234513  2.332180  0.410335  
..        ...       ...       ...       ...       ...  
190  0.657899 -6.538586  0.121952  2.657476  0.133050  
191  0.683244 -6.195325  0.129303  2.784312  0.168895  
192  0.655683 -6.787197  0.158453  2.679772  0.131728  
193  0.643956 -6.744577  0.207454  2.138608  0.123306  
194  0.664357 -5.724056  0.190667  2.555477  0.148569  

[195 rows x 22 columns]
for y
0      1
1      1
2      1
3      1
4      1
      ..
190    0
191    0
192    0
193    0
194    0
Name: status, Length: 195, dtype: int64

6]Accuracy score
-->For training 
Accuracy score of taining data: 0.8846153846153846
-->For testing
Accuracy score of testing data: 0.8717948717948718

