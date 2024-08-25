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
