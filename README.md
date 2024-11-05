
l





















# 🌟 Loan Approval Prediction Project 🌟

## 📖 Overview

This project aims to develop a predictive model that determines the likelihood of loan approval based on various features of loan applicants. By leveraging machine learning techniques, we analyze historical loan data to predict outcomes for new loan applications.


## 📁 Project Structure

```plaintext
Loan-approval-prediction/
│
├── 📓 LoanApprovalEDA.ipynb          # Jupyter Notebook for Exploratory Data Analysis
├── 📦 final_model.pkl                # Trained model for loan appproval predictions
├── 📦 loan_approval_model.pkl         # Model used for predictions
├── 📊 loan_approval_predictions.csv    # Predictions made on new data
├── 🧪 test_Y3wMUE5_7gLdaTN.csv       # Test dataset
└── 📚 train_u6lujuX_CVtuZ9i.csv      # Training dataset
```


## 📊 Data Description

### Training Data
The training dataset consists of 614 entries and the following columns:

| **Column**              | **Description**                                           |
|------------------------|-----------------------------------------------------------|
| **Loan_ID**            | Unique identifier for each loan                           |
| **Gender**             | Gender of the applicant                                   |
| **Married**            | Marital status of the applicant                           |
| **Dependents**         | Number of dependents                                      |
| **Education**          | Educational qualification                                 |
| **Self_Employed**      | Employment status of the applicant                        |
| **ApplicantIncome**     | Income of the applicant                                   |
| **CoapplicantIncome**   | Income of the coapplicant                                |
| **LoanAmount**         | Amount of loan applied for                                |
| **Loan_Amount_Term**   | Duration of the loan in months                           |
| **Credit_History**     | Credit history of the applicant                           |
| **Property_Area**      | Area of residence                                        |
| **Loan_Status**        | Approval status (Y/N)                                   |

### Test Data
The test dataset has the same structure as the training dataset but does not contain the **Loan_Status** column.

## 📈 Exploratory Data Analysis (EDA)

- Conducted statistical analysis to understand the distribution of features.
- Visualized the loan approval status distribution, highlighting class imbalance.
- Identified categorical and numerical columns for further processing.

![EDA Visualization](path/to/eda_visualization.png)

## 🔍 Data Preprocessing

1. **Missing Value Imputation**:
   - Imputed missing values in `LoanAmount`, `Loan_Amount_Term`, and `Credit_History` using appropriate strategies.

2. **Encoding Categorical Variables**:
   - Converted categorical features into numerical representations using one-hot encoding.

3. **Scaling**:
   - Scaled numerical features to normalize their distributions.

4. **Feature Selection**:
   - Dropped unnecessary columns such as `Loan_ID` before model training.

## 🏗️ Model Building

1. **Train-Test Split**: 
   - Split the training data into training and validation sets.

2. **Model Selection**: 
   - Evaluated various models, including Logistic Regression, Decision Tree, and Random Forest.

3. **Hyperparameter Tuning**:
   - Utilized GridSearchCV for hyperparameter tuning of the Random Forest model.

4. **Model Evaluation**:
   - Evaluated models based on accuracy, precision, recall, and F1-score.

## 📊 Results

| **Model**            | **Accuracy**  | **Key Parameters**                                       |
|---------------------|---------------|----------------------------------------------------------|
| Logistic Regression  | 79%           | -                                                        |
| Decision Tree        | 68%           | -                                                        |
| Random Forest        | 78%           | `max_depth`: None, `max_features`: 'sqrt', `min_samples_leaf`: 1, `min_samples_split`: 10, `n_estimators`: 50 |

**Final Model Accuracy**: 78.86%

## 🚀 Future Work

- **Address Class Imbalance**: Implement techniques to address class imbalance to improve model performance.
- **Feature Engineering**: Explore additional features that could enhance prediction accuracy.
- **Deployment**: Create a web application for real-time predictions of loan approvals.

## 🛠️ Usage

To use the trained model for making predictions, load the `final_model.pkl` and preprocess your input data similarly to the training data preprocessing steps.

```python
import pandas as pd
import joblib

# Load model
model = joblib.load('final_model.pkl')

# Preprocess new data
new_data_processed = preprocess(new_data)

# Make predictions
predictions = model.predict(new_data_processed)
```

## 📝 Acknowledgments

- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization

