# Loan Approval Prediction using Machine Learning

## Project Overview

This project predicts whether a loan application will be approved or rejected based on applicant details such as income, education, credit history, loan amount, and property area.  
Machine learning techniques are used to analyze past loan data and determine the probability of loan approval.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook / VS Code Notebook

## Dataset

The project uses the **Loan Approval Dataset**, which includes information about loan applicants such as:

- Gender
- Marital Status
- Education
- Applicant Income
- Loan Amount
- Credit History
- Property Area
- Loan Status (Target Variable)

## Project Workflow

1. **Data Loading**
   - Dataset loaded using Pandas.

2. **Data Cleaning**
   - Removed unnecessary columns.
   - Checked data types and structure.

3. **Handling Missing Values**
   - Missing values filled using appropriate techniques like mean, median, or mode.

4. **Encoding Categorical Variables**
   - Converted categorical features into numerical values using encoding techniques.

5. **Feature Scaling**
   - StandardScaler used to normalize numerical features.

6. **Model Training**
   - Logistic Regression model trained on the processed dataset.

7. **Model Evaluation**
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

## Model Performance

Algorithm Used: Logistic Regression  
Accuracy: ~80% (may vary slightly depending on the train-test split)

## How to Run the Project

1. Clone the repository
