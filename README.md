# Loan Approval Prediction using Machine Learning

## Overview

This project builds a machine learning model to predict whether a loan application will be approved based on applicant financial and demographic information.

Financial institutions evaluate several factors such as income, credit history, education, and employment status before approving loans. This project demonstrates how data preprocessing and machine learning can be used to automate this decision-making process.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## Dataset

The dataset contains information about loan applicants including:

* Gender
* Marital Status
* Dependents
* Education
* Self Employment Status
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

Target variable:

* **Loan_Status (Approved / Not Approved)**

---

## Project Workflow

### Data Preprocessing

* Removed unnecessary columns such as **Loan_ID**
* Cleaned categorical variables
* Handled missing values using **mode and median**
* Converted categorical variables into numerical format

### Feature Engineering

* Applied **Label Encoding**
* Applied **One-Hot Encoding** for Property Area

### Data Splitting

* Dataset split into **80% training data and 20% testing data**

### Feature Scaling

* Standardized numerical features using **StandardScaler**

### Model Training

* Implemented **Logistic Regression** for classification

---

## Model Performance

The Logistic Regression model achieved:

**Accuracy: ~80% on the test dataset**

Evaluation metrics used:

* Accuracy Score
* Classification Report
* Confusion Matrix

---

## Project Structure

```
loan-approval-prediction
│
├── loan_prediction.ipynb
├── loan_prediction.py
├── loan-data.csv
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Implement additional models such as Random Forest or Gradient Boosting
* Perform hyperparameter tuning
* Add visualizations for better insights
* Deploy the model as a web application

---

## Author

Anushka Shinde
