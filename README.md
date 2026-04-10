# Beta Bank Customer Churn Prediction (Machine Learning)

## Overview
This project focuses on predicting customer churn using supervised machine learning techniques. The goal is to identify customers likely to leave a bank, enabling proactive retention strategies and improved customer lifecycle management.

The workflow follows a full machine learning pipeline including data cleaning, feature engineering, class imbalance handling, model training, evaluation, and final model selection.

---

## Dataset
The dataset contains customer-level information such as:
- Demographics (age, gender, geography)
- Account details (balance, tenure, number of products)
- Activity indicators (credit card usage, active status)
- Target variable: `Exited` (1 = churn, 0 = retained)

---

## Key Steps

### Data Preparation
- Removed irrelevant identifiers (`RowNumber`, `CustomerId`, `Surname`)
- Applied one-hot encoding for categorical variables
- Handled missing and infinite values by replacing with 0 for consistency
- Split data into training, validation, and test sets using stratification

### Feature Scaling
- Applied `StandardScaler` for Logistic Regression models
- Maintained unscaled data for Random Forest (tree-based model)

---

## Handling Class Imbalance
Churn datasets are typically imbalanced. This project tested multiple strategies:

- Baseline model (no adjustment)
- Class weighting (`class_weight='balanced'`)
- Manual upsampling of minority class

---

## Models Used
- Logistic Regression (baseline)
- Logistic Regression with class balancing
- Logistic Regression with upsampling
- Random Forest Classifier (hyperparameter tuned)

---

## Evaluation Metrics
Models were evaluated using:
- **F1 Score** → balances precision and recall
- **AUC-ROC** → measures classification performance across thresholds
- Confusion Matrix
- ROC Curve

---

## Results

- Logistic Regression improved with class balancing and upsampling
- Random Forest outperformed all models after tuning depth and estimators
- Best model selected based on highest validation F1 score

### Final Model Performance
- **Best Model:** Random Forest Classifier  
- **F1 Score:** ~0.79–0.81  
- **AUC-ROC:** ~0.84–0.86  

---

## Key Insights
- Class imbalance significantly impacts model performance
- Tree-based models (Random Forest) handled feature interactions better
- Upsampling improved recall but required careful evaluation to avoid overfitting
- Proper model comparison is critical for selecting the best approach

---

## Tools & Technologies
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib

---

## Conclusion
This project demonstrates a complete machine learning workflow from raw data to production-ready model selection. It highlights the importance of preprocessing, handling imbalance, and evaluating multiple models to achieve strong predictive performance.
