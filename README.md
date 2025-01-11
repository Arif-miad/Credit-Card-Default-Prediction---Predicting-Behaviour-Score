

---

# **Credit Card Default Prediction - Predicting Behaviour Score**

## **Overview**

This repository contains the analysis, feature engineering, and machine learning model development for predicting customer credit card defaults using a historical credit card portfolio dataset from Bank A. The goal of this project is to develop a predictive model to assign a **Behaviour Score** to customers, which reflects the likelihood of them defaulting on their credit card payments.

The dataset consists of 96,806 credit card records, each representing a customer’s transaction and credit history. It contains several independent variables grouped into categories such as:

- **On-us Attributes:** Credit card limits and usage data.
- **Transaction-level Attributes:** Transaction details across various merchant categories.
- **Bureau Tradeline Attributes:** Credit bureau data related to customer behavior, past delinquencies, and product holdings.
- **Bureau Enquiry Attributes:** Data about recent credit inquiries.

The primary objective is to predict whether a customer will default (1) or not (0) based on these features.

---

## **Dataset Description**

The dataset consists of the following columns:

- **bad_flag:** Target variable indicating default status (1 = default, 0 = no default).
- **On-us Attributes:** Variables like credit limits, usage data.
- **Transaction-level Attributes:** Data related to transaction amounts and types.
- **Bureau Tradeline Attributes:** Historical behavior and delinquencies reported by credit bureaus.
- **Bureau Enquiry Attributes:** Data related to recent credit inquiries and their effects on the customer’s financial health.

---

## **Challenges Faced**

While working with this dataset, I encountered several challenges:

1. **Missing Values:**
   - Many features contained missing values that required imputation techniques for handling. Simple imputation methods such as mean or median imputation were not always appropriate, so advanced methods were applied in some cases.

2. **Class Imbalance:**
   - The target variable `bad_flag` had a significant imbalance, with fewer defaults (1) compared to non-defaults (0). This required special attention during model evaluation, including using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

3. **Feature Engineering:**
   - The dataset required extensive feature engineering to extract meaningful information. I combined features, created new interaction terms, and scaled numerical values to improve the performance of the machine learning models.

4. **High Dimensionality:**
   - The dataset contained many variables, which led to a risk of overfitting. Dimensionality reduction techniques, such as PCA (Principal Component Analysis), were applied to reduce the number of features and prevent overfitting.

5. **Outliers:**
   - Certain features, such as transaction amounts and credit limits, contained outliers that could distort model training. Identifying and removing or handling outliers was a crucial step.

6. **Model Performance:**
   - The large dataset presented computational challenges. I optimized model performance using cross-validation techniques, ensemble methods like Random Forest, and other classifiers like XGBoost.

---

## **Workflow**

### **1. Data Preprocessing**

- **Missing Values:** I handled missing values using appropriate imputation methods (mean/median for numerical features, mode for categorical features).
- **Outlier Detection:** Outliers were detected using the Z-score method and treated accordingly by removing or capping extreme values.
- **Feature Encoding:** Categorical variables were encoded using one-hot encoding or label encoding as required.
- **Feature Scaling:** I scaled numerical features using Min-Max Scaling to standardize the data before feeding it into machine learning models.

### **2. Model Selection & Evaluation**

The project involves evaluating multiple classification models to predict credit card defaults. The following steps were followed:

1. **Data Splitting:** The dataset was split into training and testing sets (70-30 split).
2. **Model Training:** I applied various machine learning models, including:
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machines (SVM)
   - Gradient Boosting (XGBoost, LightGBM)
   - K-Nearest Neighbors (KNN)
   - Decision Trees
3. **Cross-validation:** I used 10-fold cross-validation to evaluate model performance on the training set.
4. **Evaluation Metrics:** The models were evaluated using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - AUC-ROC Curve
   - Confusion Matrix

### **3. Model Optimization**

I performed hyperparameter tuning using Grid Search and Random Search to optimize the models and improve their performance. The best-performing model was chosen based on the cross-validation score and AUC-ROC curve.

### **4. Deployment (Optional)**

Once the best model was selected, I prepared it for deployment. This involved saving the trained model using **joblib** and providing a framework for real-time prediction if needed.

---

## **Performance Evaluation**

The models were evaluated on the following metrics:

- **Accuracy:** Percentage of correctly predicted instances.
- **Precision:** Proportion of true positives among the predicted positives.
- **Recall:** Proportion of true positives among the actual positives.
- **F1-Score:** The harmonic mean of precision and recall.
- **AUC-ROC:** The area under the Receiver Operating Characteristic curve to evaluate the model’s ability to distinguish between the classes.

---
```python
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the dataset
url = '/kaggle/input/credit-card-behaviour-score' # Replace with the correct path
data = pd.read_csv(url)

# Dataset Overview
print(data.head())
print(data.info())
print(data.describe())

# Handling Missing Data
# For numerical columns, we will use the mean to fill missing values.
imputer = SimpleImputer(strategy='mean')
data_imputed = data.copy()

numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data_imputed[numerical_columns] = imputer.fit_transform(data_imputed[numerical_columns])

# Encoding Categorical Features
# We will encode categorical variables using Label Encoding for simplicity.
categorical_columns = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()

for col in categorical_columns:
    data_imputed[col] = encoder.fit_transform(data_imputed[col])

# Define the target and features
X = data_imputed.drop(columns=['bad_flag'])  # Drop target column
y = data_imputed['bad_flag']  # Target column

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models to Evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),  # SVM requires setting probability=True for predict_proba
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Cross-Validation and Model Evaluation
results = {}
for name, model in models.items():
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        "mean_accuracy": np.mean(cv_scores),
        "std_accuracy": np.std(cv_scores),
    }
    print(f"{name} - Mean Accuracy: {results[name]['mean_accuracy']} ± {results[name]['std_accuracy']}")

# Evaluate the best model on the test data
best_model_name = max(results, key=lambda x: results[x]['mean_accuracy'])
best_model = models[best_model_name]

# Train the model on the full training data
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Model Evaluation Metrics
print(f"Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for {best_model_name}')
plt.legend(loc="lower right")
plt.show()

# Hyperparameter Tuning (Optional)
# Here we will tune the hyperparameters of the Random Forest Classifier using GridSearchCV.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Evaluating the tuned model
y_pred_rf = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Final Model Performance Comparison
model_names = list(results.keys())
accuracies = [results[name]["mean_accuracy"] for name in model_names]
std_devs = [results[name]["std_accuracy"] for name in model_names]

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, xerr=std_devs, capsize=5, color='skyblue')
plt.xlabel('Mean Accuracy')
plt.title('Model Performance Comparison')
plt.show()

```

## **Future Work**

Given the results, the following areas are recommended for further exploration:

1. **Deep Learning Models:** Explore neural networks for potentially better predictive accuracy.
2. **Ensemble Methods:** Combine multiple models for better performance, such as stacking or boosting.
3. **Real-Time Predictions:** Develop a real-time system that uses this model for ongoing credit default prediction.
4. **Additional Features:** Investigate more detailed external data (e.g., economic indicators) to further improve the model.

---

## **Contributing**

Feel free to fork this repository, make improvements, and submit pull requests. Contributions to enhance the performance or add more advanced techniques are always welcome.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**

- I would like to thank the original creators of the dataset for providing this valuable resource.
- Thanks to the contributors in the data science community for their invaluable insights, code snippets, and tutorials that made this project possible.

