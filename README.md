# Used Cars Price Prediction - README

## Overview
This project implements a regression analysis on a dataset of used cars to predict their prices. It uses 15 popular machine learning models and compares their performances using metrics such as R², relative error, and RMSE. Some of the complex models have been optimized for better results.

## Table of Contents
1. **Features**  
   - Exploratory Data Analysis (EDA)  
   - Data Preprocessing and Feature Engineering  
   - Model Training and Hyperparameter Tuning  
   - Model Evaluation and Comparison  
   - Prediction and Insights  

2. **Usage**  
   - Instructions for running the notebook  
   - Dependencies  

3. **Models Included**  
   - Linear Regression  
   - Support Vector Machines (SVR and Linear SVR)  
   - Multi-Layer Perceptron Regressor (MLP)  
   - Stochastic Gradient Descent (SGD)  
   - Decision Tree and Random Forest Regressors  
   - XGBoost and LightGBM  
   - Gradient Boosting Regressor  
   - Ridge Regressor  
   - Bagging Regressor  
   - ExtraTrees Regressor  
   - AdaBoost Regressor  
   - Voting Regressor  

---

## Features
### 1. Dataset
- The dataset is downloaded and preprocessed to remove unnecessary or redundant columns and handle missing values. 
- Target variable: **price**  
- Features include year, manufacturer, condition, cylinders, fuel type, odometer reading, transmission type, drive type, vehicle type, and paint color.

### 2. Preprocessing
- Categorical features are encoded using `LabelEncoder`.
- Continuous features are scaled using `StandardScaler`.
- Data is split into training, validation, and testing sets.

### 3. Model Evaluation
- Models are evaluated using:
  - **R² score** for goodness of fit  
  - **Relative Error** for accuracy  
  - **RMSE** for prediction errors  

---

## Usage

### 1. Install Dependencies
Ensure the following Python libraries are installed:
- **Data Handling & Visualization**: `numpy`, `pandas`, `matplotlib`
- **Modeling**: `sklearn`, `xgboost`, `lightgbm`
- **Hyperparameter Tuning**: `hyperopt`

To install the dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm hyperopt
```

### 2. Run the Project
The main notebook is `main.ipynb`. Open it in Jupyter Notebook or JupyterLab, and execute the cells sequentially.

1. **Load the Dataset**: Ensure the dataset is available in the specified path. Modify the file path in the notebook if required.
2. **Explore and Preprocess Data**: The notebook performs EDA and data preparation.
3. **Train and Evaluate Models**: Run each model and observe its performance metrics.
4. **Predict Prices**: Use the best-performing model for predictions on unseen data.

---

## Results
The best-performing models based on RMSE, R², and relative error are:
- **LightGBM (LGBM)**  
- **Bagging Regressor**  
- **XGBoost (XGB)**  

### Visualizations
- R² scores, relative errors, and RMSE for all models are plotted for comparison.

---

## Contribution and Feedback
Comments and feedback are welcome to improve the implementation and results.