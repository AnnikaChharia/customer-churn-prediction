# Customer Churn Prediction

## Overview
A machine learning project to predict telecom customer churn 
using XGBoost with an interactive Streamlit dashboard.

## Live Demo
[Click here to view dashboard](your streamlit link)

## Results
- ROC-AUC: 0.81
- Accuracy: 85%

## Tech Stack
- Python, Pandas, Scikit-learn
- XGBoost
- SMOTE for class imbalance
- Streamlit for dashboard

## Project Structure
- `notebook.ipynb` — data analysis and model training
- `app.py` — streamlit dashboard
- `churn_model.pkl` — saved model

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py
