import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, accuracy_score)

# --- Load data and model ---
@st.cache_data
def load_data():
    return pd.read_csv('customer_churn_dataset.csv')

@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

@st.cache_data
def load_test_data():
    x_test = pd.read_csv('x_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()
    return x_test, y_test

cc = load_data()
model = load_model()
x_test, y_test = load_test_data()

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Performance", "Predict Customer"])

# ================================================
# PAGE 1 — OVERVIEW
# ================================================
if page == "Overview":
    st.title("📊 Customer Churn Dashboard")
    st.markdown("Telecom customer churn analysis using XGBoost")

    # KPI metrics
    total = len(cc)
    churned = cc['churn'].value_counts()[1]
    not_churned = cc['churn'].value_counts()[0]
    churn_rate = churned / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}")
    col3.metric("Retained", f"{not_churned:,}")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.divider()

    # Churn distribution and contract
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        counts = cc['churn'].value_counts().sort_index()
        ax.pie(counts, autopct='%1.1f%%', labels=['No Churn', 'Churn'],
               colors=['steelblue', 'tomato'])
        ax.set_ylabel('')
        st.pyplot(fig)

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots()
        contract_churn = cc.groupby(['contract', 'churn']).size().unstack()
        contract_churn.columns = ['No Churn', 'Churn']
        contract_churn.plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.divider()

    # Tenure and charges
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Tenure")
        fig, ax = plt.subplots()
        cc.groupby('churn')['tenure'].mean().plot(
            kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
        ax.set_ylabel('Months')
        st.pyplot(fig)

    with col2:
        st.subheader("Average Monthly Charges")
        fig, ax = plt.subplots()
        cc.groupby('churn')['monthly_charges'].mean().plot(
            kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
        ax.set_ylabel('Amount ($)')
        st.pyplot(fig)

    st.divider()

    # Support calls
    st.subheader("Support Calls Distribution by Churn")
    fig, ax = plt.subplots()
    cc[cc['churn'] == 0]['support_calls'].value_counts().sort_index().plot(
        kind='bar', ax=ax, color='steelblue', alpha=0.7, label='No Churn', position=1, width=0.4)
    cc[cc['churn'] == 1]['support_calls'].value_counts().sort_index().plot(
        kind='bar', ax=ax, color='tomato', alpha=0.7, label='Churn', position=0, width=0.4)
    ax.set_xlabel('Number of Support Calls')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

# ================================================
# PAGE 2 — MODEL PERFORMANCE
# ================================================
elif page == "Model Performance":
    st.title("📈 Model Performance")

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    col2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col3.metric("Test Samples", f"{len(y_test):,}")

    st.divider()

    col1, col2 = st.columns(2)

    # ROC Curve
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='steelblue', label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

    # Confusion Matrix
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax, colorbar=False)
        st.pyplot(fig)

    st.divider()

    # Feature Importance
    st.subheader("Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax)
    st.pyplot(fig)

    st.divider()

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=['No Churn', 'Churn'],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(2))

# ================================================
# PAGE 3 — PREDICT CUSTOMER
# ================================================
elif page == "Predict Customer":
    st.title("🔍 Predict Customer Churn")
    st.markdown("Enter customer details to get churn probability")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 60.0)
        total_charges = st.number_input("Total Charges ($)", 20.0, 9000.0, float(tenure * monthly_charges))
        support_calls = st.slider("Support Calls", 0, 8, 1)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Cash", "Credit", "Debit", "UPI"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No Service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    if st.button("Predict Churn", type="primary"):

        # Build input with all 18 columns set to 0
        input_dict = {
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'support_calls': support_calls,
            'payment_method_Cash': 0,
            'payment_method_Credit': 0,
            'payment_method_Debit': 0,
            'payment_method_UPI': 0,
            'contract_Month-to-month': 0,
            'contract_One year': 0,
            'contract_Two year': 0,
            'internet_service_': 0,
            'internet_service_DSL': 0,
            'internet_service_Fiber': 0,
            'tech_support_No': 0,
            'tech_support_Yes': 0,
            'online_security_No': 0,
            'online_security_Yes': 0
        }

        # Set the correct one-hot columns to 1
        input_dict[f'payment_method_{payment_method}'] = 1
        input_dict[f'contract_{contract}'] = 1
        input_dict[f'tech_support_{tech_support}'] = 1
        input_dict[f'online_security_{online_security}'] = 1

        # Handle internet service
        if internet_service == 'No Service':
            input_dict['internet_service_'] = 1
        else:
            input_dict[f'internet_service_{internet_service}'] = 1

        input_df = pd.DataFrame([input_dict])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if prob >= 0.7:
            st.error(f"🚨 High Churn Risk: {prob:.1%} probability")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Churn Risk: {prob:.1%} probability")
        else:
            st.success(f"✅ Low Churn Risk: {prob:.1%} probability")

        st.progress(float(prob))

        # Show input summary
        st.divider()
        st.subheader("Customer Summary")
        summary = {
            'Tenure': f"{tenure} months",
            'Monthly Charges': f"${monthly_charges:.2f}",
            'Total Charges': f"${total_charges:.2f}",
            'Support Calls': support_calls,
            'Contract': contract,
            'Payment Method': payment_method,
            'Internet Service': internet_service,
            'Tech Support': tech_support,
            'Online Security': online_security
        }
        st.table(pd.DataFrame(summary.items(), columns=['Feature', 'Value']))
