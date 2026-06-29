import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- Load data and model ---
@st.cache_data
def load_data():
    cc = pd.read_csv('customer_churn_dataset.csv')
    return cc

@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

cc = load_data()
model = load_model()

# --- Rebuild x_test, y_test (same pipeline as notebook) ---
@st.cache_data
def get_test_data():
    df = cc.copy()
    df['churn'] = df['churn'].map({'No': 0, 'Yes': 1})
    df['internet_service'] = df['internet_service'].fillna('No Service')

    X = df[['tenure', 'monthly_charges', 'total_charges', 'support_calls']]
    categorical_cols = ['payment_method', 'contract', 'internet_service', 'tech_support', 'online_security']
    df_encoded = pd.get_dummies(df[categorical_cols], columns=categorical_cols, dtype=int, drop_first=True)
    X = pd.concat([X, df_encoded], axis=1)
    Y = df['churn']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    return x_test, y_test, X.columns.tolist()

x_test, y_test, feature_cols = get_test_data()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Performance", "Predict Customer"])

# ================================================
# PAGE 1 — OVERVIEW
# ================================================
if page == "Overview":
    st.title("Customer Churn Dashboard")
    st.markdown("Telecom customer churn analysis using XGBoost")

    # KPI metrics
    total = len(cc)
    churned = cc['churn'].value_counts()['Yes']
    churn_rate = churned / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.divider()

    # Churn distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        cc['churn'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Churn', 'Churn'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots()
        cc.groupby(['contract', 'churn']).size().unstack().plot(kind='bar', ax=ax)
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.divider()

    # Tenure and charges comparison
    st.subheader("Churners vs Non-Churners")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        cc.groupby('churn')['tenure'].mean().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_title('Average Tenure')
        ax.set_ylabel('Months')
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        cc.groupby('churn')['monthly_charges'].mean().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_title('Average Monthly Charges')
        ax.set_ylabel('Amount')
        plt.xticks(rotation=0)
        st.pyplot(fig)

# ================================================
# PAGE 2 — MODEL PERFORMANCE
# ================================================
elif page == "Model Performance":
    st.title("Model Performance")

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # Metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    col2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

    st.divider()

    col1, col2 = st.columns(2)

    # ROC Curve
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
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
        disp.plot(ax=ax)
        st.pyplot(fig)

    # Feature Importance
    st.subheader("Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax)
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

# ================================================
# PAGE 3 — PREDICT SINGLE CUSTOMER
# ================================================
elif page == "Predict Customer":
    st.title("Predict Customer Churn")
    st.markdown("Enter customer details to get churn probability")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 20.0, 120.0, 60.0)
        total_charges = st.number_input("Total Charges", 20.0, 9000.0, float(tenure * monthly_charges))
        support_calls = st.slider("Support Calls", 0, 8, 1)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Cash", "Credit", "Debit", "UPI"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No Service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    if st.button("Predict Churn", type="primary"):
        # Build input matching training feature columns
        input_dict = {col: 0 for col in feature_cols}

        input_dict['tenure'] = tenure
        input_dict['monthly_charges'] = monthly_charges
        input_dict['total_charges'] = total_charges
        input_dict['support_calls'] = support_calls

        # Map selectbox values to one-hot columns
        # (drop_first removes: contract_Month-to-month, payment_method_Cash,
        #  internet_service_DSL, tech_support_No, online_security_No)
        col_map = {
            f'contract_{contract}': 1,
            f'payment_method_{payment_method}': 1,
            f'internet_service_{internet_service}': 1,
            f'tech_support_{tech_support}': 1,
            f'online_security_{online_security}': 1,
        }
        for col, val in col_map.items():
            if col in input_dict:
                input_dict[col] = val

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if prob >= 0.5:
            st.error(f"⚠️ High Churn Risk: {prob:.1%} probability")
        else:
            st.success(f"✅ Low Churn Risk: {prob:.1%} probability")

        st.progress(float(prob))
