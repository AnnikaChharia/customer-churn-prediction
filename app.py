import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, accuracy_score)

# --- Page Config ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Load Data and Model ---
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
st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v2.png", width=80)
st.sidebar.title("Customer Churn")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📈 Model Performance",
    "🔍 Predict Customer"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.write(f"Total Records: {len(cc):,}")
st.sidebar.write(f"Features: {len(cc.columns)}")
st.sidebar.write(f"Churn Rate: {cc['churn'].mean()*100:.1f}%")

# ================================================
# PAGE 1 — OVERVIEW
# ================================================
if page == "🏠 Overview":

    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("Analyze customer churn patterns and predict churn risk using XGBoost.")
    st.markdown("---")

    # --- KPI Cards ---
    total = len(cc)
    churned = int(cc['churn'].sum())
    retained = total - churned
    churn_rate = churned / total * 100
    avg_revenue = cc['monthly_charges'].mean()
    total_revenue = cc['monthly_charges'].sum()
    lost_revenue = cc[cc['churn'] == 1]['monthly_charges'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("🚨 Churned", f"{churned:,}", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    col3.metric("✅ Retained", f"{retained:,}")
    col4.metric("💰 Avg Monthly Revenue", f"${avg_revenue:.0f}")
    col5.metric("📉 Revenue at Risk", f"${lost_revenue:,.0f}", delta_color="inverse")

    st.markdown("---")

    # --- Dataset Preview ---
    st.subheader("📋 Dataset Preview")
    st.dataframe(cc.head(10), use_container_width=True)
    st.markdown("---")

    # --- Pie Chart + Histogram ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Churn Distribution")
        counts = cc['churn'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            counts,
            labels=['No Churn', 'Churn'],
            autopct='%1.1f%%',
            colors=['#4C9BE8', '#E8534C'],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        ax.set_title('Churn vs No Churn', fontsize=13, fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Tenure Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(
            cc[cc['churn'] == 0]['tenure'], bins=30,
            alpha=0.7, color='#4C9BE8', label='No Churn'
        )
        ax.hist(
            cc[cc['churn'] == 1]['tenure'], bins=30,
            alpha=0.7, color='#E8534C', label='Churn'
        )
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Count')
        ax.set_title('Tenure Distribution by Churn', fontsize=13, fontweight='bold')
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # --- Bar Chart + Scatter Plot ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 4))
        contract_churn = cc.groupby(['contract', 'churn']).size().unstack()
        contract_churn.columns = ['No Churn', 'Churn']
        contract_churn.plot(
            kind='bar', ax=ax,
            color=['#4C9BE8', '#E8534C'],
            edgecolor='white'
        )
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Churn by Contract Type', fontsize=13, fontweight='bold')
        plt.xticks(rotation=30)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🔵 Monthly Charges vs Tenure")
        fig, ax = plt.subplots(figsize=(5, 4))
        scatter_no = cc[cc['churn'] == 0]
        scatter_yes = cc[cc['churn'] == 1]
        ax.scatter(
            scatter_no['tenure'], scatter_no['monthly_charges'],
            alpha=0.3, color='#4C9BE8', label='No Churn', s=10
        )
        ax.scatter(
            scatter_yes['tenure'], scatter_yes['monthly_charges'],
            alpha=0.3, color='#E8534C', label='Churn', s=10
        )
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Monthly Charges ($)')
        ax.set_title('Monthly Charges vs Tenure', fontsize=13, fontweight='bold')
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # --- Prediction Form ---
    st.subheader("🔍 Predict Customer Churn")
    st.markdown("Fill in the customer details below to predict churn probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 60.0)
            total_charges = st.number_input("Total Charges ($)", 20.0, 9000.0, float(tenure * monthly_charges))

        with col2:
            support_calls = st.slider("Support Calls", 0, 8, 1)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Cash", "Credit", "Debit", "UPI"])

        with col3:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No Service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])

        predict_btn = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    # --- Prediction Result ---
    if predict_btn:
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        # Build input
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

        input_dict[f'payment_method_{payment_method}'] = 1
        input_dict[f'contract_{contract}'] = 1
        input_dict[f'tech_support_{tech_support}'] = 1
        input_dict[f'online_security_{online_security}'] = 1

        if internet_service == 'No Service':
            input_dict['internet_service_'] = 1
        else:
            input_dict[f'internet_service_{internet_service}'] = 1

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if prob >= 0.7:
                st.error(f"🚨 High Churn Risk")
            elif prob >= 0.4:
                st.warning(f"⚠️ Medium Churn Risk")
            else:
                st.success(f"✅ Low Churn Risk")

            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(float(prob))

        with col2:
            st.markdown("**Customer Summary**")
            summary = {
                'Feature': ['Tenure', 'Monthly Charges', 'Total Charges',
                            'Support Calls', 'Contract', 'Payment Method',
                            'Internet Service', 'Tech Support', 'Online Security'],
                'Value': [f"{tenure} months", f"${monthly_charges:.2f}",
                          f"${total_charges:.2f}", support_calls,
                          contract, payment_method, internet_service,
                          tech_support, online_security]
            }
            st.table(pd.DataFrame(summary))

# ================================================
# PAGE 2 — MODEL PERFORMANCE
# ================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    col2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col3.metric("Test Samples", f"{len(y_test):,}")
    col4.metric("Features", f"{x_test.shape[1]}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#4C9BE8', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#4C9BE8')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10, 5))
    xgb.plot_importance(model, max_num_features=10, ax=ax, color='#4C9BE8')
    ax.set_title('Feature Importances', fontsize=13, fontweight='bold')
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=['No Churn', 'Churn'],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

# ================================================
# PAGE 3 — PREDICT CUSTOMER
# ================================================
elif page == "🔍 Predict Customer":
    st.title("🔍 Predict Individual Customer")
    st.markdown("---")

    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Usage Details**")
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 60.0)
            total_charges = st.number_input("Total Charges ($)", 20.0, 9000.0, float(tenure * monthly_charges))
            support_calls = st.slider("Support Calls", 0, 8, 1)

        with col2:
            st.markdown("**📄 Contract Details**")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Cash", "Credit", "Debit", "UPI"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No Service"])

        with col3:
            st.markdown("**🛠️ Services**")
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])

        predict_btn = st.form_submit_button("🔮 Predict", use_container_width=True)

    if predict_btn:
        st.markdown("---")

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

        input_dict[f'payment_method_{payment_method}'] = 1
        input_dict[f'contract_{contract}'] = 1
        input_dict[f'tech_support_{tech_support}'] = 1
        input_dict[f'online_security_{online_security}'] = 1

        if internet_service == 'No Service':
            input_dict['internet_service_'] = 1
        else:
            input_dict[f'internet_service_{internet_service}'] = 1

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(float(prob))

        with col2:
            if prob >= 0.7:
                st.error("🚨 High Churn Risk\nImmediate action recommended")
            elif prob >= 0.4:
                st.warning("⚠️ Medium Churn Risk\nMonitor this customer")
            else:
                st.success("✅ Low Churn Risk\nCustomer likely to stay")

        with col3:
            st.markdown("**Risk Breakdown**")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                [prob, 1 - prob],
                labels=['Churn', 'Stay'],
                colors=['#E8534C', '#4C9BE8'],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'white'}
            )
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Customer Input Summary**")
        summary = {
            'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Support Calls',
                        'Contract', 'Payment Method', 'Internet Service',
                        'Tech Support', 'Online Security'],
            'Value': [f"{tenure} months", f"${monthly_charges:.2f}", f"${total_charges:.2f}",
                      support_calls, contract, payment_method, internet_service,
                      tech_support, online_security]
        }
        st.table(pd.DataFrame(summary))
