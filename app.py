import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("*Predict the likelihood of a customer leaving the bank based on their profile.*")

# --- Load Model & Scaler (with caching for speed) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# --- PREPROCESSING FUNCTION ---
def preprocess_input(data_dict):
    """
    Converts user inputs into a scaled DataFrame ready for model prediction.
    """
    # 1. Convert dict to DataFrame
    df = pd.DataFrame([data_dict])
    
    # 2. Encode Categorical Variables
    # ✅ FIXED: Changed to match your model's LabelEncoder (Alphabetical: France=0, Germany=1, Spain=2)
    geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}  
    gender_map = {'Female': 0, 'Male': 1}  # ✅ Correct (F=0, M=1)
    
    df['Geography'] = df['Geography'].map(geo_map)
    df['Gender'] = df['Gender'].map(gender_map)
    
    # 3. EXACT column order from your X_train (✅ Matches your output)
    feature_names = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    df = df[feature_names]
    
    # 4. Scale the numeric features
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

# --- SIDEBAR: User Inputs ---
st.sidebar.header("👤 Customer Profile")
st.sidebar.markdown("Adjust the customer's details below:")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0)

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Churn Risk", type="primary"):
    # Gather all inputs into a dictionary
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary
    }
    
    # Preprocess
    processed_df = preprocess_input(input_data)
    
    # Get Prediction Probability
    prob = model.predict_proba(processed_df)[0][1]  # Probability of churn
    prediction_label = "⚠️ High Churn Risk" if prob > 0.5 else "✅ Low Churn Risk"
    
    # --- Display Results ---
    st.divider()
    st.subheader("📊 Prediction Result")
    
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        # Big Metric
        st.metric(label="Churn Probability", value=f"{prob:.2%}")
        if prob > 0.5:
            st.error(prediction_label)
        else:
            st.success(prediction_label)
    
    # --- SHAP Explanation ---
    with result_col2:
        st.markdown("**Why did the model make this decision?**")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_df)
            
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                processed_df.iloc[0],
                matplotlib=True,
                show=False,
                figsize=(10, 3)
            )
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning("SHAP explanation could not be generated for this model type.")
            st.caption(f"Debug: {e}")
    
    # --- Disclaimer ---
    st.divider()
    st.caption("⚠️ **Disclaimer:** This is a demonstration prototype using synthetic data. Do not use for real financial decisions.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")