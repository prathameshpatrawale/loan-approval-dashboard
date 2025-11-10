import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Set page config first
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("Loan Approval Prediction Dashboard")

@st.cache_resource
def load_assets():
    """Load model and preprocessing objects"""
    try:
        from src.predict import load_model
        model, preprocess_objs = load_model()
        background = None
        return model, preprocess_objs, background
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

# Load assets
model, preprocess_objs, background = load_assets()

# If model failed to load, show error and stop
if model is None:
    st.error("❌ Failed to load model. Please check that the model files exist.")
    st.stop()

st.sidebar.header("Enter Applicant Details")

# Input fields
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
person_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
person_education = st.sidebar.selectbox("Education", ["High School", "Graduate", "Postgraduate"])
person_income = st.sidebar.number_input("Applicant Income", min_value=0, value=4000)
person_emp_exp = st.sidebar.number_input("Employment Experience (years)", min_value=0, value=5)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["Own", "Rent", "Mortgage", "Other"])
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0, value=10000)
loan_intent = st.sidebar.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Venture", "Homeimprovement", "Debtconsolidation"])
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income = st.sidebar.number_input("Loan Percent of Income", min_value=0.0, value=0.3)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, value=10)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=700)
previous_loan_defaults_on_file = st.sidebar.selectbox("Previous Loan Default", ["Yes", "No"])

# Create input data
input_data = pd.DataFrame([{
    'person_age': person_age,
    'person_gender': person_gender,
    'person_education': person_education,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'person_home_ownership': person_home_ownership,
    'loan_amnt': loan_amnt,
    'loan_intent': loan_intent,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_loan_defaults_on_file
}])

# Debug section
if st.checkbox("Preview: Show input data"):
    st.write("Input data columns:", input_data.columns.tolist())
    st.write("Input data shape:", input_data.shape)
    st.write("Input data:", input_data)
    
    # Show model expected features if available
    if hasattr(model, 'feature_names_in_'):
        st.write("Model expected features:", model.feature_names_in_.tolist())

# Prediction button
if st.button("Predict Loan Approval"):
    try:
        from src.predict import predict_single
        
        # Make prediction
        prediction, probability = predict_single(input_data, model, preprocess_objs)

        # Display results
        st.subheader("Prediction Result")
        st.write("Approval Probability:", f"{round(probability * 100, 2)}%")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Denied")

        # Feature Importance Bar Chart
        try:
            st.subheader("🔍 Top 10 Most Important Features")
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': model.feature_names_in_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            # Create the bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(feature_importance)), 
                           feature_importance['importance'], 
                           color='skyblue', 
                           edgecolor='navy')
            
            # Customize the chart
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'], fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Features Influencing Loan Decisions', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', 
                        ha='left', va='center', fontsize=9, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Display the chart in Streamlit
            st.pyplot(fig)
            plt.close(fig)
            
            # Optional: Also show the data as a table
            with st.expander("📊 View Feature Importance Table"):
                display_df = feature_importance.copy()
                display_df['importance'] = display_df['importance'].round(6)
                st.dataframe(display_df.reset_index(drop=True))
            
        except Exception as e:
            st.warning(f" Feature importance visualization could not be generated: {e}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

# Add model info section
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"Model: XGBoost")
st.sidebar.write(f"Training ROC AUC: 0.9794")
st.sidebar.write(f"Target Distribution: 35K Rejected, 10K Approved")
