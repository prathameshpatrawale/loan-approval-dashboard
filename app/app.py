import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects with error handling
@st.cache_resource
def load_models():
    try:
        # Try loading different model versions
        model_paths = [
            "models/rf_model.pkl",
            "models/rf_model_light.pkl", 
            "models/lgb_model.pkl"
        ]
        
        model = None
        model_type = "No Model"
        
        for path in model_paths:
            if os.path.exists(path):
                model = pickle.load(open(path, "rb"))
                model_type = path.split("/")[-1].replace(".pkl", "").replace("_", " ").title()
                break
        
        if model is None:
            st.error("❌ No model files found. Please ensure model files are in the 'models' folder.")
            return None, None, "No Model"
        
        # Load label encoder if exists
        le_path = "models/le.pkl"
        le = None
        if os.path.exists(le_path):
            le = pickle.load(open(le_path, "rb"))
        
        return model, le, model_type
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, "Error"

# Load models
model, le, model_type = load_models()

# Header
st.markdown('<h1 class="main-header">📊 Walmart Sales Forecasting App</h1>', unsafe_allow_html=True)
st.markdown("""
**Predict weekly sales using machine learning models trained on historical Walmart data.**
This app helps forecast sales based on store characteristics, economic indicators, and temporal features.
""")

# Sidebar for navigation
st.sidebar.header("🔧 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Single Prediction", "Batch Prediction", "Project Insights"]
)

# Debug: Show what features the model expects
if model is not None:
    if hasattr(model, 'feature_names_in_'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Model Info")
        st.sidebar.write(f"**Model expects:** {list(model.feature_names_in_)}")

# Single Prediction Mode
if app_mode == "Single Prediction":
    st.sidebar.header("📋 Input Parameters")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Store & Department")
        store = st.number_input("Store ID", min_value=1, max_value=50, value=1)
        dept = st.number_input("Department ID", min_value=1, max_value=99, value=1)
        store_type = st.selectbox("Store Type", ["A", "B", "C"])
        size = st.number_input("Store Size", min_value=10000, max_value=220000, value=100000)
    
    with col2:
        st.subheader("Economic Indicators")
        temperature = st.number_input("Temperature (°F)", min_value=-10.0, max_value=120.0, value=75.0)
        fuel_price = st.number_input("Fuel Price ($)", min_value=1.0, max_value=5.0, value=2.5)
        cpi = st.number_input("CPI", min_value=100.0, max_value=300.0, value=210.0)
        unemployment = st.number_input("Unemployment Rate (%)", min_value=1.0, max_value=15.0, value=7.5)
    
    with col3:
        st.subheader("Temporal Features")
        is_holiday = st.selectbox("Is Holiday Week?", ["No", "Yes"])
        week = st.number_input("Week Number", min_value=1, max_value=53, value=26)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        year = st.number_input("Year", min_value=2010, max_value=2012, value=2011)

    # Encode categorical features
    type_encoded = {"A": 0, "B": 1, "C": 2}[store_type]
    is_holiday_encoded = 1 if is_holiday == "Yes" else 0

    # Create input dataframe with ALL possible features from your dataset
    input_data = pd.DataFrame({
        "Store": [store],
        "Dept": [dept],
        "IsHoliday": [is_holiday_encoded],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Type": [type_encoded],
        "Size": [size],
        "Week": [week],
        "Month": [month],
        "Year": [year],
        # MarkDown columns (set to 0 since we don't have inputs for them)
        "MarkDown1": [0.0],
        "MarkDown2": [0.0],
        "MarkDown3": [0.0],
        "MarkDown4": [0.0],
        "MarkDown5": [0.0]
    })

    # Try different feature combinations based on what the model was trained with
    feature_combinations = [
        ["Store", "Dept", "Week", "CPI", "Unemployment", "Size", "Type"],  # From your notebook
        ["Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment"],  # Original app features
        ["Store", "Dept", "Week", "Month", "Year", "CPI", "Unemployment", "Size", "Type", "IsHoliday"],  # Extended features
        list(input_data.columns)  # All features
    ]

    # Display input data for debugging
    with st.expander("🔍 View Input Data & Debug Info"):
        st.write("**All available features:**")
        st.dataframe(input_data)
        
        if model is not None and hasattr(model, 'feature_names_in_'):
            st.write(f"**Model expects:** {list(model.feature_names_in_)}")
            # Try to match model features
            model_features = list(model.feature_names_in_)
            matched_features = [col for col in model_features if col in input_data.columns]
            st.write(f"**Matched features:** {matched_features}")

    # Prediction button
    if st.button("🎯 Predict Weekly Sales", use_container_width=True):
        if model is not None:
            try:
                # Get the features the model was actually trained with
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    # Use only the features that exist in our input data
                    available_features = [col for col in model_features if col in input_data.columns]
                    prediction_data = input_data[available_features]
                    
                    st.sidebar.write(f"**Using features:** {available_features}")
                else:
                    # Fallback to the features from your notebook
                    prediction_data = input_data[["Store", "Dept", "Week", "CPI", "Unemployment", "Size", "Type"]]
                
                # Make prediction
                prediction = model.predict(prediction_data)[0]
                
                # Display prediction with nice formatting
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.success(f"🛒 **Predicted Weekly Sales:** ${prediction:,.2f}")
                
                # Additional insights based on prediction
                if prediction > 50000:
                    st.info("📈 **High Sales Alert:** This prediction indicates strong performance!")
                elif prediction < 10000:
                    st.warning("📉 **Low Sales Alert:** Consider promotional activities.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("""
                **Troubleshooting tips:**
                1. Check if the model was trained with the same features
                2. The model might be expecting different feature names
                3. Try retraining the model with consistent feature names
                """)
                
                # Show debug info
                st.write("**Debug Information:**")
                st.write(f"Model type: {type(model).__name__}")
                if hasattr(model, 'feature_names_in_'):
                    st.write(f"Model features: {list(model.feature_names_in_)}")
                st.write(f"Input features: {list(input_data.columns)}")
        else:
            st.error("❌ Model not loaded. Please check model files.")

# Batch Prediction Mode
elif app_mode == "Batch Prediction":
    st.header("📤 Batch Prediction")
    st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")
    
    # Show expected features based on model
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        st.info(f"**Expected features:** {', '.join(expected_features)}")
    
    # Download template with all possible features
    template_data = pd.DataFrame(columns=[
        "Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", 
        "CPI", "Unemployment", "Type", "Size", "Week", "Month", "Year"
    ])
    
    st.download_button(
        label="📥 Download CSV Template",
        data=template_data.to_csv(index=False),
        file_name="walmart_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process the data based on model requirements
            processed_df = df.copy()
            
            # Encode categorical columns if needed
            if "Type" in processed_df.columns and processed_df["Type"].dtype == 'object':
                type_mapping = {"A": 0, "B": 1, "C": 2}
                processed_df["Type"] = processed_df["Type"].map(type_mapping)
            
            if "IsHoliday" in processed_df.columns and processed_df["IsHoliday"].dtype == 'object':
                processed_df["IsHoliday"] = processed_df["IsHoliday"].apply(
                    lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0
                )
            
            # Add missing MarkDown columns with zeros if needed
            for md_col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
                if md_col not in processed_df.columns:
                    processed_df[md_col] = 0.0
            
            # Make predictions
            if st.button("🚀 Generate Predictions", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    try:
                        # Use model's expected features
                        if hasattr(model, 'feature_names_in_'):
                            model_features = list(model.feature_names_in_)
                            available_features = [col for col in model_features if col in processed_df.columns]
                            prediction_data = processed_df[available_features]
                        else:
                            # Fallback
                            prediction_data = processed_df[["Store", "Dept", "Week", "CPI", "Unemployment", "Size", "Type"]]
                        
                        predictions = model.predict(prediction_data)
                        result_df = df.copy()
                        result_df["Predicted_Weekly_Sales"] = predictions
                        
                        # Display results
                        st.subheader("✅ Prediction Results")
                        st.dataframe(result_df.style.format({"Predicted_Weekly_Sales": "${:,.2f}"}), 
                                   use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", len(result_df))
                        with col2:
                            st.metric("Average Sales", f"${result_df['Predicted_Weekly_Sales'].mean():,.2f}")
                        with col3:
                            st.metric("Max Sales", f"${result_df['Predicted_Weekly_Sales'].max():,.2f}")
                        with col4:
                            st.metric("Min Sales", f"${result_df['Predicted_Weekly_Sales'].min():,.2f}")
                        
                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="walmart_sales_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.write("**Available features in uploaded data:**", list(processed_df.columns))
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Project Insights Mode
elif app_mode == "Project Insights":
    st.header("🔍 Project Insights & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Findings")
        st.markdown("""
        **🎯 Business Insights:**
        - **Store Performance:** Store 20 has maximum weekly sales
        - **Seasonal Patterns:** Highest sales during November & December
        - **Holiday Impact:** Sales increase significantly during holiday weeks
        - **Store Size:** Larger stores (Type A) generally have higher sales
        
        **📈 Feature Correlations:**
        - Positive: Size, MarkDowns, Week/Month (seasonality)
        - Negative: Unemployment, Some store types
        """)
    
    with col2:
        st.subheader("💡 Recommendations")
        st.markdown("""
        1. **Inventory Planning:** Increase stock for holiday seasons (weeks 45-52)
        2. **Staff Management:** Hire temporary staff for peak seasons
        3. **Marketing:** Focus promotions during low-sales periods
        4. **Store Optimization:** Analyze underperforming stores (like Store 5)
        """)
    
    # Feature correlations from your data
    st.subheader("📈 Feature Correlations with Weekly Sales")
    correlations = {
        "Store Size": "0.29 (Strong positive)",
        "Store Type": "-0.22 (Negative)", 
        "MarkDown4": "0.06 (Weak positive)",
        "Week Number": "0.02 (Very weak positive)",
        "CPI": "-0.02 (Very weak negative)",
        "Unemployment": "-0.02 (Very weak negative)"
    }
    
    for feature, corr in correlations.items():
        st.write(f"**{feature}:** {corr}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Walmart Sales Forecasting App | Machine Learning Project | Domain: Retail Analytics"
    "</div>",
    unsafe_allow_html=True
)
