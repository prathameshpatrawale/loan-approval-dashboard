import joblib
import pandas as pd
import os

def load_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    preprocess_objs = joblib.load(os.path.join(models_dir, "preprocess.joblib"))
    return model, preprocess_objs

def predict_single(raw_input_df, model, preprocess_objs):
    from src.data_prep import basic_clean, feature_engineer, preprocess_for_model

    # 1. Clean and feature engineer
    clean_df = basic_clean(raw_input_df)
    clean_df = feature_engineer(clean_df)
    
    # Remove target if present
    if 'loan_status' in clean_df.columns:
        clean_df = clean_df.drop(columns=['loan_status'])
    
    # 2. Ensure we have all required columns
    numeric_cols = preprocess_objs['numeric_cols']
    categorical_cols = preprocess_objs['categorical_cols']
    
    # Add missing columns
    for col in numeric_cols:
        if col not in clean_df.columns:
            clean_df[col] = 0
    
    for col in categorical_cols:
        if col not in clean_df.columns:
            clean_df[col] = 'Unknown'
    
    # Reorder columns to match training
    clean_df = clean_df[numeric_cols + categorical_cols]
    
    # 3. Preprocess (this creates OHE columns)
    X_processed, _ = preprocess_for_model(clean_df, train_mode=False, fitted_objects=preprocess_objs)
    
    # 4. Align with model's expected features
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in X_processed.columns:
            X_processed[feature] = 0
    
    X_processed = X_processed[expected_features]
    
    # 5. Predict
    proba = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(proba > 0.5)
    
    return prediction, proba
