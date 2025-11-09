import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def basic_clean(df):
    """Minimal cleaning - handle numeric loan_status"""
    df = df.copy()
    
    # Clean object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
    
    # Handle loan_status - it's already numeric in your data
    if 'loan_status' in df.columns:
        # Convert to numeric and drop any problematic values
        df['loan_status'] = pd.to_numeric(df['loan_status'], errors='coerce')
        # Drop rows where loan_status became NaN during conversion
        df = df.dropna(subset=['loan_status'])
        # Convert to integer (0 or 1)
        df['loan_status'] = df['loan_status'].astype(int)
    
    return df

def feature_engineer(df):
    """Create derived features"""
    df = df.copy()
    
    # Debt-to-income ratio
    if 'person_income' in df.columns and 'loan_amnt' in df.columns:
        df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
    
    return df

def preprocess_for_model(df, train_mode=True, fitted_objects=None):
    """Preprocessing that creates consistent OHE columns"""
    df = df.copy()

    # Define numeric and categorical columns based on your data
    numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                   'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                   'credit_score']
    
    # Add engineered feature if it exists
    if 'income_to_loan_ratio' in df.columns:
        numeric_cols.append('income_to_loan_ratio')
    
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 
                       'loan_intent', 'previous_loan_defaults_on_file']
    
    # Only use columns that exist in dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    if train_mode:
        num_imputer = SimpleImputer(strategy='median')
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        X_num = pd.DataFrame(num_imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        X_cat = pd.DataFrame(ohe.fit_transform(df[categorical_cols]), 
                           columns=ohe.get_feature_names_out(categorical_cols))
        preprocessors = {'num_imputer': num_imputer, 'ohe': ohe, 
                        'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}
    else:
        num_imputer = fitted_objects['num_imputer']
        ohe = fitted_objects['ohe']
        numeric_cols = fitted_objects['numeric_cols']
        categorical_cols = fitted_objects['categorical_cols']
        
        X_num = pd.DataFrame(num_imputer.transform(df[numeric_cols]), columns=numeric_cols)
        X_cat = pd.DataFrame(ohe.transform(df[categorical_cols]), 
                           columns=ohe.get_feature_names_out(categorical_cols))
        preprocessors = fitted_objects

    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X, preprocessors