import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_prep import load_data, basic_clean, feature_engineer, preprocess_for_model

print("🚀 Starting model training...")

# Load and prepare data
df = load_data('data/raw/loan_data.csv')
print(f"✅ Data loaded: {df.shape}")

df = basic_clean(df)
print(f"✅ Data cleaned: {df.shape}")

df = feature_engineer(df)
print(f"✅ Features engineered: {df.shape}")

print(f"✅ Target distribution:\n{df['loan_status'].value_counts()}")

# Split features and target
y = df['loan_status']
X_raw = df.drop(columns=['loan_status'])

print(f"✅ Features shape: {X_raw.shape}")
print(f"✅ Target shape: {y.shape}")

# Preprocess
X_processed, preprocess_objs = preprocess_for_model(X_raw, train_mode=True)
print(f"✅ Data preprocessed: {X_processed.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape}, {y_train.shape}")
print(f"✅ Test set: {X_test.shape}, {y_test.shape}")

# Train models
print("🤖 Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"✅ RandomForest ROC AUC: {rf_score:.4f}")

print("🤖 Training XGBoost...")
xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xg.fit(X_train, y_train)
xg_score = roc_auc_score(y_test, xg.predict_proba(X_test)[:, 1])
print(f"✅ XGBoost ROC AUC: {xg_score:.4f}")

# Select best model
if rf_score > xg_score:
    best_model = rf
    print(f"🎯 Selected RandomForest: {rf_score:.4f}")
else:
    best_model = xg  
    print(f"🎯 Selected XGBoost: {xg_score:.4f}")

# Save
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
joblib.dump(preprocess_objs, os.path.join(models_dir, "preprocess.joblib"))

print(f"💾 Models saved in: {models_dir}")
print("🎉 Training completed successfully!")