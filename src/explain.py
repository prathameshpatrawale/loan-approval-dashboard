import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def global_shap_summary(model, X_bg):
    """Creates a global SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_bg)
    plt.title("Global Feature Importance (SHAP Summary)")
    shap.summary_plot(shap_values, X_bg, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

def local_shap_explanation(model, X_input, X_bg):
    """Generates SHAP values for a single prediction."""
    explainer = shap.TreeExplainer(model, data=X_bg)
    shap_values = explainer(X_input)
    return shap_values
