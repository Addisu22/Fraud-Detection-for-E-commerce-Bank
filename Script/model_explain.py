import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_model_with_shap(model, X_train, X_sample=None, max_display=10):
    """
    Generate SHAP Summary and Force plots to interpret a trained model.

    Parameters:
        model: Trained tree-based model (e.g., XGBoost, LightGBM, RandomForest)
        X_train (pd.DataFrame): Training feature set (after encoding/scaling)
        X_sample (int or pd.DataFrame): Number of rows or specific sample for local interpretation
        max_display (int): Maximum features to show in summary plot

    Returns:
        SHAP summary and force plots
    """
    try:
        print("Initializing SHAP Explainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Summary plot: Global interpretability
        print("Generating SHAP Summary Plot...")
        shap.summary_plot(shap_values, X_train, max_display=max_display, show=True)

        # Local force plot: Instance-level interpretation
        if X_sample is not None:
            if isinstance(X_sample, int):
                instance = X_train.iloc[X_sample]
            else:
                instance = X_sample
            print("Generating SHAP Force Plot for a sample...")
            shap.initjs()
            force_plot = shap.force_plot(explainer.expected_value, shap_values[X_sample], instance)
            return force_plot
        else:
            print("No sample provided for force plot.")
    except Exception as e:
        print("SHAP interpretation failed due to:", e)
