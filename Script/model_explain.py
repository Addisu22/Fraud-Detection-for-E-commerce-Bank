import shap
import matplotlib.pyplot as plt
import pandas as pd

import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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


# Load data actual dataset
# df = pd.read_csv('creditcard.csv')
# Assume df is already loaded

def prepare_data(df, target_col='Class'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    print(f"=== {name} Evaluation ===")
    print("F1 Score:", f1)
    print("AUC-PR:", auc_pr)
    print("Confusion Matrix:\n", cm)
    return f1, auc_pr

def explain_model_with_shap(model, X_train, X_sample=None, max_display=10):
    try:
        print("Initializing SHAP Explainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        print("Generating SHAP Summary Plot...")
        shap.summary_plot(shap_values, X_train, max_display=max_display, show=True)

        if X_sample is not None:
            if isinstance(X_sample, int):
                instance = X_train.iloc[X_sample]
                shap_value_instance = shap_values[X_sample]
            else:
                instance = X_sample
                shap_value_instance = explainer.shap_values(instance)

            print("Generating SHAP Force Plot...")
            shap.initjs()
            display(shap.force_plot(explainer.expected_value, shap_value_instance, instance))
        else:
            print("No instance selected for force plot.")
    except Exception as e:
        print("SHAP interpretation failed due to:", e)


