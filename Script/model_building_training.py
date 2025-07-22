import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def prepare_data(df, target_column):
    """Separate features and target, perform train-test split, apply scaling and handle imbalance."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Drop datetime or object columns if they exist
        X = X.select_dtypes(include=[np.number])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Resampling with SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        return X_train_resampled, X_test_scaled, y_train_resampled, y_test
    except Exception as e:
        print("Error in data preparation:", e)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression and Random Forest models."""
    try:
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print(f"\n{name} Results:")
            print("F1-Score:", round(f1_score(y_test, y_pred), 4))
            print("AUC-PR:", round(average_precision_score(y_test, y_pred), 4))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
    except Exception as e:
        print("Error during model training or evaluation:", e)
