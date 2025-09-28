import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def train_and_evaluate():
    # Load processed data
    X = pd.read_csv("data/processed/features.csv")
    y = pd.read_csv("data/processed/target.csv").values.ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

        # Save model
        joblib.dump(model, f"models/{name}.pkl")
        print(f"✅ {name} saved to models/{name}.pkl")

if __name__ == "__main__":
    train_and_evaluate()
