import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import os

def main():
    data_path = Path(__file__).parent.parent / "diabetes_preprocessing" / "train.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    os.makedirs("model", exist_ok=True)
    model_path = "model/model.pkl"
    import joblib
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    run_id = os.environ["MLFLOW_RUN_ID"]
    with open("mlflow_run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID saved: {run_id}")

if __name__ == "__main__":
    main()
