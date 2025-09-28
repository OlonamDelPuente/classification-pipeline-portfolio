import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw/mushrooms.csv")

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("⚠️ Missing values found and dropped.")

    # Encode categorical features
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split features and target
    X = df.drop("class", axis=1)
    y = df["class"]

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X.to_csv("data/processed/features.csv", index=False)
    y.to_csv("data/processed/target.csv", index=False)

    print("✅ Preprocessing complete. Data saved to data/processed/")
    return X, y

if __name__ == "__main__":
    preprocess_data()
