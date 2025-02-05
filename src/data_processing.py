import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Adjust to look in the 'data' folder at project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
    file_path = os.path.join(base_dir, "data", "dataset.csv")  

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    df.dropna(inplace=True)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
