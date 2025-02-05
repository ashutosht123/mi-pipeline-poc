from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from data_processing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

dump(model, "src/model.pkl")
print("✅ Model trained and saved as model.pkl")
