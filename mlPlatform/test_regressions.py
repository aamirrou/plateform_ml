import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Paths
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'mlPlatform', 'calories_pred.csv')
models_dir = os.path.join(base_dir, 'mlPlatform', 'algoML', 'ml_models', 'regression')

# Model Paths
dt_model_path = os.path.join(models_dir, 'dt_reg_opt.pkl')
rf_model_path = os.path.join(models_dir, 'rf_regression.pkl')

print(f"Loading data from {csv_path}...")
try:
    df_r = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit(1)

# --- Preprocessing (Must match views.py) ---
print("Preprocessing data...")
le_gen_r = LabelEncoder()
df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])

# Define features and target
ordre_officiel = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X = df_r[ordre_officiel]
y = df_r['Calories']

# Scale features
scaler_r = StandardScaler()
X_scaled = scaler_r.fit_transform(X)

def test_model(name, path, X_test, y_test):
    print(f"\n--- Testing {name} ---")
    if not os.path.exists(path):
        print(f"Error: Model file {path} not found.")
        return

    try:
        model = joblib.load(path)
        print(f"Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        print("\nSample Predictions (First 5):")
        for i in range(5):
            print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {y_pred[i]:.2f}, Diff: {abs(y_test.iloc[i] - y_pred[i]):.2f}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

# Run Tests
test_model("Decision Tree Regressor", dt_model_path, X_scaled, y)
test_model("Random Forest Regressor", rf_model_path, X_scaled, y)

lin_model_path = os.path.join(models_dir, 'lin_reg_poly.pkl')
test_model("Polynomial Regression", lin_model_path, X_scaled, y)

svr_model_path = os.path.join(models_dir, 'svr_reg.pkl')
test_model("SVR", svr_model_path, X_scaled, y)
