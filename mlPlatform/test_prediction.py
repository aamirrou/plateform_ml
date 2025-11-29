import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'calories_pred.csv')
model_path = os.path.join(base_dir, 'algoML/ml_models/regression/svr_reg.pkl')

print(f"Loading data from {csv_path}...")
df_r = pd.read_csv(csv_path)

# Recreate logic from views.py
print("Recreating scaler logic...")
le_gen_r = LabelEncoder()
df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])

ordre_officiel = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_finale = df_r[ordre_officiel]

scaler_r = StandardScaler()
scaler_r.fit(df_finale)

# Load model
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# Test on the first 5 rows
print("\nTesting on first 5 rows:")
for i in range(5):
    row = df_finale.iloc[i]
    actual_calories = df_r.iloc[i]['Calories']
    
    # Prepare features
    features = row.values.reshape(1, -1)
    
    # Scale
    features_scaled = scaler_r.transform(features)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    
    print(f"Row {i}: Actual={actual_calories}, Predicted={pred:.4f}")

print("\nTest without scaling (just in case):")
for i in range(5):
    row = df_finale.iloc[i]
    features = row.values.reshape(1, -1)
    pred = model.predict(features)[0]
    print(f"Row {i}: Actual={actual_calories}, Predicted (No Scale)={pred:.4f}")
