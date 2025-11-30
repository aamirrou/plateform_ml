import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Paths
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'mlPlatform', 'calories_pred.csv')
models_dir = os.path.join(base_dir, 'mlPlatform', 'algoML', 'ml_models', 'regression')

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# --- Preprocessing ---
print("Preprocessing data...")
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Features and Target
# Order must match views.py: ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X = df[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = df['Calories']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train Decision Tree ---
print("\nTraining Decision Tree Regressor...")
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)

# Evaluate
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree - MSE: {mse_dt:.4f}, R2: {r2_dt:.4f}")

# Save
dt_path = os.path.join(models_dir, 'dt_reg_opt.pkl')
joblib.dump(dt_reg, dt_path)
print(f"Saved Decision Tree to {dt_path}")

# --- Train Random Forest ---
print("\nTraining Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

# Save
rf_path = os.path.join(models_dir, 'rf_regression.pkl')
joblib.dump(rf_reg, rf_path)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# ... (Previous imports)

# --- Train Random Forest ---
# ... (Previous code)
print(f"Saved Random Forest to {rf_path}")

# --- Train Linear Regression (Polynomial) ---
print("\nTraining Linear Regression (Polynomial Degree 2)...")
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_reg.fit(X_train, y_train)

# Evaluate
y_pred_poly = poly_reg.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression - MSE: {mse_poly:.4f}, R2: {r2_poly:.4f}")

# Save
lin_path = os.path.join(models_dir, 'lin_reg_poly.pkl')
joblib.dump(poly_reg, lin_path)
from sklearn.svm import SVR

# ... (Previous imports)

# --- Train Linear Regression (Polynomial) ---
# ... (Previous code)
print(f"Saved Polynomial Regression to {lin_path}")

# --- Train SVR ---
print("\nTraining SVR...")
svr_reg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_reg.fit(X_train, y_train)

# Evaluate
y_pred_svr = svr_reg.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"SVR - MSE: {mse_svr:.4f}, R2: {r2_svr:.4f}")

# Save
svr_path = os.path.join(models_dir, 'svr_reg.pkl')
joblib.dump(svr_reg, svr_path)
print(f"Saved SVR to {svr_path}")

print("\nTraining Complete.")
