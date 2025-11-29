import joblib
import pandas as pd
import os

base_path = 'algoML/ml_models/regression'
models = ['rf_regression.pkl', 'dt_reg_opt.pkl', 'lin_reg_poly.pkl', 'svr_reg.pkl', 'xgb_reg.pkl']

for m in models:
    path = os.path.join(base_path, m)
    print(f"Checking {m}...")
    try:
        model = joblib.load(path)
        print(f"Type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Features: {list(model.feature_names_in_)}")
        else:
            print("No feature_names_in_ found.")
    except Exception as e:
        print(f"Error loading {m}: {e}")
    print("-" * 20)
