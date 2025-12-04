import pandas as pd
import numpy as np

try:
    df = pd.read_csv('mlPlatform/fitness_dataset.csv')
    print("Columns:", df.columns.tolist())
    
    # Clean like in views.py
    df['smokes'] = df['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
    
    # Check gender values before encoding
    print("\nGender unique values:", df['gender'].unique())
    
    # Basic stats
    print("\nStatistics:")
    print(df.describe())
    
    print("\nMean of features:")
    print(df.drop(columns=['is_fit', 'gender']).mean())
    
except Exception as e:
    print(e)
