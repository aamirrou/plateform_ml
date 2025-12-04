import pandas as pd

try:
    df = pd.read_csv('mlPlatform/fitness_dataset.csv')
    print("Min/Max stats:")
    print(df[['activity_index', 'nutrition_quality', 'sleep_hours']].agg(['min', 'max']))
except Exception as e:
    print(e)
