import joblib
import pandas as pd

# Chargez votre modèle suspect
model = joblib.load('algoML/ml_models/regression/rf_regression.pkl')

# Affiche les noms des colonnes mémorisées lors de l'entraînement
try:
    print("✅ L'ORDRE EXACT ATTENDU EST :")
    print(model.feature_names_in_)
except AttributeError:
    print("❌ Ce modèle n'a pas enregistré les noms de colonnes.")
    print("Essayez de mettre 'Duration' en premier dans la liste, c'est souvent ça.")