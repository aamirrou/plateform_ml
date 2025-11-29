# # # from django.shortcuts import render
# # # import os
# # # import joblib

# # # # Create your views here.
# # # def index(request):
# # #     return render(request, 'index.html')
# # # def regLog_details(request):
# # #     return render(request, 'regLog_details.html')
# # # def regLog_atelier(request):
# # #     return render(request, 'regLog_atelier.html')
# # # def regLog_tester(request):
# # #     return render(request, 'vehicles_form.html')
# # # def load_models(name):
# # #     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # #     models_dir = os.path.join(base_dir,'models_ai')
# # #     model_path = os.path.join(models_dir, name)
# # #     ml_model = joblib.load(model_path)
# # #     return ml_model
# # # def regLog_prediction(request):
# # #     #tache 1: recevoir les colis
# # #     if request.method == 'POST':
# # #         #tache 2 : deballer le colis
# # #         hauteur = float(request.POST.get('hauteur'))
# # #         nbr_roues = float(request.POST.get('Nombre_de_roues'))
# # #         #tache 3: reveiller l'expert
# # #         #cette fonction(load_models) est defini avant
# # #         model = load_models('nouveau_modele.pkl')
# # #         #tache 4: poser la question a l'expert
# # #         prediction = model.predict([[hauteur, nbr_roues]])
# # #         predicted_class = prediction[0]
# # #         #tache 5: traduire la reponse
# # #         type_vehicules = {0:'Camion', 1:'Touristique'}
# # #         img_url = {'Camion': 'images/camion.jpg', 'Touristique':'images/touristique.jpg'}
# # #         pred_vehicule = type_vehicules[predicted_class]
# # #         pred_img = img_url[pred_vehicule]
# # #         #tache 6 : preparer le plateau-repas (context)
# # #         input_data = {
# # #             'hauteur':hauteur,
# # #             'nbr_roues':nbr_roues
# # #         }
# # #         context = {
# # #             'type_vehicule': pred_vehicule,
# # #             'img_vehicule':pred_img,
# # #             'initial_data':input_data
# # #         }
# # #         return render(request, 'reglog_results.html', context)
# # #     return render(request, 'vehicles_form.html')

# # import os
# # import joblib
# # import pandas as pd
# # import numpy as np
# # from django.shortcuts import render
# # from django.conf import settings
# # from sklearn.preprocessing import StandardScaler, LabelEncoder

# # # --- 1. CONFIGURATION ET CHARGEMENT ---
# # # (On charge tout au démarrage pour que ce soit rapide)

# # MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_app/ml_models')
# # ASSETS = {'models': {}, 'scalers': {}, 'encoders': {}}

# # def init_assets():
# #     """Charge les modèles .pkl et crée les scalers depuis les CSV"""
# #     if ASSETS['models']: return # Déjà chargé

# #     print("Chargement des assets IA...")
    
# #     # A. PREPARATION CLASSIFICATION (FITNESS)
# #     try:
# #         df_c = pd.read_csv(os.path.join(settings.BASE_DIR, 'fitness_dataset.csv'))
# #         # Nettoyage minimal pour recréer le scaler
# #         df_c['smokes'] = df_c['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
# #         le_gen = LabelEncoder()
# #         df_c['gender'] = le_gen.fit_transform(df_c['gender'].astype(str))
# #         df_c = df_c.fillna(df_c.mean(numeric_only=True))
        
# #         # Création Scaler Fitness
# #         X_c = df_c.drop(columns=['is_fit'])
# #         scaler_c = StandardScaler().fit(X_c)
# #         ASSETS['scalers']['fitness'] = scaler_c
        
# #         # Chargement des modèles Fitness
# #         ASSETS['models']['log_class'] = joblib.load(os.path.join(MODELS_DIR, 'logistic_reg.pkl'))
# #         ASSETS['models']['dt_class']  = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_classification.pkl'))
# #         # Ajoutez vos autres modèles ici si vous les avez (rf_class, svm_class...)
        
# #     except Exception as e:
# #         print(f"Erreur init Fitness: {e}")

# #     # B. PREPARATION REGRESSION (CALORIES)
# #     try:
# #         df_r = pd.read_csv(os.path.join(settings.BASE_DIR, 'calories_pred.csv'))
# #         df_r = df_r.drop(columns=['User_ID', 'Unnamed: 0'], errors='ignore')
# #         le_gen_r = LabelEncoder()
# #         df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])
        
# #         # Création Scaler Calories
# #         X_r = df_r.drop(columns=['Calories'])
# #         scaler_r = StandardScaler().fit(X_r)
# #         ASSETS['scalers']['calories'] = scaler_r
        
# #         # Chargement des modèles Calories
# #         ASSETS['models']['rf_reg']  = joblib.load(os.path.join(MODELS_DIR, 'rf_regression.pkl'))
# #         ASSETS['models']['lin_reg'] = joblib.load(os.path.join(MODELS_DIR, 'linear_reg.pkl'))
# #         ASSETS['models']['dt_reg']  = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_regression.pkl'))
        
# #     except Exception as e:
# #         print(f"Erreur init Calories: {e}")

# # # Lancer l'initialisation
# # init_assets()


# # # --- 2. VUES ---

# # def dashboard(request):
# #     # Affiche la grille des cartes (votre index.html mis à jour)
# #     return render(request, 'index.html')

# # def tester_modele(request, algo_name):
# #     """
# #     Vue universelle qui gère tous les modèles.
# #     algo_name: 'log_class', 'rf_reg', etc. (passé depuis l'URL)
# #     """
    
# #     # Déterminer le type (Fitness vs Calories) selon le nom du modèle
# #     if 'class' in algo_name:
# #         type_pred = 'fitness'
# #         titre = "Classification Fitness"
# #     else:
# #         type_pred = 'calories'
# #         titre = "Prédiction Calories"
        
# #     context = {
# #         'titre_algo': titre,
# #         'nom_modele': algo_name.replace('_', ' ').upper(),
# #         'type_pred': type_pred,
# #         'resultat': None
# #     }

# #     if request.method == 'POST':
# #         try:
# #             # 1. Récupération des données formulaire
# #             age = float(request.POST.get('age'))
# #             height = float(request.POST.get('height'))
# #             weight = float(request.POST.get('weight'))
# #             heart = float(request.POST.get('heart_rate'))
# #             gender = int(request.POST.get('gender'))

# #             model = ASSETS['models'].get(algo_name)
            
# #             if not model:
# #                 context['erreur'] = "Modèle introuvable ou non chargé."
# #                 return render(request, 'modele_tester.html', context)

# #             if type_pred == 'fitness':
# #                 # Features spécifiques Fitness
# #                 bp = 120.0 # Valeur par défaut
# #                 sleep = float(request.POST.get('sleep'))
# #                 nutri = float(request.POST.get('nutrition'))
# #                 activ = float(request.POST.get('activity'))
# #                 smokes = int(request.POST.get('smokes'))
                
# #                 # Ordre: age,height,weight,heart,bp,sleep,nutri,activ,smokes,gender
# #                 features = [[age, height, weight, heart, bp, sleep, nutri, activ, smokes, gender]]
                
# #                 # Scale
# #                 X_scaled = ASSETS['scalers']['fitness'].transform(features)
                
# #                 # Predict
# #                 pred = model.predict(X_scaled)[0]
# #                 context['resultat'] = "EN FORME (FIT) 💪" if pred == 1 else "PAS EN FORME ⚠️"
                
# #                 if hasattr(model, "predict_proba"):
# #                     context['confiance'] = round(max(model.predict_proba(X_scaled)[0]) * 100, 2)

# #             else:
# #                 # Features spécifiques Calories
# #                 duration = float(request.POST.get('duration'))
# #                 temp = float(request.POST.get('body_temp'))
                
# #                 # Ordre: Gender,Age,Height,Weight,Duration,Heart,Temp
# #                 features = [[gender, age, height, weight, duration, heart, temp]]
                
# #                 # Scale
# #                 X_scaled = ASSETS['scalers']['calories'].transform(features)
                
# #                 # Predict
# #                 val = model.predict(X_scaled)[0]
# #                 context['resultat'] = f"{round(val, 2)} kCal brûlées 🔥"

# #         except Exception as e:
# #             context['erreur'] = f"Erreur de calcul : {str(e)}"

# #     return render(request, 'modele_tester.html', context)
# import os
# import joblib
# import pandas as pd
# import numpy as np
# from django.shortcuts import render
# from django.conf import settings
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # --- 1. CONFIGURATION DES MODÈLES ---
# # Mise à jour des chemins pour inclure les sous-dossiers
# MODELS_CONFIG = {
#     # --- CLASSIFICATION ---
#     'log_class': {
#         'file': os.path.join('classification', 'logistic_reg.pkl'), 
#         'type': 'fitness', 
#         'name': 'Régression Logistique'
#     },
#     'dt_class': {
#         'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
#         'type': 'fitness', 
#         'name': 'Arbre de Décision'
#     },
#     'rf_class': {
#         'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
#         'type': 'fitness', 
#         'name': 'Random Forest Classifier'
#     },
#     'svm_class': {
#         'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
#         'type': 'fitness', 
#         'name': 'SVM Classifier'
#     },
#     'xgb_class': {
#         'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
#         'type': 'fitness', 
#         'name': 'XGboost Classifier'
#     },
    
#     # --- REGRESSION ---
#     'lin_reg': {
#         'file': os.path.join('regression', 'lin_reg_poly.pkl'), 
#         'type': 'calories', 
#         'name': 'Régression Linéaire'
#     },
#     'dt_reg': {
#         'file': os.path.join('regression', 'dt_reg_opt.pkl'), 
#         'type': 'calories', 
#         'name': 'Arbre de Décision (Rég)'
#     },
#     'rf_reg': {
#         'file': os.path.join('regression', 'rf_regression.pkl'), 
#         'type': 'calories', 
#         'name': 'Random Forest (Rég)'
#     },
#     'svm_reg': {
#         'file': os.path.join('regression', 'svr_reg.pkl'), 
#         'type': 'calories', 
#         'name': 'SVM Regression'
#     },
#     'xgboost_reg': {
#         'file': os.path.join('regression', 'xgb_reg.pkl'), 
#         'type': 'calories', 
#         'name': 'XGboost Regression'
#     },
# }

# # --- 2. MOTEUR IA (SINGLETON) ---
# AI_ENGINE = {'models': {}, 'scalers': {}}

# def init_engine():
#     """Charge les modèles .pkl et reconstruit les scalers une seule fois au démarrage."""
#     if AI_ENGINE['models']: return

#     print("⚡ Initialisation du moteur IA pour algoML...")
    
#     # Chemin racine des modèles : mlPlatform/algoML/ml_models/
#     base_models_dir = os.path.join(settings.BASE_DIR, 'algoML', 'ml_models')

#     # A. SCALER FITNESS (Reconstruit depuis le CSV racine)
#     try:
#         csv_path = os.path.join(settings.BASE_DIR, 'fitness_dataset.csv')
#         df_c = pd.read_csv(csv_path)
        
#         # Nettoyage identique à l'entraînement
#         df_c['smokes'] = df_c['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
#         le_gen = LabelEncoder()
#         df_c['gender'] = le_gen.fit_transform(df_c['gender'].astype(str))
#         df_c = df_c.fillna(df_c.mean(numeric_only=True))
        
#         scaler_c = StandardScaler()
#         scaler_c.fit(df_c.drop(columns=['is_fit']))
#         AI_ENGINE['scalers']['fitness'] = scaler_c
#         print("✅ Scaler Fitness prêt.")
#     except Exception as e: 
#         print(f"⚠️ Erreur Scaler Fitness: {e}")

#     # B. SCALER CALORIES (Reconstruit depuis le CSV racine)
#     try:
#         csv_path = os.path.join(settings.BASE_DIR, 'calories_pred.csv')
#         df_r = pd.read_csv(csv_path)
        
#         le_gen_r = LabelEncoder()
#         df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])
        
        
#         # 2. FORCEZ L'ORDRE EXACT QUE VOUS AVEZ TROUVÉ
#         ordre_officiel = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
#         # On filtre et on réordonne le DataFrame d'un coup
#         # Cela supprime aussi automatiquement User_ID et Calories car ils ne sont pas dans la liste
#         df_finale = df_r[ordre_officiel]
        
#         scaler_r = StandardScaler()
#         scaler_r.fit(df_finale)
#         AI_ENGINE['scalers']['calories'] = scaler_r
#         print("✅ Scaler Calories prêt.")
#     except Exception as e: 
#         print(f"⚠️ Erreur Scaler Calories: {e}")

#     # C. CHARGEMENT DES PKL
#     for key, config in MODELS_CONFIG.items():
#         try:
#             # On combine le dossier de base avec le chemin relatif (ex: classification/file.pkl)
#             path = os.path.join(base_models_dir, config['file'])
            
#             if os.path.exists(path):
#                 AI_ENGINE['models'][key] = joblib.load(path)
#                 print(f"  -> Chargé: {config['name']}")
#             else:
#                 print(f"  ❌ Manquant: {path}")
#         except Exception as e:
#             print(f"  ❌ Erreur {key}: {e}")

# # Lancer l'initialisation au démarrage
# init_engine()


# # --- 3. VUES DJANGO ---

# def index(request):
#     """Page d'accueil Dashboard"""
#     return render(request, 'index.html')

# def regLog_details(request):
#     """Page de détails statique"""
#     return render(request, 'regLog_details.html')

# def regLog_details(request):
#     return render(request, 'regLog_details.html')

# def lin_reg_details(request):
#     return render(request, 'lin_reg_details.html')

# def dt_reg_details(request):
#     return render(request, 'dt_reg_details.html')

# def dt_class_details(request):
#     return render(request, 'dt_classi_details.html')

# def rf_class_details(request):
#     return render(request, 'rf_classi_details.html')

# def rf_reg_details(request):
#     return render(request, 'rf_reg_details.html')

# def svm_class_details(request):
#     return render(request, 'svm_classi_details.html')

# def svm_reg_details(request):
#     return render(request, 'svm_reg_details.html')

# def xgb_classi_details(request):
#     return render(request, 'XGboost_classi_details.html')

# def xgb_reg_details(request):
#     return render(request, 'XGboost_reg_details.html')

# # Dans algoML/views.py

# def tester_modele(request, algo_name):
#     config = MODELS_CONFIG.get(algo_name)
#     if not config:
#         return render(request, 'index.html', {'error': "Modèle introuvable"})

#     type_pred = config['type']
#     context = {'algo_name': algo_name, 'nom_modele': config['name'], 'type_pred': type_pred}

#     if request.method == 'POST':
#         try:
#             # 1. RECUPERATION
#             age = float(request.POST.get('age', 0))
#             height = float(request.POST.get('height', 0))
#             weight = float(request.POST.get('weight', 0))
#             heart = float(request.POST.get('heart_rate', 0))
#             gender = int(request.POST.get('gender', 0))

#             model = AI_ENGINE['models'].get(algo_name)
            
#             if type_pred == 'fitness':
#                 # Classification...
#                 bp = 120.0
#                 sleep = float(request.POST.get('sleep', 7))
#                 nutri = float(request.POST.get('nutrition', 5))
#                 activ = float(request.POST.get('activity', 5))
#                 smokes = int(request.POST.get('smokes', 0))
                
#                 features = [[age, height, weight, heart, bp, sleep, nutri, activ, smokes, gender]]
                
#                 if 'fitness' in AI_ENGINE['scalers']:
#                     features = AI_ENGINE['scalers']['fitness'].transform(features)
                
#                 pred = model.predict(features)[0]
#                 context['resultat'] = "EN FORME (FIT) 💪" if pred == 1 else "PAS EN FORME ⚠️"

#             else: 
#                 # --- REGRESSION (Le problème est ici) ---
#                 duration = float(request.POST.get('duration', 0))
#                 temp = float(request.POST.get('body_temp', 37))
                

#                 # Ordre précis des colonnes
#                 features = [[gender, age, height, weight, duration, heart, temp]]
                
#                 # Debug 2 : Vérifier le Scaler
#                 if 'calories' in AI_ENGINE['scalers']:
#                     print("✅ Scaler Calories trouvé. Transformation en cours...")
#                     features_scaled = AI_ENGINE['scalers']['calories'].transform(features)
#                     val = model.predict(features_scaled)[0]
#                 else:
#                     print("❌ ATTENTION : Scaler Calories INTROUVABLE ! Utilisation des données brutes.")
#                     val = model.predict(features)[0] # C'est sûrement ça qui cause le problème

#                 context['resultat'] = f"{round(val, 3)} kCal brûlées 🔥"

#         except Exception as e:
#             print(f"ERREUR CRITIQUE : {e}")
#             context['erreur'] = f"Erreur : {str(e)}"

#     return render(request, 'model_tester.html', context)

















# # from django.shortcuts import render
# # import os
# # import joblib
# # # Create your views here.
# # def index(request):
# #     return render(request, 'index.html')
# # def regLog_details(request):
# #     return render(request, 'regLog_details.html')
# # def regLog_atelier(request):
# #     return render(request, 'regLog_atelier.html')
# # def regLog_tester(request):
# #     return render(request, 'vehicles_form.html')
# # def load_models(name):
# #     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# #     models_dir = os.path.join(base_dir,'models_ai')
# #     model_path = os.path.join(models_dir, name)
# #     ml_model = joblib.load(model_path)
# #     return ml_model
# # def regLog_prediction(request):
# #     #tache 1: recevoir les colis
# #     if request.method == 'POST':
# #         #tache 2 : deballer le colis
# #         hauteur = float(request.POST.get('hauteur'))
# #         nbr_roues = float(request.POST.get('Nombre_de_roues'))
# #         #tache 3: reveiller l'expert
# #         #cette fonction(load_models) est defini avant
# #         model = load_models('nouveau_modele.pkl')
# #         #tache 4: poser la question a l'expert
# #         prediction = model.predict([[hauteur, nbr_roues]])
# #         predicted_class = prediction[0]
# #         #tache 5: traduire la reponse
# #         type_vehicules = {0:'Camion', 1:'Touristique'}
# #         img_url = {'Camion': 'images/camion.jpg', 'Touristique':'images/touristique.jpg'}
# #         pred_vehicule = type_vehicules[predicted_class]
# #         pred_img = img_url[pred_vehicule]
# #         #tache 6 : preparer le plateau-repas (context)
# #         input_data = {
# #             'hauteur':hauteur,
# #             'nbr_roues':nbr_roues
# #         }
# #         context = {
# #             'type_vehicule': pred_vehicule,
# #             'img_vehicule':pred_img,
# #             'initial_data':input_data
# #         }
# #         return render(request, 'reglog_results.html', context)
# #     return render(request, 'vehicles_form.html')
# import os
# import joblib
# import pandas as pd
# import numpy as np
# from django.shortcuts import render
# from django.conf import settings
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# # --- 1. CONFIGURATION ET CHARGEMENT ---
# # (On charge tout au démarrage pour que ce soit rapide)
# MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_app/ml_models')
# ASSETS = {'models': {}, 'scalers': {}, 'encoders': {}}
# def init_assets():
#     """Charge les modèles .pkl et crée les scalers depuis les CSV"""
#     if ASSETS['models']: return # Déjà chargé
#     print("Chargement des assets IA...")
    
#     # A. PREPARATION CLASSIFICATION (FITNESS)
#     try:
#         df_c = pd.read_csv(os.path.join(settings.BASE_DIR, 'fitness_dataset.csv'))
#         # Nettoyage minimal pour recréer le scaler
#         df_c['smokes'] = df_c['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
#         le_gen = LabelEncoder()
#         df_c['gender'] = le_gen.fit_transform(df_c['gender'].astype(str))
#         df_c = df_c.fillna(df_c.mean(numeric_only=True))
        
#         # Création Scaler Fitness
#         X_c = df_c.drop(columns=['is_fit'])
#         scaler_c = StandardScaler().fit(X_c)
#         ASSETS['scalers']['fitness'] = scaler_c
        
#         # Chargement des modèles Fitness
#         ASSETS['models']['log_class'] = joblib.load(os.path.join(MODELS_DIR, 'logistic_reg.pkl'))
#         ASSETS['models']['dt_class']  = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_classification.pkl'))
#         # Ajoutez vos autres modèles ici si vous les avez (rf_class, svm_class...)
        
#     except Exception as e:
#         print(f"Erreur init Fitness: {e}")
#     # B. PREPARATION REGRESSION (CALORIES)
#     try:
#         df_r = pd.read_csv(os.path.join(settings.BASE_DIR, 'calories_pred.csv'))
#         df_r = df_r.drop(columns=['User_ID', 'Unnamed: 0'], errors='ignore')
#         le_gen_r = LabelEncoder()
#         df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])
        
#         # Création Scaler Calories
#         X_r = df_r.drop(columns=['Calories'])
#         scaler_r = StandardScaler().fit(X_r)
#         ASSETS['scalers']['calories'] = scaler_r
        
#         # Chargement des modèles Calories
#         ASSETS['models']['rf_reg']  = joblib.load(os.path.join(MODELS_DIR, 'rf_regression.pkl'))
#         ASSETS['models']['lin_reg'] = joblib.load(os.path.join(MODELS_DIR, 'linear_reg.pkl'))
#         ASSETS['models']['dt_reg']  = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_regression.pkl'))
        
#     except Exception as e:
#         print(f"Erreur init Calories: {e}")
# # Lancer l'initialisation
# init_assets()
# # --- 2. VUES ---
# def dashboard(request):
#     # Affiche la grille des cartes (votre index.html mis à jour)
#     return render(request, 'index.html')
# def tester_modele(request, algo_name):
#     """
#     Vue universelle qui gère tous les modèles.
#     algo_name: 'log_class', 'rf_reg', etc. (passé depuis l'URL)
#     """
    
#     # Déterminer le type (Fitness vs Calories) selon le nom du modèle
#     if 'class' in algo_name:
#         type_pred = 'fitness'
#         titre = "Classification Fitness"
#     else:
#         type_pred = 'calories'
#         titre = "Prédiction Calories"
        
#     context = {
#         'titre_algo': titre,
#         'nom_modele': algo_name.replace('_', ' ').upper(),
#         'type_pred': type_pred,
#         'resultat': None
#     }
#     if request.method == 'POST':
#         try:
#             # 1. Récupération des données formulaire
#             age = float(request.POST.get('age'))
#             height = float(request.POST.get('height'))
#             weight = float(request.POST.get('weight'))
#             heart = float(request.POST.get('heart_rate'))
#             gender = int(request.POST.get('gender'))
#             model = ASSETS['models'].get(algo_name)
            
#             if not model:
#                 context['erreur'] = "Modèle introuvable ou non chargé."
#                 return render(request, 'modele_tester.html', context)
#             if type_pred == 'fitness':
#                 # Features spécifiques Fitness
#                 bp = 120.0 # Valeur par défaut
#                 sleep = float(request.POST.get('sleep'))
#                 nutri = float(request.POST.get('nutrition'))
#                 activ = float(request.POST.get('activity'))
#                 smokes = int(request.POST.get('smokes'))
                
#                 # Ordre: age,height,weight,heart,bp,sleep,nutri,activ,smokes,gender
#                 features = [[age, height, weight, heart, bp, sleep, nutri, activ, smokes, gender]]
                
#                 # Scale
#                 X_scaled = ASSETS['scalers']['fitness'].transform(features)
                
#                 # Predict
#                 pred = model.predict(X_scaled)[0]
#                 context['resultat'] = "EN FORME (FIT) 💪" if pred == 1 else "PAS EN FORME ⚠️"
                
#                 if hasattr(model, "predict_proba"):
#                     context['confiance'] = round(max(model.predict_proba(X_scaled)[0]) * 100, 2)
#             else:
#                 # Features spécifiques Calories
#                 duration = float(request.POST.get('duration'))
#                 temp = float(request.POST.get('body_temp'))
                
#                 # Ordre: Gender,Age,Height,Weight,Duration,Heart,Temp
#                 features = [[gender, age, height, weight, duration, heart, temp]]
                
#                 # Scale
#                 X_scaled = ASSETS['scalers']['calories'].transform(features)
                
#                 # Predict
#                 val = model.predict(X_scaled)[0]
#                 context['resultat'] = f"{round(val, 2)} kCal brûlées 🔥"
#         except Exception as e:
#             context['erreur'] = f"Erreur de calcul : {str(e)}"
#     return render(request, 'modele_tester.html', context)
import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import redirect
from .models import Historique
import json
# --- 1. CONFIGURATION DES MODÈLES ---
# Mise à jour des chemins pour inclure les sous-dossiers
MODELS_CONFIG = {
    # --- CLASSIFICATION ---
    'log_class': {
        'file': os.path.join('classification', 'logistic_reg.pkl'), 
        'type': 'fitness', 
        'name': 'Régression Logistique'
    },
    'dt_class': {
        'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
        'type': 'fitness', 
        'name': 'Arbre de Décision'
    },
    'rf_class': {
        'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
        'type': 'fitness', 
        'name': 'Random Forest Classifier'
    },
    'svm_class': {
        'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
        'type': 'fitness', 
        'name': 'SVM Classifier'
    },
    'xgb_class': {
        'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
        'type': 'fitness', 
        'name': 'XGboost Classifier'
    },
    
    # --- REGRESSION ---
    'lin_reg': {
        'file': os.path.join('regression', 'lin_reg_poly.pkl'), 
        'type': 'calories', 
        'name': 'Régression Linéaire'
    },
    'dt_reg': {
        'file': os.path.join('regression', 'dt_reg_opt.pkl'), 
        'type': 'calories', 
        'name': 'Arbre de Décision (Rég)'
    },
    'rf_reg': {
        'file': os.path.join('regression', 'rf_regression.pkl'), 
        'type': 'calories', 
        'name': 'Random Forest (Rég)'
    },
    'svm_reg': {
        'file': os.path.join('regression', 'svr_reg.pkl'), 
        'type': 'calories', 
        'name': 'SVM Regression'
    },
    'xgboost_reg': {
        'file': os.path.join('regression', 'xgb_reg.pkl'), 
        'type': 'calories', 
        'name': 'XGboost Regression'
    },
}
# --- 2. MOTEUR IA (SINGLETON) ---
AI_ENGINE = {'models': {}, 'scalers': {}}
def init_engine():
    """Charge les modèles .pkl et reconstruit les scalers une seule fois au démarrage."""
    if AI_ENGINE['models']: return
    print("⚡ Initialisation du moteur IA pour algoML...")
    
    # Chemin racine des modèles : mlPlatform/algoML/ml_models/
    base_models_dir = os.path.join(settings.BASE_DIR, 'algoML', 'ml_models')
    # A. SCALER FITNESS (Reconstruit depuis le CSV racine)
    try:
        csv_path = os.path.join(settings.BASE_DIR, 'fitness_dataset.csv')
        df_c = pd.read_csv(csv_path)
        
        # Nettoyage identique à l'entraînement
        df_c['smokes'] = df_c['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
        le_gen = LabelEncoder()
        df_c['gender'] = le_gen.fit_transform(df_c['gender'].astype(str))
        df_c = df_c.fillna(df_c.mean(numeric_only=True))
        
        scaler_c = StandardScaler()
        scaler_c.fit(df_c.drop(columns=['is_fit']))
        AI_ENGINE['scalers']['fitness'] = scaler_c
        print("✅ Scaler Fitness prêt.")
    except Exception as e: 
        print(f"⚠️ Erreur Scaler Fitness: {e}")
    # B. SCALER CALORIES (Reconstruit depuis le CSV racine)
    try:
        csv_path = os.path.join(settings.BASE_DIR, 'calories_pred.csv')
        df_r = pd.read_csv(csv_path)
        
        le_gen_r = LabelEncoder()
        df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])
        
        
        # 2. FORCEZ L'ORDRE EXACT QUE VOUS AVEZ TROUVÉ
        ordre_officiel = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
        # On filtre et on réordonne le DataFrame d'un coup
        # Cela supprime aussi automatiquement User_ID et Calories car ils ne sont pas dans la liste
        df_finale = df_r[ordre_officiel]
        
        scaler_r = StandardScaler()
        scaler_r.fit(df_finale)
        AI_ENGINE['scalers']['calories'] = scaler_r
        print("✅ Scaler Calories prêt.")
    except Exception as e: 
        print(f"⚠️ Erreur Scaler Calories: {e}")
    # C. CHARGEMENT DES PKL
    for key, config in MODELS_CONFIG.items():
        try:
            # On combine le dossier de base avec le chemin relatif (ex: classification/file.pkl)
            path = os.path.join(base_models_dir, config['file'])
            
            if os.path.exists(path):
                AI_ENGINE['models'][key] = joblib.load(path)
                print(f"  -> Chargé: {config['name']}")
            else:
                print(f"  ❌ Manquant: {path}")
        except Exception as e:
            print(f"  ❌ Erreur {key}: {e}")
# Lancer l'initialisation au démarrage
init_engine()
# --- 3. VUES DJANGO ---
def index(request):
    """Page d'accueil Dashboard"""
    return render(request, 'index.html')
def regLog_details(request):
    """Page de détails statique"""
    return render(request, 'regLog_details.html')
def regLog_details(request):
    return render(request, 'regLog_details.html')
def lin_reg_details(request):
    return render(request, 'lin_reg_details.html')
def dt_reg_details(request):
    return render(request, 'dt_reg_details.html')
def dt_class_details(request):
    return render(request, 'dt_classi_details.html')
def rf_class_details(request):
    return render(request, 'rf_classi_details.html')
def rf_reg_details(request):
    return render(request, 'rf_reg_details.html')
def svm_class_details(request):
    return render(request, 'svm_classi_details.html')
def svm_reg_details(request):
    return render(request, 'svm_reg_details.html')
def xgb_classi_details(request):
    return render(request, 'XGboost_classi_details.html')
def xgb_reg_details(request):
    return render(request, 'XGboost_reg_details.html')
def about(request):
    return render(request, 'about.html')
def tester_modele(request, algo_name):
    config = MODELS_CONFIG.get(algo_name)
    if not config:
        return render(request, 'index.html', {'error': "Modèle introuvable"})
    type_pred = config['type']
    context = {'algo_name': algo_name, 'nom_modele': config['name'], 'type_pred': type_pred}
    if request.method == 'POST':
        try:
            # 1. Récupération des données communes
            age = float(request.POST.get('age', 0))
            height = float(request.POST.get('height', 0))
            weight = float(request.POST.get('weight', 0))
            heart = float(request.POST.get('heart_rate', 0))
            gender = int(request.POST.get('gender', 0))
            
            model = AI_ENGINE['models'].get(algo_name)
            
            # Variable pour stocker le résultat texte (IMPORTANT : Doit être définie dans les 2 cas)
            res_str = "" 
            # === CAS 1 : CLASSIFICATION (FITNESS) ===
            if type_pred == 'fitness':
                bp = 120.0
                sleep = float(request.POST.get('sleep', 7))
                nutri = float(request.POST.get('nutrition', 5))
                activ = float(request.POST.get('activity', 5))
                smokes = int(request.POST.get('smokes', 0))
                
                features = [[age, height, weight, heart, bp, sleep, nutri, activ, smokes, gender]]
                
                if 'fitness' in AI_ENGINE['scalers']:
                    features = AI_ENGINE['scalers']['fitness'].transform(features)
                
                pred = model.predict(features)[0]
                
                # DÉFINITION DU RÉSULTAT POUR L'HISTORIQUE
                res_str = "FIT 💪" if pred == 1 else "NOT FIT ⚠️"
                context['resultat'] = res_str
            # === CAS 2 : RÉGRESSION (CALORIES) ===
            else: 
                duration = float(request.POST.get('duration', 0))
                temp = float(request.POST.get('body_temp', 37))
                
                features = [[gender, age, height, weight, duration, heart, temp]]
                
                if 'calories' in AI_ENGINE['scalers']:
                    features = AI_ENGINE['scalers']['calories'].transform(features)
                
                raw_val = model.predict(features)[0]
                val_float = float(raw_val)
                
                # DÉFINITION DU RÉSULTAT POUR L'HISTORIQUE
                res_str = f"{round(val_float, 2)} kCal brûlées 🔥"
                context['resultat'] = res_str
            # ====================================================
            # SAUVEGARDE (DOIT ÊTRE ALIGNÉ AVEC LE 'if/else' DU DESSUS)
            # ====================================================
            ### ATTENTION : Ce bloc ne doit PAS être indenté dans le 'else' !
            if request.user.is_authenticated:
                donnees = request.POST.dict()
                if 'csrfmiddlewaretoken' in donnees:
                    del donnees['csrfmiddlewaretoken']
                
                Historique.objects.create(
                    user=request.user,
                    algo_name=config['name'],
                    input_data=donnees,
                    resultat=res_str  # On utilise la variable créée plus haut
                )
                print(f"✅ Historique sauvegardé : {res_str}")
            # ====================================================
        except Exception as e:
            print(f"ERREUR : {e}")
            context['erreur'] = f"Erreur : {str(e)}"
    return render(request, 'model_tester.html', context)
# def tester_modele(request, algo_name):
#     config = MODELS_CONFIG.get(algo_name)
#     if not config:
#         return render(request, 'index.html', {'error': "Modèle introuvable"})
#     type_pred = config['type']
#     context = {'algo_name': algo_name, 'nom_modele': config['name'], 'type_pred': type_pred}
#             else: 
#                 # --- REGRESSION (Le problème est ici) ---
#                 duration = float(request.POST.get('duration', 0))
#                 temp = float(request.POST.get('body_temp', 37))
                
#                 # Ordre précis des colonnes
#                 features = [[gender, age, height, weight, duration, heart, temp]]
                
#                 # Debug 2 : Vérifier le Scaler
#                 if 'calories' in AI_ENGINE['scalers']:
#                     print("✅ Scaler Calories trouvé. Transformation en cours...")
#                     features_scaled = AI_ENGINE['scalers']['calories'].transform(features)
#                     val = model.predict(features_scaled)[0]
#                     val = float(val)
#                 else:
#                     print("❌ ATTENTION : Scaler Calories INTROUVABLE ! Utilisation des données brutes.")
#                     val = model.predict(features)[0] 
#                     val = float(val)
#                 context['resultat'] = f"{round(val, 2)} kCal brûlées 🔥"
                
#             if request.user.is_authenticated:
#                 # On nettoie les données du formulaire (on enlève le token de sécurité)
#                 donnees_saisies = request.POST.dict()
#                 del donnees_saisies['csrfmiddlewaretoken']                
#                 # On enregistre
#                 Historique.objects.create(
#                     user=request.user,
#                     algo_name=config['name'], # Le nom joli (ex: Random Forest)                        input_data=donnees_saisies,
#                     resultat=val
#                 )
#                 print("✅ Historique sauvegardé !")
#             # ====================================================
#         except Exception as e:
#             print(f"ERREUR CRITIQUE : {e}")
#             context['erreur'] = f"Erreur : {str(e)}"
#     return render(request, 'model_tester.html', context)
def signup_view(request):
    if request.method == 'POST':
        # 1. Récupérer les données du formulaire
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        # 2. Vérifications de base
        if password != password_confirm:
            messages.error(request, "Les mots de passe ne correspondent pas.")
            return render(request, 'signup.html')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Ce nom d'utilisateur est déjà pris.")
            return render(request, 'signup.html')
        # 3. Création de l'utilisateur
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            
            # 4. Connexion automatique après inscription
            login(request, user)
            messages.success(request, f"Bienvenue, {username} !")
            return redirect('index') # Redirige vers l'accueil
            
        except Exception as e:
            messages.error(request, f"Erreur lors de l'inscription : {e}")
    return render(request, 'signup.html')
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Vérification des identifiants
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Ravi de vous revoir, {username} !")
            return redirect('index')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    
    return render(request, 'login.html')
def logout_view(request):
    logout(request)
    messages.info(request, "Vous avez été déconnecté.")
    return redirect('login')
from django.contrib.auth.decorators import login_required
@login_required # Oblige à être connecté pour voir cette page
def historique_view(request):
    # Récupère l'historique de l'utilisateur connecté uniquement
    historiques = Historique.objects.filter(user=request.user)
    return render(request, 'historique.html', {'historiques': historiques})