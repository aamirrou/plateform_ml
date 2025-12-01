"""
Classification avec SVM, Random Forest et XGBoost
Utilisant le dataset fitness_dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ====================

print("📂 Chargement du dataset...")
df = pd.read_csv('mlPlatform/fitness_dataset.csv')

print(f"✅ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"\nPremières lignes:")
print(df.head())

print(f"\nInformations sur le dataset:")
print(df.info())

print(f"\nStatistiques descriptives:")
print(df.describe())

print(f"\nValeurs manquantes:")
print(df.isnull().sum())

print(f"\nDistribution de la variable cible (is_fit):")
print(df['is_fit'].value_counts())

# Nettoyage des données
print("\n🔧 Nettoyage des données...")

# Convertir 'smokes' en numérique
df['smokes'] = df['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)

# Encoder 'gender'
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'].astype(str))

# Remplir les valeurs manquantes
df = df.fillna(df.mean(numeric_only=True))

print("✅ Nettoyage terminé")

# Séparation features et target
X = df.drop(columns=['is_fit'])
y = df['is_fit']

print(f"\n📊 Features: {X.shape[1]} colonnes")
print(f"Features: {list(X.columns)}")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Standardisation effectuée")


# ==================== 2. SVM (Support Vector Machine) ====================

print("\n" + "="*70)
print("🤖 ENTRAÎNEMENT SVM")
print("="*70)

# SVM avec kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred_svm = svm_model.predict(X_test_scaled)
y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Métriques
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"\n📊 Résultats SVM:")
print(f"   Accuracy:  {acc_svm:.4f} ({acc_svm*100:.2f}%)")
print(f"   Precision: {prec_svm:.4f}")
print(f"   Recall:    {recall_svm:.4f}")
print(f"   F1-Score:  {f1_svm:.4f}")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Not Fit', 'Fit']))


# ==================== 3. RANDOM FOREST ====================

print("\n" + "="*70)
print("🌲 ENTRAÎNEMENT RANDOM FOREST")
print("="*70)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Prédictions
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Métriques
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"\n📊 Résultats Random Forest:")
print(f"   Accuracy:  {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"   Precision: {prec_rf:.4f}")
print(f"   Recall:    {recall_rf:.4f}")
print(f"   F1-Score:  {f1_rf:.4f}")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Not Fit', 'Fit']))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🎯 Top 5 Features importantes (Random Forest):")
print(feature_importance.head())


# ==================== 4. XGBOOST ====================

print("\n" + "="*70)
print("🚀 ENTRAÎNEMENT XGBOOST")
print("="*70)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

# Prédictions
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Métriques
acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print(f"\n📊 Résultats XGBoost:")
print(f"   Accuracy:  {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"   Precision: {prec_xgb:.4f}")
print(f"   Recall:    {recall_xgb:.4f}")
print(f"   F1-Score:  {f1_xgb:.4f}")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Not Fit', 'Fit']))


# ==================== 5. COMPARAISON DES MODÈLES ====================

print("\n" + "="*70)
print("📊 COMPARAISON DES MODÈLES")
print("="*70)

comparison_df = pd.DataFrame({
    'Modèle': ['SVM', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_svm, acc_rf, acc_xgb],
    'Precision': [prec_svm, prec_rf, prec_xgb],
    'Recall': [recall_svm, recall_rf, recall_xgb],
    'F1-Score': [f1_svm, f1_rf, f1_xgb]
})

print(comparison_df.to_string(index=False))

# Meilleur modèle
best_model_idx = comparison_df['Accuracy'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Modèle']
best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy']

print(f"\n🏆 Meilleur modèle: {best_model_name} (Accuracy: {best_accuracy:.4f})")


# ==================== 6. VISUALISATIONS ====================

print("\n📊 Création des visualisations...")

fig = plt.figure(figsize=(18, 12))

# 1. Comparaison des métriques
ax1 = plt.subplot(3, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, [acc_svm, prec_svm, recall_svm, f1_svm], width, label='SVM', color='#3498db')
plt.bar(x, [acc_rf, prec_rf, recall_rf, f1_rf], width, label='Random Forest', color='#2ecc71')
plt.bar(x + width, [acc_xgb, prec_xgb, recall_xgb, f1_xgb], width, label='XGBoost', color='#e74c3c')

plt.xlabel('Métriques')
plt.ylabel('Score')
plt.title('Comparaison des Métriques')
plt.xticks(x, metrics, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)

# 2. Matrice de confusion - SVM
ax2 = plt.subplot(3, 3, 2)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matrice de Confusion - SVM')
plt.xlabel('Prédit')
plt.ylabel('Réel')

# 3. Matrice de confusion - Random Forest
ax3 = plt.subplot(3, 3, 3)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Matrice de Confusion - Random Forest')
plt.xlabel('Prédit')
plt.ylabel('Réel')

# 4. Matrice de confusion - XGBoost
ax4 = plt.subplot(3, 3, 4)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Matrice de Confusion - XGBoost')
plt.xlabel('Prédit')
plt.ylabel('Réel')

# 5. ROC Curves
ax5 = plt.subplot(3, 3, 5)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

auc_svm = roc_auc_score(y_test, y_proba_svm)
auc_rf = roc_auc_score(y_test, y_proba_rf)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})', color='#3498db', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC = {auc_rf:.3f})', color='#2ecc71', linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})', color='#e74c3c', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)

plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbes ROC')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Feature Importance - Random Forest
ax6 = plt.subplot(3, 3, 6)
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'], color='#2ecc71')
plt.xlabel('Importance')
plt.title('Top 10 Features - Random Forest')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

# 7. Feature Importance - XGBoost
ax7 = plt.subplot(3, 3, 7)
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.barh(xgb_importance['Feature'], xgb_importance['Importance'], color='#e74c3c')
plt.xlabel('Importance')
plt.title('Top 10 Features - XGBoost')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

# 8. Distribution des probabilités - Random Forest
ax8 = plt.subplot(3, 3, 8)
plt.hist(y_proba_rf[y_test == 0], bins=30, alpha=0.5, color='red', label='Not Fit', edgecolor='black')
plt.hist(y_proba_rf[y_test == 1], bins=30, alpha=0.5, color='green', label='Fit', edgecolor='black')
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Seuil')
plt.xlabel('Probabilité')
plt.ylabel('Fréquence')
plt.title('Distribution des Probabilités - RF')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Comparaison Accuracy
ax9 = plt.subplot(3, 3, 9)
models = ['SVM', 'Random Forest', 'XGBoost']
accuracies = [acc_svm, acc_rf, acc_xgb]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Accuracy')
plt.title('Comparaison de l\'Accuracy')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('classification_models_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualisations sauvegardées: classification_models_comparison.png")


# ==================== 7. SAUVEGARDE DES MODÈLES ====================

import joblib

print("\n💾 Sauvegarde des modèles...")

joblib.dump(svm_model, 'svm_classification.pkl')
joblib.dump(rf_model, 'rf_classification.pkl')
joblib.dump(xgb_model, 'xgb_classification.pkl')
joblib.dump(scaler, 'scaler_classification.pkl')

print("✅ Modèles sauvegardés:")
print("   - svm_classification.pkl")
print("   - rf_classification.pkl")
print("   - xgb_classification.pkl")
print("   - scaler_classification.pkl")

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
