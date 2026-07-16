# Plateform ML

Projet de plateforme Machine Learning permettant d'entraîner, tester et exposer des modèles via une interface web.

## 📋 Description

Ce projet propose une petite plateforme web (atelier) qui combine une interface HTML et un back-end Python pour manipuler des modèles de Machine Learning : chargement de données, entraînement, prédiction et visualisation des résultats.

> ℹ️ N'hésite pas à compléter cette section avec une description plus précise du fonctionnement réel de ton application (type de modèles, cas d'usage, etc.).

## 🚀 Fonctionnalités

- Chargement et prétraitement de données
- Entraînement de modèle(s) de Machine Learning
- Interface web pour interagir avec le modèle
- Visualisation des résultats / prédictions

## 🏗️ Structure du projet

```
plateform_ml/
├── mlPlatform/          # Code principal de l'application
├── requirements.txt     # Dépendances Python du projet
├── .gitignore
└── README.md
```

## 🔧 Prérequis

- Python 3.x
- pip

## 📦 Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/aamirrou/plateform_ml.git
   cd plateform_ml
   ```

2. Créer et activer un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate      # Sur Linux/Mac
   venv\Scripts\activate         # Sur Windows
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Utilisation

Depuis le dossier du projet, lancer l'application :

```bash
cd mlPlatform
python app.py
```

> ℹ️ Remplace `app.py` par le nom réel du fichier de lancement de ton application si différent.

L'application sera ensuite accessible dans ton navigateur à l'adresse indiquée dans la console (généralement `http://localhost:5000` ou `http://127.0.0.1:8000`).

## 🛠️ Technologies utilisées

- **Python** — logique métier / traitement des données / Machine Learning
- **HTML** — interface utilisateur

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork ce dépôt
2. Crée une branche pour ta fonctionnalité (`git checkout -b feature/ma-fonctionnalite`)
3. Commit tes changements (`git commit -m 'Ajout de ma fonctionnalité'`)
4. Push la branche (`git push origin feature/ma-fonctionnalite`)
5. Ouvre une Pull Request

## 👤 Auteur

**aamirrou**
- GitHub: [@aamirrou](https://github.com/aamirrou)
