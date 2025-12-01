"""
Implémentation de la Régression Logistique à partir de zéro
Sans utiliser scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt

class RegressionLogistique:
    """
    Régression Logistique avec Gradient Descent
    Pour la classification binaire (0 ou 1)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Fonction sigmoïde: 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Entraîner le modèle
        X: features (n_samples, n_features)
        y: target (n_samples,) - valeurs 0 ou 1
        """
        n_samples, n_features = X.shape
        
        # Initialisation des paramètres
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Calcul linéaire: z = X*w + b
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Prédiction avec sigmoïde: y_pred = σ(z)
            y_pred = self.sigmoid(linear_model)
            
            # Calcul des gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul de la perte (Binary Cross-Entropy)
            epsilon = 1e-15  # Pour éviter log(0)
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1-y) * np.log(1 - y_pred_clipped))
            self.losses.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict_proba(self, X):
        """Prédire les probabilités"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Prédire les classes (0 ou 1)"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculer l'accuracy"""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


# ==================== EXEMPLE D'UTILISATION ====================

if __name__ == "__main__":
    # Générer des données synthétiques pour classification binaire
    np.random.seed(42)
    
    # Classe 0
    X_class0 = np.random.randn(100, 2) + np.array([2, 2])
    y_class0 = np.zeros(100)
    
    # Classe 1
    X_class1 = np.random.randn(100, 2) + np.array([5, 5])
    y_class1 = np.ones(100)
    
    # Combiner les données
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Mélanger les données
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Créer et entraîner le modèle
    model = RegressionLogistique(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # Prédictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Accuracy
    accuracy = model.score(X, y)
    print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ Poids (w): {model.weights}")
    print(f"✅ Biais (b): {model.bias:.4f}")
    
    # Visualisation
    plt.figure(figsize=(14, 5))
    
    # Graphique 1: Frontière de décision
    plt.subplot(1, 3, 1)
    
    # Tracer les points
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', alpha=0.5, label='Classe 0', edgecolors='k')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', alpha=0.5, label='Classe 1', edgecolors='k')
    
    # Tracer la frontière de décision
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Régression Logistique - Frontière de Décision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Évolution de la perte
    plt.subplot(1, 3, 2)
    plt.plot(model.losses, color='green')
    plt.xlabel('Itérations')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.title('Convergence du modèle')
    plt.grid(True, alpha=0.3)
    
    # Graphique 3: Distribution des probabilités
    plt.subplot(1, 3, 3)
    plt.hist(y_proba[y==0], bins=20, alpha=0.5, color='red', label='Classe 0', edgecolor='black')
    plt.hist(y_proba[y==1], bins=20, alpha=0.5, color='blue', label='Classe 1', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Seuil (0.5)')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Fréquence')
    plt.title('Distribution des probabilités')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_logistique_resultat.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Graphique sauvegardé: regression_logistique_resultat.png")
    
    # Matrice de confusion manuelle
    TP = np.sum((y_pred == 1) & (y == 1))  # True Positives
    TN = np.sum((y_pred == 0) & (y == 0))  # True Negatives
    FP = np.sum((y_pred == 1) & (y == 0))  # False Positives
    FN = np.sum((y_pred == 0) & (y == 1))  # False Negatives
    
    print("\n📊 Matrice de Confusion:")
    print(f"   True Positives (TP):  {TP}")
    print(f"   True Negatives (TN):  {TN}")
    print(f"   False Positives (FP): {FP}")
    print(f"   False Negatives (FN): {FN}")
    
    # Métriques
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Métriques:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
