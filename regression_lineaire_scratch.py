"""
Implémentation de la Régression Linéaire à partir de zéro
Sans utiliser scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt

class RegressionLineaire:
    """
    Régression Linéaire avec Gradient Descent
    Formule: y = w*X + b
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """
        Entraîner le modèle
        X: features (n_samples, n_features)
        y: target (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialisation des paramètres
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Prédiction: y_pred = X*w + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calcul des gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul de la perte (MSE)
            loss = np.mean((y_pred - y)**2)
            self.losses.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict(self, X):
        """Prédire les valeurs"""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculer le R² score"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        return r2


# ==================== EXEMPLE D'UTILISATION ====================

if __name__ == "__main__":
    # Générer des données synthétiques
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Créer et entraîner le modèle
    model = RegressionLineaire(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # Prédictions
    y_pred = model.predict(X)
    
    # Score R²
    r2 = model.score(X, y)
    print(f"\n✅ R² Score: {r2:.4f}")
    print(f"✅ Poids (w): {model.weights}")
    print(f"✅ Biais (b): {model.bias:.4f}")
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    # Graphique 1: Données et ligne de régression
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Données réelles')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Ligne de régression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Régression Linéaire')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Évolution de la perte
    plt.subplot(1, 2, 2)
    plt.plot(model.losses, color='green')
    plt.xlabel('Itérations')
    plt.ylabel('Loss (MSE)')
    plt.title('Convergence du modèle')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_lineaire_resultat.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Graphique sauvegardé: regression_lineaire_resultat.png")
