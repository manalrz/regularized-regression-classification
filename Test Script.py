import numpy as np
import matplotlib.pyplot as plt

# Charger les données
data = np.load("data3.npy")
X = data[0, :]  # Caractéristiques (features)
y = data[1, :]  # Cibles (targets)

# Fonction pour créer les transformations polynomiales
def polynomial_features(X, degree):
    """Génère des features polynomiales jusqu'à l'ordre donné."""
    return np.vstack([X**i for i in range(degree + 1)]).T

# Fonction pour la régression OLS dans un espace transformé
def ols_with_transformation(X, y, phi):
    """Ajuste une régression OLS avec transformation des features."""
    X_phi = phi(X)
    beta = np.linalg.inv(X_phi.T @ X_phi) @ X_phi.T @ y
    return beta

# Tester avec un modèle polynomial de degré ajustable
def polynomial_regression_plot(X, y, degree):
    # Générer les features polynomiales
    phi = lambda X: polynomial_features(X, degree)
    X_phi = phi(X)
    
    # Calcul des coefficients OLS
    beta = ols_with_transformation(X, y, phi)
    
    # Calcul des prédictions
    y_pred = X_phi @ beta
    
    # Calcul de l'erreur quadratique moyenne
    error = np.mean((y - y_pred) ** 2)
    
    # Visualisation
    plt.scatter(X, y, label="Données", color="blue")
    x_plot = np.linspace(X.min(), X.max(), 500)
    y_plot = phi(x_plot) @ beta
    plt.plot(x_plot, y_plot, color="red", label=f"Modèle polynomial (degré {degree})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Régression polynomial (degré {degree})\nErreur quadratique moyenne: {error:.4f}")
    plt.show()
    
    print(f"Coefficients pour le degré {degree} : {beta}")
    print(f"Erreur d'apprentissage : {error:.4f}")

# Ajuster le modèle pour différents degrés
for degree in [1, 3, 5, 10]:
    print(f"=== Modèle polynomial de degré {degree} ===")
    polynomial_regression_plot(X, y, degree)