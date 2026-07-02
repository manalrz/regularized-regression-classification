###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
###############################################################################

###############################################################################
# Download DATA
data1 = np.load('data1.npy')
data2 = np.load('data2.npy')
data3 = np.load('data3.npy')

# Fonction pour afficher les données et créer la régression linéaire
def affiche(data, title):

    X = data[0].reshape(-1,1)
    Y = data[1]

    # Affichage nuage de points
    plt.scatter(data[0], data[1], label='Données', color='blue')

    # Régression linéaire
    model = LinearRegression()
    model.fit(X,Y)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    Y_prediction = model.predict(X_range)

    # Calcul de l'erreur d'apprentissage
    n = len(Y)
    erreur_apprentissage = (1 / n) * np.sum((Y - Y_prediction) ** 2)

    # Traçage de la régression
    plt.plot(X_range, Y_prediction, label='Ligne de régression', color='red')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    print(f"Erreur d'apprentissage pour {title}: {erreur_apprentissage}")
    return erreur_apprentissage

def polynomiale(data, q, title):

    X = data[0].reshape(-1,1)
    Y = data[1]

    # Transformation polynomiale
    poly = PolynomialFeatures(degree=q, include_bias=True)
    PHI_poly = poly.fit_transform(X)

    # Résolution de la régression OLS
    beta = np.linalg.inv(PHI_poly.T @ PHI_poly) @ PHI_poly.T @ Y

    # Prédictions
    Y_prediction = PHI_poly @ beta

    # Calcul de l'erreur d'apprentissage
    n = len(Y)
    erreur_apprentissage = (1 / n) * np.sum((Y - Y_prediction) ** 2)

    # Affichage nuage de points
    plt.scatter(data[0], data[1], label='Données', color='blue')
    
    # Traçage de la régression
    plt.plot(np.sort(X, axis=0), np.sort(Y_prediction, axis=0), label='Polynomiale', color='red')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    print(f"Erreur d'apprentissage pour {title}: {erreur_apprentissage} avec comme degré {q}")
    return erreur_apprentissage

def ridge(data, lambd, title):

    X = data[0].reshape(-1,1)
    Y = data[1]

    n = len(Y)

    # Ajouter une colonne de biais (1) à X
    X_bias = np.hstack([np.ones((n, 1)), X])
    
    # Matrice d'identité (régularisation, pas de régularisation sur le biais)
    I = np.eye(X_bias.shape[1])
    I[0, 0] = 0

    # Résolution de ridge
    beta = np.linalg.inv(X_bias.T @ X_bias + n * lambd * I) @ (X_bias.T @ Y)

    # Prédictions
    Y_prediction = X_bias @ beta

    # Calcul de l'erreur d'apprentissage
    erreur_apprentissage = (1 / n) * np.sum((Y - Y_prediction) ** 2)

    # Affichage nuage de points
    plt.scatter(data[0], data[1], label='Données', color='blue')

    # Traçage de la régression
    plt.plot(np.sort(X, axis=0), np.sort(Y_prediction, axis=0), label='Ridge', color='red')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    print(f"Erreur d'apprentissage pour {title}: {erreur_apprentissage} avec régularisation λ = {lambd}")
    return erreur_apprentissage

def lasso(data, lambd, title):
    X = data[0].reshape(-1,1)
    Y = data[1]

    # Modèle Lasso avec régularisation
    lasso = Lasso(alpha=lambd, fit_intercept=True)
    lasso.fit(X, Y)

    # Prédictions
    Y_prediction = lasso.predict(X)

    # Calcul de l'erreur d'apprentissage
    n = len(Y)
    erreur_apprentissage = (1 / n) * np.sum((Y - Y_prediction) ** 2)

    # Affichage nuage de points
    plt.scatter(data[0], data[1], label='Données', color='blue')

    # Traçage de la régression
    plt.plot(np.sort(X, axis=0), np.sort(Y_prediction, axis=0), label='Lasso', color='red')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    print(f"Erreur d'apprentissage pour {title}: {erreur_apprentissage} avec régularisation λ = {lambd}")
    return erreur_apprentissage
    
# affiche(data1, 'data 1 : Régression Linéaire OLS')
# affiche(data2, 'data 2 : Régression Linéaire OLS')
#polynomiale(data2, 10, 'data 2 : polynomiale')
#polynomiale(data3, 10, 'data 3 : polynomiale')
# question 5 : faire attention au nombres de points, il faudrait prendre ne compte la taille du nuage de points dans le calcul de l'erreur d'apprentissage 
#ridge(data3, 0.01, 'data 3 : ridge')
#lasso(data3,0.01, 'data 3 : lasso')
# Utiliser Ridge : Lorsque toutes les variables sont pertinentes et corrélées entre elles. 
# Les coefficients sont réduits de manière continue en fonction de λ, mais aucune variable n'est complètement supprimée.
# Utiliser Lasso : Lorsque certaines variables sont inutiles et peuvent être exclues. 
# En augmentant λ, certains coefficients deviennent exactement nuls, ce qui réduit le nombre de variables dans le modèle.

