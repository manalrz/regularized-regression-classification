###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='auto')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 
###############################################################################
# CLASSIFICATION
###############################################################################
# Normalisation de données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# entrainement des données
logistic_clf = LogisticRegression() # penalty='l1', solver='saga', multi_class='multinomial'
logistic_clf.fit(X_train_scaled, y_train)

# prédictions et calcul de la matrice de confusion
y_pred = logistic_clf.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcul coefficent beta logistique
beta_logistic = logistic_clf.coef_

###############################################################################
# DISPLAY A SAMPLE
###############################################################################
# Affichage 
m=16
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  plt.title("Label = %i" % lb[i])
plt.show()

# Affichage coefficients beta
plt.figure(figsize=(10,10))
for i in np.arange(10):
  ex_plot = plt.subplot(2, 5, i + 1)
  plt.imshow(beta_logistic[i].reshape((28,28)), cmap='RdBu')
  plt.title(f"Classe {i}")
plt.show()

# Affichage matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logistic_clf.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Matrice de Confusion - Régression Logistique")
plt.show()

