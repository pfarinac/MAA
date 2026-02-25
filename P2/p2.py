import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mnist_reader
from sklearn.manifold import TSNE


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)


fig , axes = plt.subplots(2, 5, figsize = (10,5))

for i,ax in  enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Etiqueta real: {y_train[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


unique, counts = np.unique(y_train, return_counts=True)

plt.bar(unique, counts)
plt.xlabel("Clase")
plt.ylabel("Número de muestras")
plt.title("Distribución de clases en Fashion-MNIST")
plt.show()

print("Valor mínimo:", X_train.min())
print("Valor máximo:", X_train.max())
print("Media:", X_train.mean())
print("Desviación estándar:", X_train.std())

from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train)

print("Varianza explicada acumulada:",
      np.sum(pca.explained_variance_ratio_))



pca_2d = PCA(n_components=2, random_state=42)


plt.figure(figsize=(6,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], s=1, alpha=0.3)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proyección PCA (2D) de Fashion-MNIST")
plt.show()