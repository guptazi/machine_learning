import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv('Prostate_cancer-training.csv')
X=df[['radius','texture','perimeter','area','smoothness','compactness','symmetry','fractal_dimension']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)
X_pca_with_clusters = np.column_stack((X_pca, clusters))


plt.figure(figsize=(8, 6))
plt.scatter(X_pca_with_clusters[:, 0], X_pca_with_clusters[:, 1], c=X_pca_with_clusters[:, 2], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA with K-means Clustering')
plt.colorbar(label='Cluster Label')
plt.show()

rec = []
k_range = range(1, 5) 

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)  # Use your PCA-reduced data
    rec.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, rec, '-o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

