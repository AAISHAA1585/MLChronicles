import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample Data: Annual Income ($000) vs. Spending Score (1-100)
X = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76], [21, 6]])

# Train K-Means Model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Predict cluster labels
labels = kmeans.predict(X)

# Visualize Clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", label="Clusters")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="X", label="Centroids")
plt.xlabel("Annual Income ($000)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
