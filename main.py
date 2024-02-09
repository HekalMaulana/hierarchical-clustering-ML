import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Find ideal number of cluster with dendogram method
# import scipy.cluster.hierarchy as sch
# dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
# plt.title("Dendogram")
# plt.xlabel("Customers")
# plt.ylabel("Euclidean distance")
# plt.show()

ideal_n_cluster = 5

# Training the hierarchical clustering model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering( n_clusters= ideal_n_cluster, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Random Color
colors = np.random.rand(ideal_n_cluster, 3)

# Visualising Hierarchical clustering model
for cluster_num in range(ideal_n_cluster):
    plt.scatter(X[y_hc == cluster_num, 0], X[y_hc == cluster_num,1], s=100, c=[colors[cluster_num]], label = f"Cluster {cluster_num}")
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()