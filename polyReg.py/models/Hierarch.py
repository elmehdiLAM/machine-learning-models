from sklearn.cluster import AgglomerativeClustering
from models import clustring as c
import matplotlib.pyplot as plt
import numpy as np


# importing data from clustring model ( already processed)
X=c.X
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
labels=np.array(hc.fit_predict(X))

fig = plt.figure(figsize=(6, 4))
# defining colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
# ploting
plt.scatter(X[:, 1], X[:, 2], c=labels.astype(np.float) )
plt.title("hiarchique clustring")
plt.xlabel('RevenuAnn')
plt.ylabel('DepenceScore')
plt.show()


