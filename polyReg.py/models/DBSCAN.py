import numpy as np
import matplotlib.pyplot as plt
from models import clustring as c
from sklearn.cluster import DBSCAN


# importing data from clustring model ( already processed)
data=c.Clus_dataSet
X=c.X
db_default = DBSCAN(eps = 0.55, min_samples = 4).fit(data)
labels = np.array(db_default.labels_)

fig = plt.figure(figsize=(6, 4))
# defining colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
# ploting
plt.scatter(X[:, 1], X[:, 2], c=labels.astype(np.float) )
plt.title("DBSCAN clustring")
plt.xlabel('RevenuAnn')
plt.ylabel('DepenceScore')
plt.show()


print(labels)