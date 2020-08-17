import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import helpers as h
from sklearn.preprocessing import StandardScaler

data=h.custommers
# removing the first id and gender columns
df=data[['Age', 'RevenuAnn','DepenseScore']]


# plot data after clustring (Age - Depence score)

plt.scatter(data[['Age']],data[['DepenseScore']],color='red')
plt.xlabel("Age")
plt.ylabel("Depense score")
plt.show()

# applying Kmeans algorithm

X = df.values[:,:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
#print("###### labels ###########")
#print(labels)
# adding the cluster label colomn to dataframe columns
data['cluster_label']=labels
#print(data.head(4))
# let plot data after clustring
fig = plt.figure(figsize=(6, 4))
# defining colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means.labels_))))
# ploting 
plt.scatter(X[:, 0], X[:, 2], c=labels.astype(np.float) )
plt.xlabel('Age')
plt.ylabel('DepenceScore')
plt.show()











