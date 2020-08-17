import pandas as pd
from sklearn import preprocessing
from models import processing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)

X_train = processing.X_train_data
y_train = processing.Y_train_data
X_test = processing.X_test_data
y_test = processing.Y_test_data


# normalize data
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))

# build the model
# cheking for the best choice of K
#for i in range(2,10,1):
#    KNN_model = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
#    Y_predected = KNN_model.predict(X_test)
#    print("Test set Accuracy for", i, ": ", metrics.accuracy_score(y_test, Y_predected))

# after runing k=2 is the best choice

KNN_model = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
# applied the model on test data
Y_predected = KNN_model.predict(X_test)
#print(Y_predected[0:5])

# showing accaurancy ( in individual case )
#print("Train set Accuracy: ", metrics.accuracy_score(y_train, KNN_model.predict(X_train)))
Accuracy = metrics.accuracy_score(y_test, Y_predected)
