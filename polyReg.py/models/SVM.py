import pandas as pd
from sklearn import preprocessing
from models import processing
from sklearn import metrics,svm

pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)

X_train = processing.X_train_data
y_train = processing.Y_train_data
X_test = processing.X_test_data
y_test = processing.Y_test_data

# normalize data
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
# training model
SVM_model=svm.SVC(kernel='rbf').fit(X_train,y_train)

# appluing model
Y_predected=SVM_model.predict(X_test)
print(Y_predected[0:5])
# showing accaurancy ( in individual case )
#print("Train set Accuracy: ", metrics.accuracy_score(y_train, SVM_model.predict(X_train)))
Accuracy = metrics.accuracy_score(y_test, Y_predected)