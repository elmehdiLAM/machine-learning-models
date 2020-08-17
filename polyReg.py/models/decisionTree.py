import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from models import processing
from sklearn import metrics,svm

# to print all data on console
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)

# importing data
X_train = processing.X_train_data
y_train = processing.Y_train_data
X_test = processing.X_test_data
y_test = processing.Y_test_data

#intialising model
DT_model = DecisionTreeClassifier(criterion="entropy")
print(DT_model)

# training model
DT_model.fit(X_train,y_train)

# predecting class from test dataset
Y_predected=DT_model.predict(X_test)
print(Y_predected[0:5])
# showing accaurancy ( in individual case )
#print("Train set Accuracy: ", metrics.accuracy_score(y_train, DT_model.predict(X_train)))
Accuracy = metrics.accuracy_score(y_test, Y_predected)