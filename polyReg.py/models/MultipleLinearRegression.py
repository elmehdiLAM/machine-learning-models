import pandas as pd
import numpy as np
import helpers as h
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=h.custommers
df=pd.get_dummies(data)

X=np.asarray(df[['Age', 'RevenuAnn', 'Genre_Female', 'Genre_Male']])
y=np.asarray(df[['DepenseScore']])


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)


lr= LinearRegression()
# training model
lr.fit(X_train,Y_train)
# show result
intercept = lr.intercept_
coef = lr.coef_
y_red=lr.predict(X_test)
# computing RMSE score
RMSE = np.sqrt(metrics.mean_squared_error(Y_test,y_red))

our_test_value=(np.asarray([65, 43, 0, 1]))
