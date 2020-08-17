import numpy as np
import matplotlib.pyplot as plt
import helpers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data=helpers.custommers

a=data.get('Age').values.reshape(-1,1)
b=data.get('DepenseScore').values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(a,b,test_size=0.2,random_state=42)
# declaring linear regression class
lr= LinearRegression()
# training model
lr.fit(X_train,Y_train)
# show result
intercept = lr.intercept_
coef = lr.coef_
y_red=lr.predict(X_test)


# ploting data and regresson line
plt.scatter(X_train,Y_train,edgecolors='red')
plt.plot(X_train, lr.coef_[0][0]*X_train + lr.intercept_[0], '-r')
plt.xlabel("Age")
plt.ylabel("facteur de depence ")
plt.show()


RMSE = np.sqrt(metrics.mean_squared_error(Y_test,y_red))




