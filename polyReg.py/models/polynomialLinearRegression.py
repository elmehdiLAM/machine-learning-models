import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import helpers
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = helpers.custommers

a=data.get('Age').values.reshape(-1,1)
b=data.get('DepenseScore').values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(a,b,test_size=0.2,random_state=42)
# appling polynomial matrix
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(X_train)

# declaring linear regression class
plr= LinearRegression()

plr.fit(train_x_poly, Y_train)

intercept = plr.intercept_
coef = plr.coef_
X_test_poly= poly.fit_transform(X_test)
y_red=plr.predict(X_test_poly)


RMSE =np.sqrt(metrics.mean_squared_error(Y_test,y_red))