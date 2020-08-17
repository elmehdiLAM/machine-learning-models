import helpers
from models import polynomialLinearRegression, LinearRegression,clustring,MultipleLinearRegression
import matplotlib.pyplot as plt
import seaborn as sb



# Undestanding data using chart
data=helpers.custommers


# gender statistcs
plt.figure(1 , figsize = (15 , 5))
sb.countplot(x="Genre",  data=data)
plt.show()


# ploting 3,3 matrix of important columns
plt.figure(1 , figsize = (15 , 7))
n = 0
for x in data.columns[2:-1]:
    for y in data.columns[2:-1]:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sb.regplot(x=x, y=y, data=data)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
plt.show()

# clustring using Kmeans algorithm

new_data=clustring.data
print(" Kmeans clustring ")
print("######## the last columns represent the class of item ############")
print(new_data)
print("###########################################")
# we alredy plot data on Linearregression file
print("\n\n\n\n\n\n")


# regression of the column < Depense score > using linear regression
print("###### using Linear regression model we found  ########")
y_red=LinearRegression.y_red
Y_test=LinearRegression.Y_test
for i in range(len(y_red)):
    print(f"real value : {Y_test[i]} , predected value : {y_red[i]}")
print("intercept value :", LinearRegression.intercept)
print("coeficient value :", LinearRegression.coef)
print("RMSE score: ", LinearRegression.RMSE)
print("###########################################")
print("\n\n\n\n")



# regression of the column < Depense score > using multiple regression
print("###### using multiple regression model we found  ########")
y_pred=MultipleLinearRegression.y_red
Y_test=MultipleLinearRegression.Y_test
for i in range(len(y_red)):
    print(f"real value : {Y_test[i]} , predected value : {y_red[i]}")
print("intercept value :", MultipleLinearRegression.intercept)
print("coeficient value :", MultipleLinearRegression.coef)
print("RMSE score: ", MultipleLinearRegression.RMSE)
print("###########################################")
print("\n\n\n\n")


# regression of the column < Depense score > using polynomial regression
print("###### using polynomial regression model we found  ########")
y_pred=polynomialLinearRegression.y_red
Y_test=polynomialLinearRegression.Y_test
for i in range(len(y_red)):
    print(f"real value : {Y_test[i]} , predected value : {y_red[i]}")
print("intercept value :", polynomialLinearRegression.intercept)
print("coeficient value :", polynomialLinearRegression.coef)
print("RMSE score: ", polynomialLinearRegression.RMSE)
print("###########################################")
print("\n\n\n\n")