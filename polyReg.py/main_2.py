import helpers
from models import SVM,KNN,logisticRegression,decisionTree

# Understanding data using chart
data= helpers.test_data

# classification using KNN model
print(" ##### Classification using KNN model #################")
y_predected=KNN.Y_predected
for i in range(10):
    print(f"passeneger with id :{data.iloc[i]['id']} real satisfaction : {data.iloc[i]['satisfaction']} , pretected one :{y_predected[i]}")
print("accurancy of KNN model :", KNN.Accuracy)
print("###########################################")
print("\n\n\n\n")

# classification using SVM model
print(" ##### Classification using SVM model #################")
y_predected=SVM.Y_predected
for i in range(10):
    print(f"passeneger with id :{data.iloc[i]['id']} real satisfaction : {data.iloc[i]['satisfaction']} , pretected one :{y_predected[i]}")
print("accurancy of SVM model :", SVM.Accuracy)
print("###########################################")
print("\n\n\n\n")

# classification using LR model
print(" ##### Classification using logistic Regression model #################")
y_predected=logisticRegression.Y_predected
for i in range(10):
    print(f"passeneger with id :{data.iloc[i]['id']} real satisfaction : {data.iloc[i]['satisfaction']} , pretected one :{y_predected[i]}")
print("accurancy of logistic Regression model :", logisticRegression.Accuracy)
print("###########################################")
print("\n\n\n\n")


# classification using LR model
print(" ##### Classification using decision tree model #################")
y_predected=decisionTree.Y_predected
for i in range(10):
    print(f"passeneger with id :{data.iloc[i]['id']} real satisfaction : {data.iloc[i]['satisfaction']} , pretected one :{y_predected[i]}")
print("accurancy of decision tree  model :", decisionTree.Accuracy)
print("###########################################")
print("\n\n\n\n")