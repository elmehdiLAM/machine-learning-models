import pandas as pd
import os

# load data from data directory
custommers=pd.read_csv("C:/Users/lamsy/Desktop/mini_projet_ML/polyReg.py/data/Mall_Customers.csv")
train_data=pd.read_csv("C:/Users/lamsy/Desktop/mini_projet_ML/polyReg.py/data/train.csv")
test_data=pd.read_csv("C:/Users/lamsy/Desktop/mini_projet_ML/polyReg.py/data/test.csv")

custommers.columns=['ID','Genre','Age','RevenuAnn','DepenseScore']
# devide in two part for good practice
under_data=custommers.where(custommers.get('Age')<= 35).dropna()
over_data=custommers.where(custommers.get('Age')>35).dropna()






