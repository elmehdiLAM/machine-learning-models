import pandas as pd
import numpy as np
import helpers

train_data=helpers.train_data
test_data=helpers.test_data

pd.set_option('display.max_rows',train_data.shape[0]+1)
pd.set_option('display.max_columns',train_data.shape[1]+1)

# data shapes
#print(train_data.shape)
satisfied_number=len(train_data.where(train_data['satisfaction'] == 'satisfied').dropna())
dis_neur_number=len(train_data.where(train_data['satisfaction'] == 'neutral or dissatisfied').dropna())
#print(f"satisfied : {satisfied_number} , neural or disatisfied : {dis_neur_number}")

# ploting


# understanding data
data=pd.concat([train_data,test_data])
#print(data.describe())
#print(data.shape)


# spliting data function (x2)
def split_data(donne):
    y = donne.get('satisfaction')
    y=[1 if e == 'satisfied' else 0 for e in y ]
    y=np.asarray(y)
    donne["Arrival Delay in Minutes"] = donne["Arrival Delay in Minutes"].fillna(np.mean(donne["Arrival Delay in Minutes"]))
    x_temp=donne.iloc[:,2:-1]
    x=pd.get_dummies(x_temp)
    return x, y


X_train_data,Y_train_data=split_data(train_data)
X_test_data,Y_test_data=split_data(test_data)




