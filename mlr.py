import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('taxi.csv')
# print(df.head())

x = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values
# print(x)
# print(y)

X_train,X_test,y_train,y_test =train_test_split(x , y ,test_size = 0.3 , random_state = 0)

reg = LinearRegression()
reg.fit(X_train,y_train)

print("Training Score : ",reg.score(X_train,y_train))
print("Testing Score : ",reg.score(X_test,y_test))

pickle.dump(reg , open("taxi_data.pkl" , 'wb'))

model = pickle.load(open("taxi_data.pkl" , 'rb'))
print(model.predict([[80 , 1770000 , 6000,  80]]))   #   Priceperweek , Population , Monthlyincome , Averageparkingpermonth 