import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

data=pd.read_csv("houseprice.csv")
data=pd.DataFrame(data)
data = pd.get_dummies(data) 
data.dropna(inplace=True)

model=LinearRegression()
x=data.drop(columns=["median_house_value"])
y=data["median_house_value"]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)        
model.fit(x_train,y_train)
y_pred=model.predict(x_test)    

#print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
#print("R^2 Score:", r2_score(y_test, y_pred))   
