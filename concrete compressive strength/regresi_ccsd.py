import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#%%
data_x=train.drop(['Id'],axis=1)
data_y=train['Concrete compressive strength(MPa, megapascals)']
data_x.describe()
#%%
reg=linear_model.LinearRegression()
reg.fit(data_x, data_y)
reg.coef_

prediksi=reg.predict(test)
regresi=pd.DataFrame()
regresi['Id']=test.Id
regresi['Concrete compressive strength(MPa, megapascals)']=prediksi
regresi.to_csv('regresi.csv', index=False)