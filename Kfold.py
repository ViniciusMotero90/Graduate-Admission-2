import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',8)
data = pd.read_csv('Admission_Predict.csv')

data.drop('Serial No.',axis=1,inplace=True)


y = data['Chance of Admit ']
x = data.drop('Chance of Admit ',axis=1)

modelo = LinearRegression()
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo,x,y,cv=kfold)
print(resultado.mean())