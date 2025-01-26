import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',8)
data = pd.read_csv('Admission_Predict.csv')

data.drop('Serial No.',axis=1,inplace=True)


y = data['Chance of Admit ']
x = data.drop('Chance of Admit ',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=45)

def modelosregressao(a,b,c,d):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet

    x_train = a
    y_train = b
    x_test = c
    y_test = d

    modelo = LinearRegression()
    modeloRidge = Ridge()
    modeloLasso = Lasso()
    modeloElasticNet = ElasticNet()

    modelo.fit(x_train,y_train)
    modeloRidge.fit(x_train,y_train)
    modeloLasso.fit(x_train,y_train)
    modeloElasticNet.fit(x_train,y_train)

    resultado = modelo.score(x_test,y_test)
    resultadoRidge = modeloRidge.score(x_test,y_test)
    resultadoLasso = modeloLasso.score(x_test,y_test)
    resultadoElasticNet = modeloElasticNet.score(x_test,y_test)

    dic_regmodels = {'Linear': resultado, 'Ridge': resultadoRidge, 'Lasso': resultadoLasso, 'ElasticNet': resultadoElasticNet}
    melhor_modelo = max(dic_regmodels, key=dic_regmodels.get)
    print('Regressão Linear: ', resultado, 'Regressão Ridge: ', resultadoRidge, 'Regressão Lasso: ', resultadoLasso, 'Regressão ElasticNet', resultadoElasticNet)
    print(f'O melhor modelo foi {melhor_modelo} com o valor {dic_regmodels[melhor_modelo]}')
modelosregressao(x_train,y_train,x_test,y_test)