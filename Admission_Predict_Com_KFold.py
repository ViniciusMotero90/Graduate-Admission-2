import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',8)
data = pd.read_csv('Admission_Predict.csv')

data.drop('Serial No.',axis=1,inplace=True)


y = data['Chance of Admit ']
x = data.drop('Chance of Admit ',axis=1)

def modelosregressao(a,b):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    modelo = LinearRegression()
    modeloRidge = Ridge()
    modeloLasso = Lasso()
    modeloElasticNet = ElasticNet()

    x = a
    y = b

    kfold = KFold(n_splits=10)

    resultadoLinearKfold = cross_val_score(modelo,x,y,cv=kfold)
    resultadoRidgeKfold = cross_val_score(modeloRidge,x,y,cv=kfold)
    resultadoLassoKfold = cross_val_score(modeloLasso,x,y,cv=kfold)
    resultadoElasticNetKfold = cross_val_score(modeloElasticNet,x,y,cv=kfold)

    dic_regmodels = {'Linear': resultadoLinearKfold.mean(), 'Ridge': resultadoRidgeKfold.mean(), 'Lasso': resultadoLassoKfold.mean(), 'ElasticNet': resultadoElasticNetKfold.mean()}
    melhor_modelo = max(dic_regmodels, key=dic_regmodels.get)
    print(f'Kfold na Regressão Linear: {resultadoLinearKfold.mean()} | Kfold no Ridge: {resultadoRidgeKfold.mean()} | Kfold no Lasso: {resultadoLassoKfold.mean()} | Kfold no ElasticNet {resultadoElasticNetKfold.mean()}')
    print(f'O melhor modelo foi {melhor_modelo} com o valor {dic_regmodels[melhor_modelo]}')
    
modelosregressao(x,y)