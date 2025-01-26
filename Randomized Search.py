import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet

pd.set_option('display.max_columns',8)
data = pd.read_csv('Admission_Predict.csv')

data.drop('Serial No.',axis=1,inplace=True)


y = data['Chance of Admit ']
x = data.drop('Chance of Admit ',axis=1)

valores = {'alpha': [0.1,0.5,1,2,5,10,25,50,100,150,200,300,500,750,1000,1500,2000,3000,5000],'l1_ratio': [0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

modelo = ElasticNet()
procura = RandomizedSearchCV(estimator=modelo,param_distributions=valores,n_iter=150,cv=5,random_state=15)
procura.fit(x,y)

print('Melhor score: ', procura.best_score_)
print('Melhor Alpha: ', procura.best_estimator_.alpha)
print('Melhor L1_ratio: ', procura.best_estimator_.l1_ratio)