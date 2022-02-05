# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:21:14 2021

@author: Gallo
"""

#Ajuste de Hiperparámetros para proyección de demanda de potencia eléctrica

#%%
#Importar bibliotecas 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from hyperopt import fmin,tpe,hp,Trials, space_eval
from dateutil import relativedelta

#Importar datos de demanda 
dem=pd.read_excel('SNI_MW_06-2020.xlsx',sheet_name='demanda')
hor=dem.loc[:,'Fecha':'Hora']
hor['Fecha']=pd.to_datetime(hor['Fecha'])
hor['Hora']=pd.to_timedelta(hor['Hora'])
todo=pd.DataFrame(hor['Fecha']+hor['Hora'])
todo.rename(columns={0:'Fecha'},inplace=True)
dem1=todo.join(dem['Total'])
dat_en=dem1['Fecha'].max()+relativedelta.relativedelta(weeks=-1)+relativedelta.relativedelta(seconds=1800)
dem1=(dem1.set_index(['Fecha']))[dat_en:dem1['Fecha'].max()]


#Retrasos y Ingeniería de caracerísticas
def ing_carc(dem):
    global label_col
    label_col='y'#Label de la variable objetivo 
    data=pd.DataFrame(dem['Total'].copy())#copio datos de demanda de dem1
    data.columns=[label_col]
    #Inferir retrasos 
    for i in range(1,49):
        data['lag_{}'.format(i)]=data[label_col].shift(i)
    #Procedimiento de Ingeniería de Caracter´siticas 
    data['hour']=data.index.hour
    data['minutos']=data.index.minute
    data['mes']=data.index.month
    data['weekday']=data.index.weekday
    
    #Etiquetar comportamientos 
    conds=[(data['weekday']==5),
          (data['weekday']==6)]
    choices=[100,101]
    data['is_weekend']=np.select(conds,choices)


    data = data.dropna()
    lag_cols=[col for col in data.columns if 'lag' in col]
    #Demanda promedio, mínima, y máxima
    data['d_prom']=data[lag_cols].mean(axis=1)
    data['d_max']=data[lag_cols].max(axis=1)
    data['d_min']=data[lag_cols].min(axis=1)
   
    #Extraer varaibles características y varaible objetivo
    y=(data[label_col].values)
    data2=data.drop(label_col,axis=1)
    X=data2.values
    return X,y,data

X,y,data=ing_carc(dem1)

#Ajustar hiperparámetros 
random_state=2020#semilla de aleatoriedad 
n_iter=50
#Separar los datos de entrenamiento y prueba 
test_size=0.11
test_index=int(len(X)*(1-test_size))
train_data=X[:test_index]
train_targets=y[:test_index]
num_folds=2
kf=KFold(n_splits=num_folds,random_state=random_state,shuffle=True)
def opt_r2(params,random_state=random_state,cv=kf,X=train_data,y=train_targets):
    #Paramatros a evaluar 
    params={'n_estimators':int(params['n_estimators']),
            'max_depth':int(params['max_depth']),
            'max_features': int(params['max_features']),
            'criterion':params['criterion']}
    #Ajustar modelo
    model=RandomForestRegressor(random_state=random_state,**params)
    #Evaluar el modelo 
    score=-cross_val_score(model,X,y,cv=cv,scoring='r2',n_jobs=-1).mean()
    
    return score 
#Dominio Busqueda 
space={'n_estimators':hp.quniform('n_estimators',1,50,1),
       'max_depth':hp.quniform('max_depth',2,20,1),
       'max_features':hp.quniform('max_features',1,40,1),
       'criterion':hp.choice('criterion',['squared_error'])}
#Historial de evaluación 
trials=Trials()

best=fmin(fn=opt_r2,space=space,algo=tpe.suggest,max_evals=n_iter,trials=trials,
          )

#Almacenar mejor hiperparámetros

best_params_rf=pd.DataFrame((space_eval(space,best)).values())

model2=RandomForestRegressor(random_state=random_state, criterion=best_params_rf[0][0],
                             max_depth=int(best_params_rf[0][1]),
                             max_features=int(best_params_rf[0][2]),
                             n_estimators=int(best_params_rf[0][3]))
score2=cross_val_score(model2,X,y,cv=kf,scoring='r2',n_jobs=-1).mean()
print("Best r2 {:.3f} params{}".format(score2,best))


#Métrica de precisión 

def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean((np.abs(y_true-y_pred))/y_true)*100

#Apredinzaje supervisadeo 
size_fore=672# tamaño de datos de MW a proyectarse
dem2=todo.join(dem['Total'])
start_date=dem2['Fecha'].max()+relativedelta.relativedelta(days=-15)+relativedelta.relativedelta(seconds=1800)
dem2=(dem2.set_index(['Fecha']))[start_date:dem2['Fecha'].max()]

X, y, data =ing_carc(dem2)
x_forec=data[-size_fore:]
x_forec=x_forec.drop('y',axis=1)

#Seprar sets de entrneamiento y prueba 
def timeseries_train_test_split(X,y,test_size):
    global X_train,X_test,y_train,y_test,test_index
    test_index=int(len(X)*(1-test_size))
    X_train=X[:test_index]
    X_test=X[test_index:]
    y_train=y[:test_index]
    y_test=y[test_index:]
    return X_train, X_test, y_train, y_test
def llamar_modelo(X,y,test_size=0.11):
    global prediction
    X_train,X_test,y_train,y_test=timeseries_train_test_split(X,y,test_size)
#Definir el algorimto de aprendizaje supervisado 
    model=RandomForestRegressor(criterion=best_params_rf[0][0],
                             max_depth=int(best_params_rf[0][1]),
                             max_features=int(best_params_rf[0][2]),
                             n_estimators=int(best_params_rf[0][3]))
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    return model

model=llamar_modelo(X,y)
#Proyectar la demanda (2 semanas)
dias_imput=pd.DataFrame(model.predict(x_forec))
dias_imput.rename(columns={0:'Total'},inplace=True)
dias_imput.index=np.arange(len(dias_imput))
    
plt.figure(figsize=(8,5))
x=range(prediction.size)
plt.plot(x,y_test,label='real',linewidth=2.0)
plt.plot(x,prediction,label='proyección',linewidth=2.0)
error=mean_absolute_percentage_error(prediction,y_test)
plt.title('Porcentaje de Errror Medio Absoluto {0:.2f}%'.format(error))
plt.legend(loc='best')
plt.tight_layout()
plt.grid(True)

#Escribir Excel 
t=pd.DataFrame(pd.date_range(dem1.index.max()+relativedelta.relativedelta(seconds=1800),periods=48*14,freq=str(1800)+'S'))
t.rename(columns={0:'Fecha'},inplace=True)
writer=pd.ExcelWriter('Proyección_RF.xlsx',engine='xlsxwriter')
t.join(dias_imput).to_excel(writer,sheet_name='MW')
writer.save()