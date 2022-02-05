# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:51:57 2021

@author: Gallo
"""

#Ajuste de parámetros p,d,q para proyección con el modelo ARIMA

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from dateutil import relativedelta

warnings.filterwarnings("ignore")

#Importar datos de demanda
data=pd.read_excel('SNI_MW_06-2020.xlsx')
datap=(data.set_index(['Fecha']))['2020-06-10':'2020-06-16']
# Evaluar un modelo ARIMA para una combinación (p,d,q)
def evaluate_arima_model(X, arima_order):
	# Preparar el cojunto de datos de entrenamiento
	train_size = len(datap)-48
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# Hacer predicciones
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# Calcular el error de predicción
	error = mean_squared_error(test, predictions)
	return error
 
# Evaluar combinaciones de valores p, d, q para un modelo ARIMA 
def evaluate_models(dataset, p_values, d_values, q_values):
    global best_cfg 
    dataset=dataset.astype('float32')
    best_score, best_cfg=float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
 
# Espacio de búsqueda
p_values = [0, 1, 2, 4, 6]
d_values = range(0, 3)
q_values = range(0, 3)

#Llamar a Búsqueda de Cuadrícula
evaluate_models(datap['Total'].values, p_values, d_values, q_values)


#Proyección de datos con el modelo ARIMA con parámetros definidos

#Controlar compoenente estacional para hacer estacionarios las series de tiempo
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
 
# Devolver componenete esacional
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
full=[]
X = data['Total'].values
days_in_year = 672#Definir mismos días de la semana anterior
differenced = difference(X, days_in_year)
# Ajustar modelo
model = ARIMA(differenced, order=(best_cfg[0],best_cfg[1],best_cfg[2]))
model_fit = model.fit(disp=0)
# Proyectar mediciones en un intervalo definido
forecast = model_fit.forecast(steps=672)[0]
# Invertir la diferencia estacionaria para obtener valores reales
history = [x for x in X]
day = 1
# Almacenar mediciones proyectadas y ordenarlas
for yhat in forecast:
    inverted = inverse_difference(history, yhat, days_in_year)
    print('Day %d: %f' % (day, inverted))
    history.append(inverted)
    day += 1
    full.append(inverted)
    proyeccion=pd.DataFrame(full)  
proyeccion.rename(columns={0:'Total'},inplace=True)

#Exportar datos a Excel 
t=pd.DataFrame(pd.date_range(data['Fecha'].max()+relativedelta.relativedelta(days=1), periods=48*14,freq=str(1800)+'S'))
t.rename(columns={0:'Fecha'},inplace=True) 
writer = pd.ExcelWriter('Proyección_ARIMA.xlsx', engine='xlsxwriter')
# Position the dataframes in the worksheet.
t.join(proyeccion).to_excel(writer, sheet_name='Semanal')  # Default position, cell A1.
writer.save() 
