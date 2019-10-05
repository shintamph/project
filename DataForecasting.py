# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:11:42 2019

@author: Shinta
"""

import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 2
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


#importing data
climate = pd.read_csv(r'D:\KURSUS\climate.csv')
climate.info()

#mengubah tipe data attribute date manjadi datetime, meantemp menjadi integer, men-select attribute date dan meamtemp
climate.date = pd.to_datetime(climate.date)
cl=climate.iloc[:,0:2]
cl['meantemp'] = cl['meantemp'].astype(int)
cl= cl.iloc[0:112,:,]
cl.info()

#mengubah date menjadi index
yy = cl.set_index(['date'])
yy.head(5)

#melihat plot meantemp
yy.plot(figsize=(10, 2))
plt.show()

#menggunakan command "sm.tsa.seasonal_decompose" untuk mendekomposisikan runtun data menjadi 3 bagian, yaitu seasonality, trend, dan noise
from pylab import rcParams
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(yy, model='additive')
fig = decomposition.plot()
plt.show()

#membuat kombinasi model SARIMA menggunakan fungsi itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]
print('Contoh model acak SARIMA dengan kombinasi p,d,q= range (0,2)')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


#Mencari nilai Akaike terkecil untuk memnentukan model SARIMA terbaik
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(yy,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}30 - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            continue
		#didapat nilai akaike terkecil pada ARIMA(1, 1, 0)x(1, 1, 0, 30)30 sebesar AIC:207.3803036843613


#Menguji asumsi kenormalam sebelum melakukan forecasting
modd = sm.tsa.statespace.SARIMAX(yy,
                                order=(1, 1, 0),
                                seasonal_order=(1, 1, 0, 30),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = modd.fit()
print(results.summary().tables[1])

#menampilkan grafik kenormalan
results.plot_diagnostics(figsize=(20, 10))
plt.show()

#Langkah ini terdiri dari membandingkan nilai sebenarnya dengan prediksi perkiraan. Prakiraan kami cocok dengan nilai sebenarnya dengan sangat baik.
pred = results.get_prediction(start=pd.to_datetime('2017-04-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = yy['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(10, 3))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('temp')
plt.legend()
plt.show()


yy_forecasted = pred.predicted_mean
yy_truth = yy['2017-04-01':]
mse = ((yy_forecasted - yy_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

#Di sini memperkirakan rata-rata temperature untuk 30 hari ke depan
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = yy.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('temp')
plt.legend()
plt.show()

#Langkah ini menunjukkan nilai prediksi tes yang telah dijalankan sebelumnya.
yy_forecasted = pred.predicted_mean
yy_forecasted.head(30)

#Langkah ini menunjukkan true value dari meantemp. Kita dapat membandingkan dua seri di atas untuk mengukur akurasi model.
yy_truth.head(30)

#menampilkan kemungkinan nilai terkecil dan terbesar dari forcesting data
pred_ci.head(30)

#mengukur akurasi (masih error)
accuracy_score(yy_truth, yy_forecasted)

#Memprediksi forecasting meantemp untuk 30 hari kedepan
forecast = pred_uc.predicted_mean
forecast.head(15)

