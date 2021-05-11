# %%
# imports
import pandas as pd
import numpy as np
# plots
import matplotlib.pyplot as plt
from darkstyle import dark_style as dks
# statistic
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
# modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
# automation sarimax
import pmdarima as pm
# ignore
import warnings
warnings.filterwarnings('ignore')

#%%
# function 1


def settings():
	"""
	Function to set plot styles and pandas params
	"""

	dks.dark_style()
	plt.rcParams['figure.figsize'] = [10, 6]
	plt.rcParams['font.size'] = 8
	pd.options.display.max_columns = None
	pd.set_option('display.expand_frame_repr', False)

settings()

# %%
# dataset
df = pd.read_csv('dataset/Electric_Production.csv',
                 parse_dates=['DATE'], index_col='DATE', )
print(df.head(2))
print(df.tail(2))

# %% 
# time feature
df['year'] = df.index.year
df['month'] = df.index.month
# log transformation
df['log'] = np.log(df.IPG2211A2N)
# percent change for two adjacents periods
df['pct_change'] = df.log.pct_change().mul(100)
# difference
df['diff'] = df.log.diff()
# drop null values
df = df.dropna()

#%%
print(df.head())

# %%
df[['IPG2211A2N', 'log', 'diff']].plot(subplots=True)
plt.show()

# %%
# applying the adfuller test
results = adfuller(df['diff'])
print(results)
print(f'The statistic test is: {np.round(results[0], 5)}')
print(f'p-value is: {results[1]}')
if results[1] > 0.05:
    print("Accept non-stationary")
else:
    print("Reject non-stationary")

# %%
# seasonal decomposition
decomp_results = seasonal_decompose(df.IPG2211A2N, freq=12)
# plot decomposed data
decomp_results.plot()
plt.show()

# %%
# provides train test indices
train = df.loc[df.year < 2017]
print(train.shape)
test = df.loc[df.year >= 2017]
print(test.shape)

# ARIMA

# %%
# ARIMA parameters search
results_arima = pm.auto_arima(
    train['diff'], d=0, start_p=1, start_1=1, max_p=3, max_q=3)
print(results_arima.summary())

#%%
# using the ARIMA model
model_arima = SARIMAX(train['diff'], order=(3, 0, 2)).fit()
# prediction
prediction_arima = model_arima.get_prediction(
    start=-50, dynamic=True).predicted_mean
# forecasting
forecast_arima = model_arima.get_forecast(steps=20).predicted_mean

#%%
# model diagnostics
arima_residual = model_arima.resid
arima_mae = np.mean(np.abs(arima_residual))
print(arima_mae)

#%%
# pymarima results
results_arima.plot_diagnostics()
plt.show()

# %%
# plot ARIMAX
plt.plot(df['diff'], label='True values')
plt.plot(prediction_arima, color='r', label='Prediction')
plt.plot(forecast_arima, color='pink', label='Forecast')
plt.legend()
plt.show()

# SARIMAX

# %%
# SARIMA parameters search
results_sarima = pm.auto_arima(
    train['diff'],
    seasonal=True,
    m=7,
    D=1,
    start_P=1,
    start_Q=1,
    information_criterion='aic',
    trace=True,
    error_action='ignore',
    stepwise=True)
print(results_sarima.summary())

#%%
# using the SARIMA model
model_sarima = SARIMAX(train['diff'], order=(5, 0, 1),
                       seasonal_order=(1, 1, [1, 2], 7)).fit()
# prediction
prediction_sarima = model_sarima.get_prediction(
    start=-50, dynamic=True).predicted_mean
# forecasting
forecast_sarima = model_sarima.get_forecast(steps=20).predicted_mean

#%%
# plot SARIMA diagnostics
results_sarima.plot_diagnostics()
plt.show()

# %%
# plot SARIMAX
plt.plot(df['diff'], label='True values')
plt.plot(prediction_sarima, color='r', label='Prediction')
plt.plot(forecast_sarima, color='pink', label='Forecast')
plt.legend()
plt.show()

#%%
# model diagnostics
sarima_residual = model_sarima.resid
sarima_mae = np.mean(np.abs(sarima_residual))
print(sarima_mae)

#%%
print(f'The Mean Absolute Error for ARIMA model is: {arima_mae}')
print(f'The Mean Absolute Error for SARIMA model is: {sarima_mae}')
