import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import AutoReg
from scipy import interpolate

# Define the model function
def model_func(t, a, b, c, d, alpha, beta, theta):
    omega = 2 * np.pi / 365.25
    return a + b*t + c*t**2 + d*t**3 + alpha*np.sin(omega*t + theta) + beta*np.cos(omega*t + theta)

def T_model(x, a, b, c, d, alpha, beta, theta):
    omega = 2 * np.pi / 365.25
    return a + b*x + c*x**2 + d*x**3 + alpha*np.sin(omega*x + theta) + beta*np.cos(omega*x + theta)

# Define derivative of T model equation
def dT_model(x, a, b, c, d, alpha, beta, theta):
    omega = 2 * np.pi / 365.25
    dT = b + 2*c*x + 3*d*x**2 + alpha*omega*np.cos(omega*x + theta) - beta*omega*np.sin(omega*x + theta)
    return dT

# Define the error function
def error_func(t, temp, params):
    return temp - model_func(t, *params)

# Function to fit model to city data and visualize the fitting process for the last year
def fit_and_visualize(city_data):
    # Calculate time dimension for the entire dataset
    t = (city_data.index - city_data.index[0]).days.values
    temp = city_data['Temp Mean'].values

    # Split data into train and test sets
    train_data, test_data = train_test_split(city_data, test_size=0.2, shuffle=False, random_state=42)
    t_train = (train_data.index - train_data.index[0]).days.values
    temp_train = train_data['Temp Mean'].values
    t_test = (test_data.index - test_data.index[0]).days.values
    temp_test = test_data['Temp Mean'].values

    # Fit the model to the train data
    initial_guess = [0, 0, 0, 0, 0, 0, 0]  # Initial guess for parameters
    params, _ = curve_fit(model_func, t_train, temp_train, p0=initial_guess)

    # Fit an AutoRegressive model to residuals
    residuals_all = temp - model_func(t, *params)
    residuals_all_df = pd.DataFrame(data=residuals_all, index=city_data.index)
    residuals_all_df.index = pd.DatetimeIndex(residuals_all_df.index).to_period('D')
    model = AutoReg(residuals_all_df.squeeze(), lags=1, old_names=True, trend='n')
    model_fit = model.fit()
    
    # Calculate kappa
    gamma = model_fit.params[0]
    kappa = 1 - gamma

    # Store parameters for T model equation
    Tbar_params = params

    temp_t = city_data.copy()  
    if isinstance(temp_t.index, pd.DatetimeIndex):
        first_ord = temp_t.index.map(dt.datetime.toordinal)[0]
        temp_t.index = temp_t.index.map(dt.datetime.toordinal)

    temp_t['model_fit'] = T_model(temp_t.index - first_ord, *Tbar_params)

    if not isinstance(temp_t.index, pd.DatetimeIndex):
        temp_t.index = temp_t.index.map(dt.datetime.fromordinal)

    return T_model, dT_model, Tbar_params, kappa

