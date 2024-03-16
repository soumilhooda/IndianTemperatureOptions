import numpy as np
import pandas as pd
import datetime as dt

# Define Indian winter and summer months
WINTER_MONTHS = [1, 2, 10, 11, 12]
MONSOON_MONTHS = [3, 4, 5, 6, 7, 8, 9]

def T_model(x, a, b, c, d, alpha, beta, theta):
    omega = 2 * np.pi / 365.25
    return a + b*x + c*x**2 + d*x**3 + alpha*np.sin(omega*x + theta) + beta*np.cos(omega*x + theta)

# Define derivative of T model equation
def dT_model(x, a, b, c, d, alpha, beta, theta):
    omega = 2 * np.pi / 365.25
    dT = b + 2*c*x + 3*d*x**2 + alpha*omega*np.cos(omega*x + theta) - beta*omega*np.sin(omega*x + theta)
    return dT

def euler_step(row, kappa, M, lamda):
    """Function for euler scheme approximation step in
    modified OH dynamics for temperature simulations
    Inputs:
    - dataframe row with columns: T, Tbar, dTbar and vol
    - kappa: rate of mean reversion
    Output:
    - temp: simulated next day temperatures
    """
    if not np.isnan(row['T']):
        T_i = row['Tbar']
    else:
        T_i = row['T']
    T_det = T_i + row['dTbar']
    T_mrev =  kappa*(row['Tbar'] - T_i)
    sigma = row['vol']*np.random.randn(M)
    riskn = lamda*row['vol']
    return T_det + T_mrev + sigma - riskn

def monte_carlo_temp(trading_dates, Tbar_params, vol_model, first_ord, kappa, city, no_sims, lamda=0):
    """Monte Carlo simulation of temperature
    Inputs:
    - trading_dates: pandas DatetimeIndex from start to end dates
    - M: number of simulations
    - Tbar_params: parameters used for Tbar model
    - vol_model: fitted volatility model with days in year index
    - first_ord: first ordinal of fitted Tbar model
    Outputs:
    - mc_temps: DataFrame of all components and simulated temperatures
    """
    if isinstance(trading_dates, pd.DatetimeIndex):
        trading_date = trading_dates.map(dt.datetime.toordinal)

    Tbar_params = Tbar_params[city]
    vol_model = vol_model[city]
    first_ord = first_ord[city]

    Tbars = T_model(trading_date-first_ord, *Tbar_params)
    dTbars = dT_model(trading_date-first_ord, *Tbar_params)
    mc_temps = pd.DataFrame(data=np.array([Tbars, dTbars]).T,
                            index=trading_dates, columns=['Tbar','dTbar'])
    mc_temps['day'] = mc_temps.index.dayofyear
    mc_temps['vol'] = vol_model[mc_temps['day']-1]

    mc_temps['T'] = mc_temps['Tbar'].shift(1)
    data = mc_temps.apply(euler_step, args=[kappa, no_sims, lamda], axis=1)
    mc_sims = pd.DataFrame(data=[x for x in [y for y in data.values]],
                 index=trading_dates,columns=range(1,no_sims+1))
    
    return mc_temps, mc_sims

def determine_option_type(start_date, end_date):
    """Determines the option type based on contract dates."""
    start_month = dt.datetime.strptime(start_date, "%Y-%m-%d").month
    end_month = dt.datetime.strptime(end_date, "%Y-%m-%d").month

    if start_month in WINTER_MONTHS and end_month in WINTER_MONTHS:
        return 'winter'
    elif start_month in MONSOON_MONTHS and end_month in MONSOON_MONTHS:
        return 'monsoon'
    else:
        return None
