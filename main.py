import data, tref, tempmodel, tempvolatility, tempsimulation
import pandas as pd
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# Set path to your csv files and list of cities 
data_path = '/Users/soumilhooda/Desktop/WeatherDerivatives/V1/India/data'
cities = ['kolkata', 'amd', 'delhi', 'hyd', 'pune', 'bengaluru', 'mumbai', 'chennai']
electricity_file_path = '/Users/soumilhooda/Desktop/WeatherDerivatives/V1/India/electricity.csv'

# Load and process data for all cities
all_city_dfs = [data.load_and_process_city_data(os.path.join(data_path, f"{city}-temp-rains.csv")) for city in cities]

# Load and process temperature data
temperature_data_2018 = tref.load_city_temperature_data(data_path, cities)

# Load electricity data
electricity_data_2018 = tref.load_electricity_data(electricity_file_path, cities)

# Compare electricity usage with temperature to determine HDD and CDD reference temperatures
hdd_reference_temps, cdd_reference_temps = tref.compare_electricity_with_temperature(temperature_data_2018, electricity_data_2018, cities)

# Plot temperature data and electricity usage
tref.plot_temperature_and_electricity(temperature_data_2018, electricity_data_2018, hdd_reference_temps, cdd_reference_temps, cities)

# Calculate rolling correlations
rolling_correlations = tref.calculate_rolling_correlations(electricity_data_2018, temperature_data_2018, cities)

# Identify correlation peaks
positive_peaks, negative_peaks = tref.identify_correlation_peaks(rolling_correlations)

# Calculate HDD and CDD Tref for the entire year
hdd_tref_yearly_diff, cdd_tref_yearly_diff = tref.calculate_reference_temperatures(rolling_correlations, temperature_data_2018, cities)

T_models = {}
dT_models = {}
Tbar_params_list = {}
kappas = {}
for city_data in all_city_dfs:
    T_model_func, dT_model_func, Tbar_params, kappa = tempmodel.fit_and_visualize(city_data)
    city_name = city_data["City"].iloc[0]
    T_models[city_name] = T_model_func
    dT_models[city_name] = dT_model_func
    Tbar_params_list[city_name] = Tbar_params
    kappas[city_name] = kappa

volatility = tempvolatility.calculate_volatility(all_city_dfs)

data_start_date = dt.datetime(1951, 1, 1)

# Define first ordinal values for each city
first_ord = {
    'kolkata': data_start_date.toordinal(),
    'amd': data_start_date.toordinal(),
    'delhi': data_start_date.toordinal(),
    'hyd': data_start_date.toordinal(),
    'pune': data_start_date.toordinal(),
    'bengaluru': data_start_date.toordinal(),
    'mumbai': data_start_date.toordinal(),
    'chennai': data_start_date.toordinal()
}

no_sims=1000

def temperature_option(trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, first_ord, option_type, opt='c', no_sims=1000, lamda=0):
    "Evaluates the price of a temperature call option"
    for city in Tbar_params_list.keys():
        # print(city)
        mc_temps, mc_sims = tempsimulation.monte_carlo_temp(trading_dates, Tbar_params_list, volatility, first_ord, kappas, city, no_sims=1000, lamda=0)
        N, M = np.shape(mc_sims)
        mc_arr = mc_sims.values
        # print('this is mc_arr value inside temp option after exiting monte carlo: ', mc_arr)
        if option_type == 'winter':
            DD = np.sum(np.maximum(hdd_tref_yearly_diff[city]-mc_arr,0), axis=0)
        elif option_type == 'monsoon':
            DD = np.sum(np.maximum(mc_arr-hdd_tref_yearly_diff[city],0), axis=0)
        else:
            print('Options cannot run across or outside a season.')
            exit
        if opt == 'c':
            CT = alpha*np.maximum(DD-K,0)
        else:
            CT = alpha*np.maximum(K-DD,0)
        C0 = np.exp(-r*tau)*np.sum(CT)/M
        sigma = np.sqrt( np.sum( (np.exp(-r*tau)*CT - C0)**2) / (M-1) )
        SE = sigma/np.sqrt(M)
        # print(f'{opt.upper()} Price for {city.capitalize()} City: {np.round(C0, 2)} +/- {np.round(SE*2, 2)} (2SE)')
        return C0

def years_between(d1, d2):
    d1 = dt.datetime.strptime(d1, "%Y-%m-%d")
    d2 = dt.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)/365.25

# Function to plot call and put option prices against premium
def plot_option_prices(start_date, end_date, option_type):
    trading_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    tau = years_between(start_date, end_date)
    option_type = tempsimulation.determine_option_type(start_date, end_date)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'{option_type.capitalize()} Months Option Prices')

    # Calculate option prices for the first city
    first_city = cities[0]
    call_prices_first = []
    put_prices_first = []
    for K in strike_prices:
        C0_first = temperature_option(trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, first_ord, option_type, opt='c', no_sims=1000, lamda=0)
        P0_first = temperature_option(trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, first_ord, option_type, opt='p', no_sims=1000, lamda=0)

        call_prices_first.append(C0_first)
        put_prices_first.append(P0_first)

    # Plot option prices for the first city
    ax.plot(strike_prices, call_prices_first, label=f'{first_city} - Call Option', color='blue')
    ax.plot(strike_prices, put_prices_first, label=f'{first_city} - Put Option', color='orange')

    # Plot variance around the first city's plot for other cities
    for city in cities[1:]:
        call_prices = []
        put_prices = []
        for K in strike_prices:
            C0 = temperature_option(trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, first_ord, option_type, opt='c', no_sims=1000, lamda=0)
            P0 = temperature_option(trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, first_ord, option_type, opt='p', no_sims=1000, lamda=0)

            call_prices.append(C0)
            put_prices.append(P0)

        # Plot variance with increased transparency
        ax.fill_between(strike_prices, call_prices, call_prices_first, label=f'{city} - Call Option Variance', alpha=0.5)
        ax.fill_between(strike_prices, put_prices, put_prices_first, label=f'{city} - Put Option Variance', alpha=0.5)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Option Premium (USD)')  # Update ylabel to include currency
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot
    ax.set_xticks(np.arange(min(strike_prices), max(strike_prices)+1, 5))  # Add ticks at intervals of 5 on x-axis
    ax.set_yticks(np.arange(0, max(call_prices_first + put_prices_first) + 100, 100))  # Add more ticks on y-axis
    plt.tight_layout()
    plt.show()

# Define strike prices
strike_prices = np.arange(975, 1025, 5)
r = 0.05
alpha = 25

# Define dates within winter and summer months
monsoon_start = '2021-03-01'
monsoon_end = '2021-09-20'

# Plot for winter months
plot_option_prices(monsoon_start, monsoon_end, 'monsoon')
