import pandas as pd
import os
import matplotlib.pyplot as plt
from data import load_and_process_city_data

def load_city_temperature_data(data_path, cities):
    """Load and process temperature data for all cities."""
    all_city_dfs = [load_and_process_city_data(os.path.join(data_path, f"{city}-temp-rains.csv")) for city in cities]
    return all_city_dfs

def load_electricity_data(electricity_file_path, cities):
    """Load electricity data for all cities for the year 2018."""
    electricity_data_2018 = pd.read_csv(electricity_file_path)
    electricity_data_2018['Date'] = pd.to_datetime(electricity_data_2018['Date'])
    electricity_data_2018.set_index('Date', inplace=True)
    return electricity_data_2018[cities]

def compare_electricity_with_temperature(temperature_data_2018, electricity_data_2018, cities):
    """Compare electricity usage with mean temperature to determine HDD and CDD reference temperatures."""
    hdd_reference_temps = []
    cdd_reference_temps = []

    for city in cities:
        temp_mean_2018 = temperature_data_2018[city].mean()
        electricity_usage = electricity_data_2018[city].mean()
        if electricity_usage > temp_mean_2018:
            cdd_reference_temps.append(temp_mean_2018)
            hdd_reference_temps.append(None)
        else:
            hdd_reference_temps.append(temp_mean_2018)
            cdd_reference_temps.append(None)
    
    return hdd_reference_temps, cdd_reference_temps

def plot_temperature_and_electricity(temperature_data_2018, electricity_data_2018, hdd_reference_temps, cdd_reference_temps, cities):
    """Plot temperature data and electricity usage for each city."""
    fig, axs = plt.subplots(len(cities), 1, figsize=(10, 20), sharex=True)

    for i, city in enumerate(cities):
        axs[i].plot(temperature_data_2018[city].index, temperature_data_2018[city], label='Temperature')
        axs[i].set_ylabel('Temperature (°C)')
        axs[i].legend(loc='upper left')

        axs2 = axs[i].twinx()
        axs2.plot(electricity_data_2018.index, electricity_data_2018[city], color='orange', label='Electricity Usage')
        axs2.set_ylabel('Electricity Usage')
        axs2.legend(loc='upper right')

        if hdd_reference_temps[i] is not None:
            axs[i].axhline(y=hdd_reference_temps[i], color='blue', linestyle='--', label='HDD Reference Temp')
        if cdd_reference_temps[i] is not None:
            axs[i].axhline(y=cdd_reference_temps[i], color='red', linestyle='--', label='CDD Reference Temp')

        axs[i].set_title(city)

    plt.tight_layout()
    plt.xlabel('Date')
    plt.savefig('cities_temperature_electricity.png')

def calculate_rolling_correlations(electricity_data_2018, temperature_data_2018, cities):
    """Calculate rolling correlations between electricity usage and temperature."""
    rolling_correlations = {}
    for city in cities:
        rolling_correlations[city] = electricity_data_2018[city].rolling(window=12).corr(temperature_data_2018[city])
    return rolling_correlations

def identify_correlation_peaks(rolling_correlations):
    """Identify weeks with the highest positive and negative correlations."""
    positive_peaks = {}
    negative_peaks = {}
    for city, correlation in rolling_correlations.items():
        positive_peaks[city] = correlation[correlation > 0].idxmax()
        negative_peaks[city] = correlation[correlation < 0].idxmin()
    return positive_peaks, negative_peaks

def get_tref_from_correlation(city, correlation_data, temperature_data):
    """Identifies temperature at a correlation peak in electricity/temperature data for the entire year."""
    peak_index = correlation_data.idxmax()
    peak_temperature = temperature_data.loc[peak_index]
    return peak_temperature

def calculate_reference_temperatures(rolling_correlations, temperature_data_2018, cities):
    """Calculate HDD and CDD Tref for the entire year."""
    hdd_tref_yearly_diff = {}
    cdd_tref_yearly_diff = {}
    for city in cities:
        hdd_correlation_data = rolling_correlations[city][rolling_correlations[city] < 0]
        hdd_tref_yearly_diff[city] = get_tref_from_correlation(city, hdd_correlation_data, temperature_data_2018[city])

        cdd_correlation_data = rolling_correlations[city][rolling_correlations[city] > 0]
        cdd_tref_yearly_diff[city] = get_tref_from_correlation(city, cdd_correlation_data, temperature_data_2018[city])

    return hdd_tref_yearly_diff, cdd_tref_yearly_diff


