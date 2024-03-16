import numpy as np
from scipy import interpolate

def calculate_volatility(all_city_dfs):
    # Initialize an empty dictionary to store volatility for each city
    volatility = {}

    # Iterate over all city dataframes
    for city_data in all_city_dfs:
        # Assuming temp_t is the dataframe with temperature data for each city
        temp_vol = city_data['Temp Mean'].copy(deep=True)
        temp_vol = temp_vol.to_frame()
        temp_vol['day'] = temp_vol.index.dayofyear

        vol = temp_vol.groupby(['day'])['Temp Mean'].agg(['mean', 'std'])
        days = np.array(vol['std'].index)
        T_std = np.array(vol['std'].values)

        def spline(knots, x, y):
            x_new = np.linspace(0, 1, knots+2)[1:-1]
            t, c, k = interpolate.splrep(x, y, t=np.quantile(x, x_new), s=3)
            yfit = interpolate.BSpline(t,c, k)(x)
            return yfit

        volatility[city_data['City'].iloc[0]] = spline(15, days, T_std)

    return volatility

