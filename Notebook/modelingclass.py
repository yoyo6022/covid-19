
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd
from pmdarima.arima import auto_arima



class modeling:
    warnings.filterwarnings("ignore")

    def __init__(self, data, storeid):
        self.data = data
        self.storeid = storeid

    def forecast(self, start_date, end_date):
        storeweeklysales = self.data.loc[self.data.Store == self.storeid].set_index('Date').resample('w').Sales.sum()
        # storeweeklysales = storeweeklysales[storeweeklysales!=0]
        forecastindex = pd.date_range(start_date, end_date, freq='W')
        if len(storeweeklysales[(storeweeklysales == 0)].index) == 0:
            train, test = (storeweeklysales[:-9], storeweeklysales[-9:-1])
            arima_model = auto_arima(storeweeklysales[:test.index[-1]], start_p=0, d=0, start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
            forecastresult1 = pd.DataFrame([forecast, CI[:, 0], CI[:, 1]],
                                           index=['Forecast', 'lower CI', 'upper CI']).T.set_index(forecastindex)
            return forecastresult1

        else:
            train_startdate = storeweeklysales.loc[storeweeklysales[(storeweeklysales == 0)].index[-1]:].index[1]
            if len(storeweeklysales.loc[train_startdate:]) >= 11:
                train_enddate = len(storeweeklysales.loc[train_startdate:]) - 9
                train = storeweeklysales.loc[train_startdate:][:train_enddate]
                test = storeweeklysales.loc[train_startdate:][train_enddate:-1]

                arima_model = auto_arima(storeweeklysales.loc[train_startdate:test.index[-1]], start_p=0, d=0,
                                         start_q=0,
                                         max_p=5, max_d=5, max_q=5, start_P=0,
                                         D=1, start_Q=0, max_P=5, max_D=5,
                                         max_Q=5, m=2, seasonal=True,
                                         error_action='warn', trace=False,
                                         supress_warnings=True, stepwise=False,
                                         random=True,
                                         random_state=20, n_fits=50)

                forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
                return pd.DataFrame([forecast, CI[:, 0], CI[:, 1]],
                                    index=['Forecast', 'lower CI', 'upper CI']).T.set_index(forecastindex)


            else:
                storeweeklysales = storeweeklysales[storeweeklysales != 0]
                train, test = (storeweeklysales[:-9], storeweeklysales[-9:-1])
                arima_model = auto_arima(storeweeklysales[:test.index[-1]], start_p=0, d=0, start_q=0,
                                         max_p=5, max_d=5, max_q=5, start_P=0,
                                         D=1, start_Q=0, max_P=5, max_D=5,
                                         max_Q=5, m=2, seasonal=True,
                                         error_action='warn', trace=False,
                                         supress_warnings=True, stepwise=False,
                                         random=True,
                                         random_state=20, n_fits=50)

                forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
                forecastresult1 = pd.DataFrame([forecast, CI[:, 0], CI[:, 1]],
                                               index=['Forecast', 'lower CI', 'upper CI']).T.set_index(forecastindex)
                return forecastresult1
