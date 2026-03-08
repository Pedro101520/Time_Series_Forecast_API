import pandas as pd
from .pre_processing import tratamento_base
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

class SarimaModel(tratamento_base):
    def __init__(self):
         self.df = None
         self.freq = None
         self.treino = None
         self.teste = None
         self.auto_model = None
         self.mae = None
         self.rmse = None
         self.mape = None
    
    def avaliar(self, df, treino, teste):
        self.df = df
        self.freq = self.frequencia(df)

        self.treino = treino
        self.teste = teste

        self.treino.set_index('Data', inplace=True)
        self.teste.set_index('Data', inplace=True)

        auto_model = auto_arima(
            df,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            m=7,                  
            seasonal=True,
            d=1,                   
            D=1,                  
            max_P=1, max_Q=1,
            test='adf',             
            stepwise=True,
            trace=True,
            n_fits=20,              
            error_action='ignore',
            suppress_warnings=True
        )
                
        self.auto_model = auto_model

        model = SARIMAX(self.treino, order=self.auto_model.order, seasonal_order=self.auto_model.seasonal_order)
        model_fit = model.fit()

        forecast_test = model_fit.forecast(len(self.teste))

        self.mae = mean_absolute_error(self.teste, forecast_test)
        self.mape = mean_absolute_percentage_error(self.teste, forecast_test)
        self.rmse = np.sqrt(mean_squared_error(self.teste, forecast_test))

    
    def retorna_comparacao(self):
        return self.rmse
    
    def retorna_metricas(self):
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape
        }

    
    def prever_futuro(self):
        self.df.index.freq = self.freq
        model = SARIMAX(self.df, order=self.auto_model.order, seasonal_order=self.auto_model.seasonal_order)
        model_fit = model.fit() 

        match self.freq:
            case 'D':
                qtde_pred = 90
            case 'B':
                qtde_pred = 60
            case 'MS':
                qtde_pred = 24
            case 'W':
                qtde_pred = 52
            case _:
                qtde_pred = 30

        forecast_result  = model_fit.get_forecast(qtde_pred)
        forecast_mean = forecast_result.predicted_mean.round(2)
        conf_int = forecast_result.conf_int(alpha=0.05) 
        
        df_forecast = pd.DataFrame({
            'forecast':         forecast_mean,
            'limite_inferior':  conf_int.iloc[:, 0].round(2),
            'limite_superior':  conf_int.iloc[:, 1].round(2),

        })
        
        return df_forecast
