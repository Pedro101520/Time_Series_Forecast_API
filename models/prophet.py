from .pre_processing import tratamento_base
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ProphetModel(tratamento_base):
    def __init__(self):
        self.rmse = None
        self.parametros = None
        self.data = None
        self.valor = None
        self.freq = None
        self.df = None
        self.pred = None
        self.mae = None
        self.rmse = None
        self.mape = None

    def padroniza_nome(self, treino, teste):
        self.treino = treino.rename(columns={"Data": "ds", "Valor_sem_outliers": "y"})
        self.teste  = teste.rename(columns={"Data": "ds", "Valor_sem_outliers": "y"})

    def avaliar(self, df):
        self.df = df.rename(columns={"Data": "ds", "Valor_sem_outliers": "y"})

        n_test = len(self.teste)

        cps_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
        sps_values = [0.1, 0.5, 1, 5, 10, 20, 40]

        melhor_rmse = float('inf')

        self.freq = self.frequencia(df)

        if (self.treino["y"].std() / self.treino["y"].mean()) > 0.15:
            sm = "multiplicative"
        else:
            sm = "additive"

        for cps in cps_values:
            for sps in sps_values:
                modelo = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=sm,
                    interval_width=0.70
                )

                modelo.fit(self.treino)

                future = modelo.make_future_dataframe(periods=n_test, freq=self.freq)
                forecast = modelo.predict(future)

                y_pred = forecast.tail(n_test)["yhat"]
                mae = mean_absolute_error(self.teste["y"], y_pred)
                rmse = np.sqrt(mean_squared_error(self.teste["y"], y_pred))
                mape = mean_absolute_percentage_error(self.teste["y"], y_pred)

                if rmse < melhor_rmse:
                    melhor_rmse = rmse
                    self.parametros = (cps, sps)

                    self.mae = mae
                    self.rmse = rmse
                    self.mape = mape

    def retorna_comparacao(self):
        return self.rmse, self.mape
    
    def retorna_metricas(self):
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape
        }
    
    def prever_futuro(self):

        if (self.treino["y"].std() / self.treino["y"].mean()) > 0.15:
            sm = "multiplicative"
        else:
            sm = "additive"

        modelo = Prophet(
            changepoint_prior_scale=self.parametros[0],
            seasonality_prior_scale=self.parametros[1],
            seasonality_mode=sm,
            interval_width=0.70
        )
        
        modelo.fit(self.df)

        qtde_pred = 0
        match self.freq:
            case 'D':
                qtde_pred = 30
            case 'B':
                qtde_pred = 30
            case 'MS' | 'M' | 'ME':
                qtde_pred = 24
            case 'W':
                qtde_pred = 40
            case 'YS' | 'YE' | 'Y' | 'A':
                qtde_pred = 10
            case _:
                qtde_pred = 30

        future = modelo.make_future_dataframe(periods=qtde_pred, freq=self.freq)
        self.pred = modelo.predict(future)

        result = self.pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result[["yhat", "yhat_lower", "yhat_upper"]] = result[["yhat", "yhat_lower", "yhat_upper"]].round(2)
        return result
        