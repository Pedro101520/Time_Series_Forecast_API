from .pre_processing import tratamento_base
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

class Holt_Winters_Model(tratamento_base):
    def __init__(self):
        self.df = None
        self.freq = None
        self.treino = None
        self.teste = None
        self.results = None
        self.melhor_modelo = None
        self.mae = None
        self.mape = None
        self.rmse = None
    
    def treino_teste(self, df):
        df.reset_index(inplace=True)
        df_avaliar = df[["Data", "Valor_sem_outliers"]]

        treino, teste = train_test_split(df_avaliar, test_size=0.2, shuffle=False, random_state=42)
        treino.set_index("Data", inplace=True)
        teste.set_index("Data", inplace=True)

        return treino, teste
        

    def avaliar(self, df):
        self.df = df

        self.df["Data"] = pd.to_datetime(self.df["Data"])
        self.df = self.df.set_index("Data")

        freq = pd.infer_freq(self.df.index)
        self.freq = freq.split('-')[0]
        self.df = self.df.asfreq(freq)

        self.treino, self.teste = self.treino_teste(self.df)


        if not(df["Valor_sem_outliers"] < 0).any():
            configs = [
                {'trend': 'add', 'seasonal': 'add', 'damped': False},
                {'trend': 'add', 'seasonal': 'mul', 'damped': False},
                {'trend': 'mul', 'seasonal': 'add', 'damped': False},
                {'trend': 'mul', 'seasonal': 'mul', 'damped': False},
                {'trend': 'add', 'seasonal': 'add', 'damped': True},
                {'trend': 'add', 'seasonal': 'mul', 'damped': True},
            ]
        else:
            configs = [
                {'trend': 'add', 'seasonal': 'add', 'damped': False},
                {'trend': 'add', 'seasonal': 'add', 'damped': True},
            ]

        self.results = []

        for cfg in configs:
            try:
                model = ExponentialSmoothing(
                self.treino['Valor_sem_outliers'],
                trend=cfg['trend'],
                seasonal=cfg['seasonal'],
                damped_trend=cfg['damped'],
                seasonal_periods=7
                ).fit(optimized=True)

                forecast = model.forecast(len(self.teste))
                rmse = np.sqrt(mean_squared_error(self.teste['Valor_sem_outliers'], forecast))
                mae = mean_absolute_error(self.teste['Valor_sem_outliers'], forecast)
                mape = np.mean(np.abs((self.teste['Valor_sem_outliers'] - forecast) / self.teste['Valor_sem_outliers'])) * 100

                self.results.append({
                'trend': cfg['trend'],
                'seasonal': cfg['seasonal'],
                'damped': cfg['damped'],
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'model': model
            })

            except Exception as e:
                raise("Erro:", cfg, e)
        
        self.melhor_modelo = min(self.results, key=lambda x: x['rmse'])
        self.rmse = self.melhor_modelo["rmse"]
        self.mae = self.melhor_modelo["mae"]
        self.mape = self.melhor_modelo["mape"]
    
    def retorna_comparacao(self):
        return self.rmse
    
    def retorna_metricas(self):
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape
        }

    def prever_futuro(self):
        self.df["Data"] = pd.to_datetime(self.df["Data"])
        self.df = self.df.set_index("Data")
        self.df = self.df.sort_index()

        freq = pd.infer_freq(self.df.index)

        self.df = self.df.asfreq(freq)

        qtde_pred = 0
        seasonal = None
        match self.freq:
            case 'D':
                qtde_pred = 90
                seasonal = 7
            case 'B':
                qtde_pred = 60
                seasonal = 5
            case 'MS' | 'M' | 'ME':
                qtde_pred = 24
                seasonal = 12
            case 'W':
                qtde_pred = 52
                seasonal = 4
            case 'YS' | 'YE' | 'Y' | 'A':
                qtde_pred = 10
                seasonal = None
            case _:
                qtde_pred = 30

        if not(seasonal == None):
            model = ExponentialSmoothing(
                self.df['Valor_sem_outliers'],
                trend=self.melhor_modelo['trend'],
                seasonal=self.melhor_modelo['seasonal'],
                damped_trend=self.melhor_modelo['damped'],
                seasonal_periods=seasonal
            ).fit(optimized=True)
        else: 
            model = ExponentialSmoothing(
                self.df['Valor_sem_outliers'],
                trend=self.melhor_modelo['trend'],
                damped_trend=self.melhor_modelo['damped'],
            ).fit(optimized=True)

        forecast = model.forecast(qtde_pred)
        forecast_df = forecast.reset_index()
        forecast_df.columns = ["Data", "Valor"]
        return forecast_df