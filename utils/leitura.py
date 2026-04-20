import pandas as pd

def ler_arquivo(arquivo):
    configs = [
        {"sep": ","},
        {"sep": ";"},
        {"sep": ";", "decimal": ","},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": None, "engine": "python"},
    ]

    for config in configs:
        try:
            arquivo.seek(0)
            df = pd.read_csv(arquivo, on_bad_lines="skip", **config)
            if len(df.columns) >= 2:
                return df
        except Exception as e:
            continue

    raise ValueError("Não foi possível ler o CSV com os formatos suportados")