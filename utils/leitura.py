import pandas as pd

def ler_arquivo(arquivo):
    configs = [
        {"sep": ",", "on_bad_lines": "skip"},
        {"sep": ";", "on_bad_lines": "skip"},
        {"sep": ";", "decimal": ",", "on_bad_lines": "skip"},
        {"sep": ",", "encoding": "latin1", "on_bad_lines": "skip"},
        {"sep": ";", "encoding": "latin1", "on_bad_lines": "skip"},
        {"sep": ";", "decimal": ",", "encoding": "latin1", "on_bad_lines": "skip"},
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
    ]

    for config in configs:
        try:
            return pd.read_csv(arquivo, **config)
        except Exception:
            continue

    raise ValueError("Não foi possível ler o CSV com os formatos suportados")