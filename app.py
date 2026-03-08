from flask import Flask, make_response, jsonify, request
import pandas as pd
from models.pre_processing import tratamento_base
from models.prophet import ProphetModel
from models.sarima import SarimaModel
from models.holt_winters import Holt_Winters_Model

app = Flask(__name__) 

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

@app.route("/pipeline/predicao", methods=["POST"])
def upload_csv():
    pipeline = tratamento_base()
    prophet = ProphetModel()
    sarima = SarimaModel()
    holt_winters = Holt_Winters_Model()


    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Arquivo não é CSV"}), 400
    

    df = ler_arquivo(file)
    
    try:
        pipeline.carregar_base(df)
        pipeline.validar_serie()
        pipeline.padroniza_nome()
        pipeline.tratamento_nulo()
        pipeline.tratamento_outliers()
        treino, teste = pipeline.treino_teste()
        df_tratado = pipeline.retorna()

        print(df_tratado)

        prophet.padroniza_nome(treino, teste)
        prophet.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        sarima.avaliar(df_tratado[["Data", "Valor_sem_outliers"]], treino, teste)

        holt_winters.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        rmse_compara = []
        rmse_compara.append(prophet.retorna_comparacao())
        rmse_compara.append(sarima.retorna_comparacao())
        rmse_compara.append(holt_winters.retorna_comparacao())

        melhor_rmse = prophet.retorna_comparacao()
        for i in rmse_compara:
            if i < melhor_rmse:
                melhor_rmse = i
        
        melhor_modelo = rmse_compara.index(min(rmse_compara))

        modelo = ""
        metricas = None
        forecast = None
        match melhor_modelo:
            case 0:
                prophet.prever_futuro()
                metricas = prophet.retorna_metricas()
                modelo = "Prophet"
                forecast = prophet.prever_futuro()
            case 1:
                sarima.prever_futuro()
                metricas = sarima.retorna_metricas()
                modelo = "SARIMA"
                forecast = sarima.prever_futuro()
            case 2:
                holt_winters.prever_futuro()
                metricas = holt_winters.retorna_metricas()
                modelo = "Holt-Winters"
                forecast = holt_winters.prever_futuro()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400


    return jsonify({
        "message": "CSV tratado com sucesso",
        "Melhor Modelo": f"{modelo}",
        "Metricas": f"{metricas}",
        "Serie_Temporal_Tratada": df_tratado.to_dict(orient="records"),
        "Forecast": forecast.to_dict(orient="records")
    }), 200


# if __name__ == "__main__":
#     app.run()
app.run()