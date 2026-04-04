from flask import Flask, request, abort, jsonify
from routes.prophet import prophet_bp
from routes.pipeline_completa import pipeline_bp
from routes.sarima import sarima_bp
from routes.holt_winters import holt_winters_bp
from routes.tratar_base import tratamento_bp
from routes.analise import analitico_bp
from dotenv import load_dotenv
import os

app = Flask(__name__) 

load_dotenv()

API_KEY = os.getenv("API_KEY")

@app.before_request
def verificar_api_key():
    chave = request.headers.get("x-api-key")

    if chave != API_KEY:
        abort(401)


@app.errorhandler(401)
def unauthorized(e):
    return jsonify({
        "erro": "Acesso não autorizado",
        "mensagem": "API Key inválida ou ausente"
    }), 401

app.register_blueprint(prophet_bp)
app.register_blueprint(pipeline_bp)
app.register_blueprint(sarima_bp)
app.register_blueprint(holt_winters_bp)
app.register_blueprint(tratamento_bp)
app.register_blueprint(analitico_bp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)