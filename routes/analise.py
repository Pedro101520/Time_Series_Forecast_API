from flask import jsonify, request, Blueprint
from models.pre_processing import tratamento_base
from utils.leitura import ler_arquivo

analitico_bp = Blueprint("analitico", __name__)

@analitico_bp.route("/analitico", methods=["POST"])
def upload_csv_analitico():
    pipeline = tratamento_base()

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
        df_tratado = pipeline.retorna()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400


    return jsonify({
        "message": "CSV tratado com sucesso",
        "Serie_Temporal_Tratada_Analitico": df_tratado.to_dict(orient="records")
    }), 200