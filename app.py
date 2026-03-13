from flask import Flask
from routes.prophet import prophet_bp
from routes.pipeline_completa import pipeline_bp
from routes.sarima import sarima_bp
from routes.holt_winters import holt_winters_bp
from routes.tratar_base import tratamento_bp

app = Flask(__name__) 

app.register_blueprint(prophet_bp)
app.register_blueprint(pipeline_bp)
app.register_blueprint(sarima_bp)
app.register_blueprint(holt_winters_bp)
app.register_blueprint(tratamento_bp)


# if __name__ == "__main__":
#     app.run()
app.run()