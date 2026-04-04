# 📊 Forecast API

Esta API tem como objetivo fornecer uma forma simples e automatizada de aplicar modelos estatísticos para previsão de séries temporais univariadas.

Atualmente, estão disponíveis três modelos de previsão:

- SARIMA
- Holt-Winters
- Prophet

Ao utilizar o endpoint principal, a API realiza automaticamente:

- Tratamento da série temporal
- Verificação automática entre modelos
- Seleção do melhor modelo com base em métricas de desempenho

A métrica com maior peso na seleção do modelo é o RMSE (Root Mean Squared Error).

---

🌐 Disponibilidade da API

A API está hospedada no Google Cloud Platform (GCP), utilizando o serviço Cloud Run para execução e escalabilidade.

Atualmente, o acesso público à API está desativado por decisão de projeto, com o objetivo de controle de uso e recursos.

No entanto, caso haja interesse em utilizar a API, é possível solicitar acesso entrando em contato comigo.
As informações de contato estão disponíveis no meu perfil do GitHub.

---

# 🛠️ Tecnologias utilizadas

- Python
- Flask
- Pandas
- Numpy
- SARIMA (statsmodels)
- Holt-Winters
- Prophet
- Google Cloud
- Docker

---

# 📄 Endpoints

A API possui 5 endpoints:

### 1️⃣ `/pipeline/predicao`
Endpoint principal da API.

Responsável por:

- Realizar o tratamento da série temporal
- Treinar os modelos
- Selecionar automaticamente o melhor modelo
- Retornar a previsão

Exemplo de saída:

```json
{
  "message": "Modelo treinado com sucesso",
  "Melhor Modelo": "Prophet",
  "Metricas": {
    "RMSE": 123.45,
    "MAE": 67.89,
    "MAPE": 4.56
  },
  "Forecast": [
    {"Data": "2026-03-14", "Valor": 150.32},
    {"Data": "2026-03-15", "Valor": 152.10},
    {"Data": "2026-03-16", "Valor": 148.75}
  ]
}
```

---

### 2️⃣ `/tratamento`

Realiza apenas o tratamento da série temporal, sem aplicar modelos de previsão.

Exemplo de saída:

```json
{
  "message": "CSV tratado com sucesso",
  "Serie_Temporal_Tratada": [
    {"Data": "2026-01-01", "Valor": 120.5},
    {"Data": "2026-01-02", "Valor": 123.0},
    {"Data": "2026-01-03", "Valor": 119.7},
    {"Data": "2026-01-04", "Valor": 121.2}
  ]
}
```
---

A rota `/analitico` extende da rota acima, com a diferença que retorna o histórico da série temporal sem o tratamento dos outliers incluso.

Exemplo de saída:
```json
{
  "message": "CSV tratado com sucesso",
  "Serie_Temporal_Tratada_Analitico": [
    {"Data": "2026-01-01", "Valor": 120.5},
    {"Data": "2026-01-02", "Valor": 123.0},
    {"Data": "2026-01-03", "Valor": 119.7},
    {"Data": "2026-01-04", "Valor": 121.2}
  ]
}
```

---

### 3️⃣ `/sarima`

Realiza o forecast utilizando apenas o modelo SARIMA.

---

### 4️⃣ `/prophet`

Realiza o forecast utilizando apenas o modelo Prophet.

---

### 5️⃣ `/holt_winters`

Realiza o forecast utilizando apenas o modelo Holt-Winters.

---

# O exemplo de saída abaixo serve para os endpoints individuais (SARIMA, Prophet e Holt-Winters)
```json
{
  "message": "Modelo treinado com sucesso",
  "Modelo": "Prophet",
  "Metricas": {
    "RMSE": 123.45,
    "MAE": 67.89,
    "MAPE": 4.56
  },
  "Forecast": [
    {"Data": "2026-03-14", "Valor": 150.32},
    {"Data": "2026-03-15", "Valor": 152.10},
    {"Data": "2026-03-16", "Valor": 148.75}
  ]
}
```
---

# ⚙️ Limitações da API

Para garantir desempenho e controle de recursos, algumas restrições foram implementadas:

- Máximo de 200.000 linhas por série temporal
- Séries com frequência horária ou menor (minutos, segundos) são agrupadas para frequência diária utilizando `resample()` do pandas
- A quantidade de períodos previstos é pré-definida de acordo com a frequência da série

---

# 📂 Formato de entrada

A API aceita apenas arquivos `.csv`.

Exemplo de envio usando Python:

```python
import requests

files = {"file": open("serie_temporal.csv", "rb")}

headers = {
    "x-api-key": "API_KEY"
}

response = requests.post(
    "http://localhost:5000/pipeline/predicao",
    files=files,
    headers=headers
)

print(response.json())
