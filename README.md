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

# 🛠️ Tecnologias utilizadas

- Python
- Flask
- Pandas
- Numpy
- SARIMA (statsmodels)
- Holt-Winters
- Prophet

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

---

### 2️⃣ `/tratamento`

Realiza apenas o tratamento da série temporal, sem aplicar modelos de previsão.

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

response = requests.post(
    "http://localhost:5000/pipeline/predicao",
    files=files
)

print(response.json())
