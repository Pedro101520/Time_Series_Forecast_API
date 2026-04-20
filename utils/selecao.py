def calculo_ponderado(metricas, pesos):
    rmse = []
    mape = []
    for i in metricas:
        rmse.append(i[0])
    for j in metricas:
        mape.append(j[1])
    
    maior_rmse = max(rmse) or 1
    maior_mape = max(mape) or 1

    peso_rmse, peso_mape = pesos
    scores = []

    for rmse, mape in metricas:
        score = (peso_rmse * (rmse/maior_rmse) + peso_mape * (mape/maior_mape))
        scores.append(score)
    return scores