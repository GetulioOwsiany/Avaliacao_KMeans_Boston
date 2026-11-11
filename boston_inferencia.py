import pandas as pd
from pickle import load

# 1. Carregar os modelos e a estrutura de colunas salvos no treinamento
modelo_normalizador = load(open('modelo_normalizador.model', 'rb'))
modelo_cluster = load(open('cluster_boston.model', 'rb'))
colunas_modelo = load(open('colunas_modelo.pkl', 'rb'))

# 2. Receber dados de uma nova residência (exemplo com valores fictícios)
nova_residencia = pd.DataFrame([{
    'CRIM': 0.05, 'ZN': 20.0, 'INDUS': 3.0, 'CHAS': 1.0, 
    'NOX': 0.45, 'RM': 7.0, 'AGE': 50.0, 'DIS': 5.0, 
    'RAD': 3, 'TAX': 250, 'PTRATIO': 16.0, 'B': 390.0, 
    'LSTAT': 6.5, 'MEDV': 30.0
}])

# 3. Separar os dados numéricos e categóricos da nova instância
instancia_num = nova_residencia.drop(columns=['CHAS'])
instancia_cat = nova_residencia['CHAS']
dados_cat_rotulos = 'CHAS'

# 4. Normalizar os dados quantitativos com o modelo carregado
instancia_num_normalizada = modelo_normalizador.transform(instancia_num)
instancia_num_normalizada = pd.DataFrame(instancia_num_normalizada, columns=instancia_num.columns)

# 5. Aplicar get_dummies no dado categórico
instancia_cat_normalizada = pd.get_dummies(instancia_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)

# 6. Combinar as bases e alinhar com a estrutura do treinamento
instancia_final = pd.concat([instancia_num_normalizada, instancia_cat_normalizada], axis=1)

# O reindex preenche automaticamente com 0 colunas do get_dummies que existiam no treino mas faltaram nesta instância única
instancia_final = instancia_final.reindex(columns=colunas_modelo, fill_value=0)

# 7. Realizar a inferência
cluster_inferido = modelo_cluster.predict(instancia_final)

print(f"Os dados da nova residência foram normalizados e classificados no Grupo (Cluster): {cluster_inferido[0]}")