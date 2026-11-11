import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pickle import dump

# 1. Carregamento e limpeza inicial
dados = pd.read_csv('HousingData.csv')
dados = dados.dropna().reset_index(drop=True)

# 2. Separação de dados categóricos (CHAS) e quantitativos
dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']
dados_cat_rotulos = 'CHAS'

# 3. Normalização dos dados quantitativos
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)

dados_num_normalizados = modelo_normalizador.transform(dados_num)
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

# 4. Tratamento dos dados categóricos (get_dummies)
dados_cat_normalizado = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)

# 5. Recomposição da base pronta para o treinamento
dados_finais = pd.concat([dados_num_normalizados, dados_cat_normalizado], axis=1)

# Salvar a estrutura de colunas para garantir o alinhamento perfeito na inferência
dump(dados_finais.columns.tolist(), open('colunas_modelo.pkl', 'wb'))

# Salvar o modelo normalizador treinado
dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

# 6. Método do Cotovelo para encontrar o K ótimo
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(dados_finais)
    inertias.append(kmeans.inertia_)

# Gerar e salvar o gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', linestyle='-')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig('elbow.png')
print("Gráfico do cotovelo salvo como 'elbow.png'. Observando o gráfico, K=3 é o ponto ideal.")

# 7. Treinamento do modelo final com K=3 (definido pelo cotovelo)
modelo_cluster = KMeans(n_clusters=3, random_state=42, n_init=10)
modelo_cluster.fit(dados_finais)

# Salvar o modelo clusterizador
dump(modelo_cluster, open('cluster_boston.model', 'wb'))
print("Módulo de treinamento finalizado com sucesso. Modelos salvos.")