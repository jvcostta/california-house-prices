import os
import joblib
from utils import carregar_dados

# Definir o caminho do modelo treinado
caminho_modelo = os.path.abspath("models/modelo.pkl")

# Verificar se o modelo existe antes de carregar
if not os.path.exists(caminho_modelo):
    raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")

# Carregar o modelo
modelo = joblib.load(caminho_modelo)
print(f"Modelo carregado de: {caminho_modelo}")

# Carregar os dados
df = carregar_dados()

# Selecionar um exemplo para previsão
exemplo = df.drop(columns=['Median_House_Value']).iloc[0].values.reshape(1, -1)

# Fazer previsão
preco_predito = modelo.predict(exemplo)

print(f"Preço previsto da casa: ${preco_predito[0]:.2f}")
