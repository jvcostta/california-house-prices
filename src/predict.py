import joblib
from src.utils import carregar_dados

modelo = joblib.load("../models/modelo.pkl")

df = carregar_dados()
exemplo = df.drop(columns=['median_house_value']).iloc[0].values.reshape(1, -1) 

preco_predito = modelo.predict(exemplo)

print(f"Preço previsto da casa: ${preco_predito[0]:.2f}")
