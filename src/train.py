import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import carregar_dados

# Carregar os dados
df = carregar_dados()

X = df.drop(columns=['Median_House_Value'])
y = df['Median_House_Value']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")
print(f"parametros X_test:\n{X_test}")

# Definir caminho do modelo
caminho_modelo = os.path.abspath("models/modelo.pkl")

# Criar o diretório models/ se ele não existir
os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)

# Salvar o modelo
joblib.dump(modelo, caminho_modelo)
print(f"Modelo salvo em: {caminho_modelo}")
