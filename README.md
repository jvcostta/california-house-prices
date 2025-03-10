# california-house-prices

Este projeto usa **Machine Learning** para prever preços de casas na Califórnia com base em diversas características do imóvel. Ele utiliza **Regressão Linear** para modelar os preços e possibilita a reutilização do modelo salvo.

##  Objetivo
O objetivo deste projeto é construir um modelo de regressão linear para prever os preços de casas na Califórnia com base em diversas variáveis do dataset `california.csv`. Utilizei a biblioteca `scikit-learn` para treinar o modelo e a biblioteca `joblib` para salvá-lo e reutilizá-lo posteriormente.

Utilizei:
- **Pandas** → Para carregar e manipular os dados.
- **Scikit-learn** → Para modelagem e avaliação.
- **Joblib** → Para salvar e carregar o modelo.


##  Como Usar

###  1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/PrevisaoPrecoCasa.git
cd PrevisaoPrecoCasa
```
### 2. Criar um Ambiente Virtual
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```
### 4. Adicionar o Dataset
Baixe o dataset California Housing Prices do Kaggle e coloque dentro da pasta `data/`.
```bash
https://www.kaggle.com/datasets/fedesoriano/california-housing-prices-data-extra-features/discussion?sort=hotness
```
Renomeie o arquivo para `california.csv`.

### 5. Treinar o Modelo
Agora, basta rodar o script de **treinamento** para gerar o modelo:
```bash
python src/train.py
```
Saída esperada:
```bash
Dataset carregado com 20640 linhas e 14 colunas.
MAE: 50404.85
MSE: 4.8e+09
R²: 0.63
Modelo salvo em: models/modelo.pkl
```
### 6. Fazer uma Predição
Após o treinamento, podemos carregar o modelo e prever preços de casas:
```bash
python src/predict.py
```
Saída esperada:
```bash
Modelo carregado de: ..\PrevisaoPrecoCasa\models\modelo.pkl
Dataset carregado com 20640 linhas e 14 colunas.
Preço previsto da casa: $411124.12
```
## Licença
Este projeto é de código aberto e pode ser utilizado livremente.