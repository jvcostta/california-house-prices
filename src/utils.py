import pandas as pd
import os
#carregando base de dados
def carregar_dados():
    caminho_arquivo = os.path.join(os.path.dirname(__file__), "../data/california.csv")

    df = pd.read_csv(caminho_arquivo)

    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df
