import pandas as pd
import googlemaps
import time
from itertools import combinations

def calcular_distancia_tempo(input_tsv: str, output_tsv: str, api_key: str):
    """
    Função que lê um arquivo .tsv contendo nomes e endereços, calcula a distância e o tempo entre todos os pares de endereços
    usando a API do Google Maps, e salva os resultados num novo arquivo .tsv.
    
    Parâmetros:
    - input_tsv (str): Caminho para o arquivo .tsv de entrada contendo os nomes e endereços.
    - output_tsv (str): Caminho para salvar o arquivo .tsv de saída com as distâncias e tempos calculados.
    - api_key (str): Chave de API do Google Maps.
    """
    
    # Inicializar o cliente da API do Google Maps
    gmaps = googlemaps.Client(key=api_key)
    
    # Ler o arquivo .tsv de entrada (esperado ter colunas 'nome' e 'endereço')
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Gerar todas as combinações possíveis de pares de nomes e endereços
    pares = list(combinations(df.itertuples(index=False), 2))
    
    # Criar lista para armazenar os resultados
    resultados = []
    
    # Iterar sobre os pares de endereços
    for origem, destino in pares:
        origem_nome, origem_endereco = origem.nome, origem.endereço
        destino_nome, destino_endereco = destino.nome, destino.endereço
        
        try:
            # Fazer a requisição para a API do Google Maps
            resultado = gmaps.distance_matrix(origins=origem_endereco, destinations=destino_endereco, mode="driving")
            
            # Extrair distância e tempo da resposta
            distancia = resultado['rows'][0]['elements'][0]['distance']['value'] / 1000  # Converter metros para km
            duracao = resultado['rows'][0]['elements'][0]['duration']['value'] / 60  # Converter segundos para minutos
            
            # Adicionar resultado à lista
            resultados.append([origem_nome, origem_endereco, destino_nome, destino_endereco, distancia, duracao])
        
        except Exception as e:
            print(f"Erro ao processar {origem_nome} -> {destino_nome}: {e}")
            resultados.append([origem_nome, origem_endereco, destino_nome, destino_endereco, None, None])
        
        # Aguardar para evitar limite de requisições
        time.sleep(1)
    
    # Criar DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados, columns=['origem_nome', 'origem_endereco', 'destino_nome', 'destino_endereco', 'distância(km)', 'tempo(min)'])
    
    # Salvar o DataFrame em um arquivo .tsv
    df_resultados.to_csv(output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    input_tsv = "./fabricantes.tsv"  # Substitua pelo caminho do arquivo .tsv de entrada
    output_tsv = "./custos_percursos.tsv"  # Substitua pelo caminho do arquivo .tsv de saída
    api_key = "YOUR_GOOGLE_MAPS_API_KEY"  # Substitua pela sua chave da API do Google Maps
    
    calcular_distancia_tempo(input_tsv, output_tsv, api_key)