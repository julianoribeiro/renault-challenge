import csv
import time
import googlemaps
from geopy.distance import geodesic

# Substitua 'YOUR_API_KEY' pela sua chave de API do Google Maps
gmaps = googlemaps.Client(key='AIzaSyCyUcw780lVX8gMFvYLLe77HVkEt_4EDmo')

def processar_endereco(campos):
    # Inicializar variáveis
    cep = ''
    estado = ''
    cidade = ''
    bairro = ''
    logradouro = ''
    numero = ''
    
    # Remover espaços em branco dos campos
    campos = [campo.strip() for campo in campos if campo.strip()]
    
    # Verificar se o primeiro campo é um CEP válido (8 dígitos numéricos)
    if campos[0].isdigit() and len(campos[0]) == 8:
        cep = campos[0]
        campos = campos[1:]  # Remover o CEP da lista de campos
    else:
        # Verificar se algum campo é um CEP
        for i, campo in enumerate(campos):
            if campo.strip().replace('-', '').isdigit() and len(campo.strip().replace('-', '')) == 8:
                cep = campo.strip()
                campos.pop(i)
                break
    
    # Identificar os demais campos
    for campo in campos:
        if campo.strip().upper() in ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 
                                     'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 
                                     'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']:
            estado = campo.strip()
        elif not cidade:
            cidade = campo.strip()
        elif any(char.isdigit() for char in campo) and not numero:
            numero = campo.strip()
        elif not logradouro:
            logradouro = campo.strip()
        else:
            logradouro += ', ' + campo.strip()
    
    # Construir o endereço formatado
    endereco_formatado = ', '.join(filter(None, [logradouro, numero, bairro, cidade, estado, cep, 'Brasil']))
    
    return endereco_formatado if endereco_formatado else None

# Ler endereços do arquivo .txt
with open('enderecosOutbound.txt', 'r', encoding='utf-8') as file:
    linhas = [linha.strip() for linha in file if linha.strip()]

enderecos_processados = {}
for linha in linhas:
    campos = linha.split(',')
    endereco_formatado = processar_endereco(campos)
    if endereco_formatado:
        enderecos_processados[linha] = endereco_formatado
    else:
        print(f"Endereço inválido ou incompleto: {linha}")
        enderecos_processados[linha] = None

# Geocodificar endereços
coordenadas = {}
for endereco_original, endereco_formatado in enderecos_processados.items():
    if endereco_formatado:
        tentativa = 0
        sucesso = False
        while tentativa < 3 and not sucesso:
            try:
                geocode_result = gmaps.geocode(endereco_formatado)
                if geocode_result:
                    location = geocode_result[0]['geometry']['location']
                    coordenadas[endereco_original] = (location['lat'], location['lng'])
                    sucesso = True
                else:
                    print(f"Não foi possível geocodificar o endereço: {endereco_formatado}")
                    coordenadas[endereco_original] = None
                    sucesso = True  # Para sair do loop
            except Exception as e:
                tentativa += 1
                print(f"Erro ao geocodificar {endereco_formatado}: {e}. Tentativa {tentativa} de 3.")
                time.sleep(1)  # Espera antes de tentar novamente
        if not sucesso:
            coordenadas[endereco_original] = None
    else:
        coordenadas[endereco_original] = None

# Calcular distâncias e salvar em um arquivo CSV
with open('distancias.csv', 'w', newline='', encoding='utf-8') as csvfile:
    campos_csv = ['origem', 'destino', 'distancia']
    escritor = csv.DictWriter(csvfile, fieldnames=campos_csv, delimiter=';')
    escritor.writeheader()
    
    enderecos_originais = list(coordenadas.keys())
    for i, origem in enumerate(enderecos_originais):
        for destino in enderecos_originais[i+1:]:
            coords_origem = coordenadas[origem]
            coords_destino = coordenadas[destino]
            if coords_origem and coords_destino:
                distancia = geodesic(coords_origem, coords_destino).kilometers
                escritor.writerow({'origem': origem, 'destino': destino, 'distancia': distancia})
            else:
                escritor.writerow({'origem': origem, 'destino': destino, 'distancia': 'N/A'})
