import googlemaps
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import json

MY_KEY = 'AIzaSyCMKdKZUSLFBIWjidxyeshLmbJxyd72Jwc'

gmaps = googlemaps.Client(key=MY_KEY)

origem = "Maringá, PR"
destino = "Paranavaí, PR"
waypoints = ["Castelo Branco, PR", "Alto Paraná, PR"]  # Pontos intermediários

now = datetime.now()
directions_result = gmaps.directions(origem,
                                     destino,
                                     waypoints=waypoints,
                                     mode="driving",
                                     departure_time=now,
                                     language="pt-BR",
                                     region="br")

# Salvando o directions_result em um arquivo JSON
if directions_result:
    # Nome do arquivo onde queremos salvar
    with open('directions_result.json', 'w', encoding='utf-8') as json_file:
        json.dump(directions_result, json_file, ensure_ascii=False, indent=4)

    print("directions_result salvo em 'directions_result.json' com sucesso!")

# Extraia os pontos de caminho (polyline) da rota
if directions_result:
    # Pegue o primeiro 'leg' da rota e a polyline
    route = directions_result[0]['overview_polyline']['points']

    # Gera uma URL para o mapa estático com a rota
    static_map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"size=640x640"  # Tamanho da imagem
        f"&path=enc:{route}"  # Caminho da rota (encoded polyline)
        f"&markers=color:blue|label:S|{origem}"  # Marcador de início
        f"&markers=color:red|label:E|{destino}"  # Marcador de fim
    )
    # Adiciona os marcadores dos waypoints
    for i, waypoint in enumerate(waypoints):
        static_map_url += f"&markers=color:green|label:{chr(65+i)}|{waypoint}"  # Marcadores para waypoints com letras A, B...

    static_map_url += f"&key={MY_KEY}"

    print("URL da imagem com a rota: ", static_map_url)
else:
    print("Nenhuma rota encontrada.")

response = requests.get(static_map_url)

if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img.show() 
else:
    print("Erro ao baixar a imagem.")