import googlemaps
from urllib.parse import urlparse, parse_qs, unquote 

def extract_route_info(google_maps_url, custo_tempo_parada=0):

    # Substitua pela sua chave de API
    API_KEY = 'your_api_key'
    
    # Crie um cliente da API do Google Maps
    gmaps = googlemaps.Client(key=API_KEY)

    # Extraia os pontos de origem, destino e waypoints da URL
    parsed_url = urlparse(google_maps_url)
    
    # Desconsidera tudo após "/@"
    path = parsed_url.path.split('/@')[0]
    path_parts = path.split('/dir/')[1].split('/')

    # A origem é o primeiro elemento, o destino é o último, e os waypoints são os elementos no meio
    origin = unquote(path_parts[0]).replace('+', ' ') if len(path_parts) > 0 else None
    destination = unquote(path_parts[-1]).replace('+', ' ') if len(path_parts) > 1 else None
    waypoints = [unquote(part).replace('+', ' ') for part in path_parts[1:-1] if part]

    """
    # Print da origem, destino e waypoints (para debug)
    print(f"Origem: {origin}")
    print(f"Destino: {destination}")
    print(f"Waypoints: {waypoints}")
    """
    
    if origin and destination:
        # Obtenha a rota completa usando a API de Direções
        directions_result = gmaps.directions(origin, destination, waypoints=waypoints, mode="driving")

        if directions_result:
            # Inicialize as variáveis para a duração e distância totais
            duration_total_seconds = 0
            distance_total = 0

            # Percorra cada trecho (leg) da rota e some a duração e a distância
            for leg in directions_result[0]['legs']:
                duration_total_seconds += leg['duration']['value']
                distance_total += leg['distance']['value']

            # Converta a duração total para minutos e adicione o custo de tempo por parada (em minutos)
            duration_total_minutes = (duration_total_seconds // 60) + (custo_tempo_parada * len(waypoints))

            # Converta a distância total para quilômetros como float
            distance_total_km = distance_total / 1000.0

            # Retorna os dados como um dicionário
            route_info = {
                "origem": origin,
                "destino": destination,
                "waypoints": waypoints,
                "distancia": distance_total_km,
                "duracao": duration_total_minutes
            }

            return route_info
        else:
            print("Não foi possível obter as direções.")
            return None
    else:
        print("Não foi possível encontrar a origem ou destino.")
        return None
    
if __name__ == '__main__':
    # Exemplo de uso
    google_maps_url = "https://www.google.com.br/maps/dir/Av.+Renault,+1300+-+Roseira,+S%C3%A3o+Jos%C3%A9+dos+Pinhais+-+PR/Hanon+Systems+Climatizacao+do+Brasil+Industria+e+Comercio+LTDA+-+Ponte+Alta,+Atibaia+-+SP/Av.+Renault,+1300+-+Roseira,+S%C3%A3o+Jos%C3%A9+dos+Pinhais+-+PR/@-24.0279753,-48.1222133,8z/data=!4m20!4m19!1m5!1m1!1s0x94dcf4056b5c4b7b:0x4f5bf93662e20ed1!2m2!1d-49.1133185!2d-25.5229496!1m5!1m1!1s0x94cecfdf5e407fe5:0x2eec07ac2eed5643!2m2!1d-46.6702738!2d-23.0421892!1m5!1m1!1s0x94dcf4056b5c4b7b:0x4f5bf93662e20ed1!2m2!1d-49.1133185!2d-25.5229496!3e0?hl=pt-BR&entry=ttu&g_ep=EgoyMDI0MDgyMS4wIKXMDSoASAFQAw%3D%3D"
    custo_tempo_parada = 15
    route_info = extract_route_info(google_maps_url, custo_tempo_parada)

    print(route_info)