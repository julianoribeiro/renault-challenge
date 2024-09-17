import pandas as pd
import numpy as np
import random
from collections import defaultdict
import logging
import json
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes
NUM_DOCAS = 9
LOADING_TIME = 1  # 1 hora de tempo de carga
MAX_MORPHOLOGY = 11
IDEAL_MORPHOLOGY = 9.6
MAX_DAYS = 3
PERIODS_PER_DAY = 2  # manhã e tarde
MAX_ATTEMPTS = 300

# Classe para representar uma entrega
class Delivery:
    def __init__(self, route_id, carrier, day, period, dock):
        self.route_id = route_id
        self.carrier = carrier
        self.day = day
        self.period = period
        self.dock = dock
        self.cars = []
        self.total_morphology = 0
        self.total_distance = 0
        self.cost = 0

    def add_car(self, vin, model, origin, destination, city, state, morphology):
        if self.total_morphology + morphology <= MAX_MORPHOLOGY and morphology > 0:
            self.cars.append({
                'vin': vin,
                'model': model,
                'origin': origin,
                'destination': destination,
                'city': city,
                'state': state,
                'morphology': morphology
            })
            self.total_morphology += morphology
            return True
        return False

    def calculate_distance(self, distances):
        if not self.cars:
            return 0
        
        total_distance = 0
        current_location = self.cars[0]['origin']
        
        for car in self.cars:
            distance_key = f"{current_location};{car['destination']}"
            if distance_key in distances:
                total_distance += distances[distance_key]
            current_location = car['destination']
        
        # Adicionar distância de retorno à origem
        return_distance_key = f"{current_location};{self.cars[0]['origin']}"
        if return_distance_key in distances:
            total_distance += distances[return_distance_key]
        
        self.total_distance = total_distance
        return total_distance

    def calculate_cost(self, costs):
        # Encontrar a faixa de custo apropriada
        for km, cost_per_km in sorted(costs.items(), key=lambda x: x[0], reverse=True):
            if self.total_distance >= km:
                self.cost = self.total_distance * cost_per_km
                break
        return self.cost
    
# Classe para representar uma solução
class Solution:
    def __init__(self, deliveries):
        self.deliveries = deliveries
        self.fitness = float('inf')
        self.total_cost = 0
        self.total_distance = 0

    def calculate_fitness(self, distances, costs, total_cars):
        self.total_distance = sum(delivery.calculate_distance(distances) for delivery in self.deliveries)
        self.total_cost = sum(delivery.calculate_cost(costs) for delivery in self.deliveries)
        penalty = self.calculate_penalty(total_cars)
        #self.fitness = total_distance + penalty
        self.fitness = (self.total_distance * 0.3) + (self.total_cost * 0.7) + penalty
        return self.fitness

    def calculate_penalty(self, total_cars):
        penalty = 0
        allocated_cars = sum(len(delivery.cars) for delivery in self.deliveries)
        
        # Penalidade severa para carros não alocados
        unallocated_cars = total_cars - allocated_cars
        penalty += unallocated_cars * 10000  # Penalidade alta para cada carro não alocado
        
        for delivery in self.deliveries:
            # Penalidade para morfologia abaixo do ideal
            if delivery.total_morphology < IDEAL_MORPHOLOGY:
                penalty += (IDEAL_MORPHOLOGY - delivery.total_morphology) * 100
            
            # Penalidade para rotas com poucos carros
            if len(delivery.cars) < 5:
                penalty += (5 - len(delivery.cars)) * 100
        
        return penalty

# Funções para leitura dos dados de entrada
def read_morphologies(file_path):
    df = pd.read_csv(file_path, sep=';')
    morphologies = dict(zip(df['Modelo'], df['MORFOLOGIA']))
    logging.info(f"Morfologias lidas: {morphologies}")
    return morphologies

def read_carriers(file_path):
    df = pd.read_csv(file_path, sep=';')
    carriers = defaultdict(list)
    for _, row in df.iterrows():
        carriers[row['Cidade']].append(row['Transportadora'])

    return carriers

def read_distances(file_path):
    df = pd.read_csv(file_path, sep=';')
    return dict(zip(df['origem'] + ';' + df['destino'], df['distancia']))

def read_outbound(file_path):
    df = pd.read_csv(file_path, sep=';')
    # Filtrar linhas com destino válido
    df_valid = df.dropna(subset=['Destino'])
    if len(df) != len(df_valid):
        logging.warning(f"Removidas {len(df) - len(df_valid)} linhas com destino inválido.")
    return df_valid

def read_costs(file_path):
    df = pd.read_csv(file_path, sep=';')
    return dict(zip(df['Km'], df['Cegonha']))

# Função para gerar uma população inicial
def generate_initial_population(outbound_data, carriers, morphologies, population_size):
    population = []
    total_cars = len(outbound_data)

    for _ in range(population_size):
        deliveries = []
        unassigned_cars = outbound_data.copy()
        unassigned_mask = pd.Series(True, index=unassigned_cars.index)
        route_id = 1

        while unassigned_mask.any():
            # Selecionar uma cidade com carros não alocados
            cities_with_unassigned_cars = unassigned_cars.loc[unassigned_mask, 'Cidade']
            if cities_with_unassigned_cars.empty:
                break
            city = cities_with_unassigned_cars.value_counts().idxmax()

            # Selecionar uma transportadora que atenda a cidade
            possible_carriers = carriers.get(city, [])
            if not possible_carriers:
                # Marcar os carros desta cidade como não alocáveis
                unassigned_mask &= unassigned_cars['Cidade'] != city
                continue
            carrier = random.choice(possible_carriers)

            day = random.randint(1, MAX_DAYS)
            period = random.choice(['morning', 'afternoon'])
            dock = f"Doca{random.randint(1, NUM_DOCAS)}"

            delivery = Delivery(f"Route{route_id}", carrier, day, period, dock)

            # Adicionar carros da mesma cidade à rota
            city_mask = (unassigned_cars['Cidade'] == city) & unassigned_mask
            city_cars = unassigned_cars[city_mask]

            for idx, car in city_cars.iterrows():
                morphology = morphologies.get(car['Modelo'], None)
                if morphology is None:
                    continue
                if delivery.add_car(car['VIN'], car['Modelo'], car['Origem'], car['Destino'], car['Cidade'], car['Estado'], morphology):
                    unassigned_mask.at[idx] = False
                else:
                    break  # Se não puder adicionar mais carros, parar

            # Se ainda houver capacidade, adicionar carros do mesmo estado
            if delivery.total_morphology < MAX_MORPHOLOGY:
                state = car['Estado']  # Usar o estado da cidade atual
                state_mask = (unassigned_cars['Estado'] == state) & unassigned_mask & (~city_mask)
                state_cars = unassigned_cars[state_mask]

                for idx, car in state_cars.iterrows():
                    morphology = morphologies.get(car['Modelo'], None)
                    if morphology is None:
                        continue
                    if delivery.add_car(car['VIN'], car['Modelo'], car['Origem'], car['Destino'], car['Cidade'], car['Estado'], morphology):
                        unassigned_mask.at[idx] = False
                    else:
                        break  # Se não puder adicionar mais carros, parar

            # Se ainda houver capacidade, adicionar carros de outros estados
            if delivery.total_morphology < MAX_MORPHOLOGY:
                other_mask = unassigned_mask & (~city_mask) & (~state_mask)
                other_cars = unassigned_cars[other_mask]
                for idx, car in other_cars.iterrows():
                    morphology = morphologies.get(car['Modelo'], None)
                    if morphology is None:
                        continue
                    if delivery.add_car(car['VIN'], car['Modelo'], car['Origem'], car['Destino'], car['Cidade'], car['Estado'], morphology):
                        unassigned_mask.at[idx] = False
                    else:
                        break  # Se não puder adicionar mais carros, parar

            if delivery.cars:
                deliveries.append(delivery)
                route_id += 1

        solution = Solution(deliveries)
        population.append(solution)

    return population



# Função de seleção por torneio
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x.fitness)

# Função de crossover
def crossover(parent1, parent2):
    child_deliveries = []
    used_cars = set()
    
    # Escolher pontos de corte
    size = min(len(parent1.deliveries), len(parent2.deliveries))
    if size < 2:
        return parent1  # Não é possível aplicar crossover
    
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)
    
    # Iniciar o filho com rotas do primeiro pai entre os pontos de corte
    child_routes = parent1.deliveries[cx_point1:cx_point2]
    used_cars = {car['vin'] for delivery in child_routes for car in delivery.cars}
    
    # Preencher o restante com rotas do segundo pai sem duplicar carros
    for delivery in parent2.deliveries:
        if all(car['vin'] not in used_cars for car in delivery.cars):
            child_routes.append(delivery)
            used_cars.update(car['vin'] for car in delivery.cars)
    
    # Adicionar rotas faltantes do primeiro pai, se necessário
    for delivery in parent1.deliveries:
        if all(car['vin'] not in used_cars for car in delivery.cars):
            child_routes.append(delivery)
            used_cars.update(car['vin'] for car in delivery.cars)
    
    # **Remover entregas vazias após o crossover**
    child_routes = [delivery for delivery in child_routes if delivery.cars]
    
    # Criar a nova solução filho com as rotas resultantes
    return Solution(child_routes)



# Função de mutação
def mutate(solution, carriers, mutation_rate):
    deliveries = solution.deliveries
    num_mutations = max(1, int(len(deliveries) * mutation_rate))

    # Mutação de troca de carros entre entregas, respeitando o agrupamento
    for _ in range(num_mutations):
        delivery1, delivery2 = random.sample([d for d in deliveries if d.cars], 2)
        if delivery1 != delivery2:
            # Verificar se as entregas são do mesmo agrupamento (cidade ou estado)
            cities_delivery1 = set(car['city'] for car in delivery1.cars)
            cities_delivery2 = set(car['city'] for car in delivery2.cars)

            # Permitir trocas apenas se as entregas tiverem cidades em comum ou estados em comum
            if cities_delivery1 & cities_delivery2:
                # Cidades em comum, pode trocar
                pass
            else:
                states_delivery1 = set(car['state'] for car in delivery1.cars)
                states_delivery2 = set(car['state'] for car in delivery2.cars)
                if not states_delivery1 & states_delivery2:
                    continue  # Não trocar se não houver estados em comum

            car1 = random.choice(delivery1.cars)
            car2 = random.choice(delivery2.cars)

            # Calcular nova morfologia após a troca
            new_morph1 = delivery1.total_morphology - car1['morphology'] + car2['morphology']
            new_morph2 = delivery2.total_morphology - car2['morphology'] + car1['morphology']

            # Verificar se a troca é válida
            if new_morph1 <= MAX_MORPHOLOGY and new_morph2 <= MAX_MORPHOLOGY:
                # Trocar os carros
                delivery1.cars.remove(car1)
                delivery1.cars.append(car2)
                delivery1.total_morphology = new_morph1

                delivery2.cars.remove(car2)
                delivery2.cars.append(car1)
                delivery2.total_morphology = new_morph2

    # Remover entregas vazias, se houver
    solution.deliveries = [delivery for delivery in deliveries if delivery.cars]

    return solution



# Algoritmo genético principal
def genetic_algorithm(outbound_data, carriers, morphologies, distances, costs, population_size=100, generations=300, tournament_size=5, crossover_rate=0.8, mutation_rate=0.2, reset_threshold=50, reset_percentage=0.3):
    total_cars = len(outbound_data)
    
    try:
        population = generate_initial_population(outbound_data, carriers, morphologies, population_size)
    except ValueError as e:
        logging.error(f"Falha ao gerar população inicial: {e}")
        return None

    best_fitness = float('inf')
    best_solution = None
    generations_without_improvement = 0
    
    for generation in range(generations):
        # Avaliar a população
        for solution in population:
            solution.calculate_fitness(distances, costs, total_cars)
        
        # Ordenar a população pelo fitness
        population.sort(key=lambda x: x.fitness)
        
        # Verificar se houve melhoria
        if population[0].fitness < best_fitness:
            best_fitness = population[0].fitness
            best_solution = population[0]
            generations_without_improvement = 0
            print(f"Generation {generation + 1}: New Best Fitness = {best_fitness:.2f}, Total Cost = {best_solution.total_cost:.2f}, Total Distance = {best_solution.total_distance:.2f}")
        else:
            generations_without_improvement += 1
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.2f}, Total Cost = {best_solution.total_cost:.2f}, Total Distance = {best_solution.total_distance:.2f}")
        
        # Verificar se é necessário fazer reset parcial
        if generations_without_improvement >= reset_threshold:
            print(f"Resetting {reset_percentage*100}% of the population due to {generations_without_improvement} generations without improvement")
            reset_size = int(population_size * reset_percentage)
            try:
                new_individuals = generate_initial_population(outbound_data, carriers, morphologies, reset_size)
                population = population[:population_size - reset_size] + new_individuals
            except ValueError as e:
                logging.warning(f"Falha ao gerar novos indivíduos para reset: {e}")
            generations_without_improvement = 0
            continue
        
        new_population = []
        
        # Elitismo: manter os melhores indivíduos
        elite_size = int(population_size * 0.1)  # 10% de elitismo
        new_population.extend(population[:elite_size])
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            child = mutate(child, carriers, mutation_rate)

            
            # Aplicar busca local com uma certa probabilidade
            if random.random() < 0.1:  # 10% de chance de aplicar busca local
                child = local_search(child, distances, costs, total_cars)
            
            new_population.append(child)
        
        population = new_population
    
    return best_solution

def local_search(solution, distances, costs, total_cars):
    for _ in range(10):  # Número de iterações da busca local
        delivery1, delivery2 = random.sample([d for d in solution.deliveries if len(d.cars) > 0], 2)
        if delivery1 != delivery2:
            # Verificar se as entregas são do mesmo agrupamento (cidade ou estado)
            cities_delivery1 = set(car['city'] for car in delivery1.cars)
            cities_delivery2 = set(car['city'] for car in delivery2.cars)

            if cities_delivery1 & cities_delivery2:
                # Cidades em comum, pode trocar
                pass
            else:
                states_delivery1 = set(car['state'] for car in delivery1.cars)
                states_delivery2 = set(car['state'] for car in delivery2.cars)
                if not states_delivery1 & states_delivery2:
                    continue  # Não trocar se não houver estados em comum

            car1 = random.choice(delivery1.cars)
            car2 = random.choice(delivery2.cars)

            # Calcular nova morfologia após a troca
            new_morph1 = delivery1.total_morphology - car1['morphology'] + car2['morphology']
            new_morph2 = delivery2.total_morphology - car2['morphology'] + car1['morphology']

            # Verificar se a troca é válida
            if new_morph1 <= MAX_MORPHOLOGY and new_morph2 <= MAX_MORPHOLOGY:
                # Trocar os carros
                delivery1.cars.remove(car1)
                delivery1.cars.append(car2)
                delivery1.total_morphology = new_morph1

                delivery2.cars.remove(car2)
                delivery2.cars.append(car1)
                delivery2.total_morphology = new_morph2

    # Recalcular o fitness após a busca local
    solution.calculate_fitness(distances, costs, total_cars)
    return solution




def save_solution_to_json(solution, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_outbound_solution_{timestamp}.json"
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    solution_dict = {
        "fitness": solution.fitness,
        "total_routes": len(solution.deliveries),
        "total_distance": solution.total_distance,
        "total_routes": solution.total_cost,
        "total_cars_allocated": sum(len(delivery.cars) for delivery in solution.deliveries),
        "deliveries": [
            {
                "route_id": d.route_id,
                "carrier": d.carrier,
                "day": convert_to_serializable(d.day),
                "period": d.period,
                "dock": d.dock,
                "total_morphology": convert_to_serializable(d.total_morphology),
                "distance": convert_to_serializable(d.total_distance),
                "cost": convert_to_serializable(d.cost),
                "cars": [
                    {
                        "vin": car['vin'],
                        "model": car['model'],
                        "origin": car['origin'],
                        "destination": car['destination'],
                        "city": car['city'],
                        "morphology": convert_to_serializable(car['morphology'])
                    } for car in d.cars
                ]
            } for d in solution.deliveries
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=4, default=convert_to_serializable)
    
    print(f"Solução salva em {filename}")

# Função principal
def main():
    # Ler dados de entrada
    morphologies = read_morphologies('morfologies.csv')
    carriers = read_carriers('carrier.csv')
    distances = read_distances('distances.csv')
    outbound_data = read_outbound('outbound.csv')
    costs = read_costs('costOutbound.csv')
    
    # Verificar dados
    print(f"Número de morfologias: {len(morphologies)}")
    print(f"Número de cidades com transportadoras: {len(carriers)}")
    logging.info(f"Número de distâncias: {len(distances)}")
    print(f"Número de carros para entrega: {len(outbound_data)}")
    logging.info(f"Número de faixas de custo: {len(costs)}")
    
    # Verificar se todos os modelos em outbound_data têm uma morfologia correspondente
    missing_morphologies = set(outbound_data['Modelo']) - set(morphologies.keys())
    if missing_morphologies:
        logging.error(f"Modelos sem morfologia definida: {missing_morphologies}")
    
    # Verificar se todas as cidades em outbound_data têm transportadoras definidas
    missing_carriers = set(outbound_data['Cidade']) - set(carriers.keys())
    if missing_carriers:
        logging.error(f"Cidades sem transportadoras definidas: {missing_carriers}")
    
    # Verificar se há transportadoras disponíveis para todas as cidades
    cities_without_carriers = [city for city, carrier_list in carriers.items() if not carrier_list]
    if cities_without_carriers:
        logging.error(f"Cidades sem transportadoras disponíveis: {cities_without_carriers}")
    
    # Imprimir algumas estatísticas
    total_morphology = sum(outbound_data['Modelo'].map(morphologies))
    logging.info(f"Morfologia total de todos os carros: {total_morphology}")
    estimated_min_routes = total_morphology / MAX_MORPHOLOGY
    print(f"Número mínimo estimado de rotas necessárias: {estimated_min_routes:.2f}")

    # Imprimir informações detalhadas sobre os carros
    logging.info("Detalhes dos carros para entrega:")
    #for _, car in outbound_data.iterrows():
    #    morph = morphologies.get(car['Modelo'], 'N/A')
    #    carrier = carriers.get(car['Cidade'], 'N/A')
    #    logging.info(f"VIN: {car['VIN']}, Modelo: {car['Modelo']}, Morfologia: {morph}, Cidade: {car['Cidade']}, Transportadora: {carrier}")

    # Verificar se há distâncias para todas as combinações necessárias
    missing_distances = []
    for _, car in outbound_data.iterrows():
        origin = car['Origem']
        destination = car['Destino']
        if f"{origin};{destination}" not in distances:
            missing_distances.append((origin, destination))
    
    if missing_distances:
        logging.error(f"Distâncias não encontradas para {len(missing_distances)} rotas:")
        for origin, destination in missing_distances:
            logging.error(f"  {origin} -> {destination}")

   # Executar o algoritmo genético
    print("Iniciando o algoritmo genético...")
    best_solution = genetic_algorithm(outbound_data, carriers, morphologies, distances, costs)
    
    if best_solution:
        total_cars = len(outbound_data)
        allocated_cars = sum(len(delivery.cars) for delivery in best_solution.deliveries)
        
        if allocated_cars < total_cars:
            print("AVISO: A solução não alocou todos os carros!")
    
    
        print("Melhor solução encontrada:")
        print(f"Fitness: {best_solution.fitness}")
        print(f"Distância total: {best_solution.total_distance:.2f}")
        print(f"Custo total: {best_solution.total_cost:.2f}")
        print(f"Número total de rotas: {len(best_solution.deliveries)}")
        print(f"Número total de carros alocados: {sum(len(delivery.cars) for delivery in best_solution.deliveries)}")
        for delivery in best_solution.deliveries:
            print(f"\nRota: {delivery.route_id}")
            print(f"Transportadora: {delivery.carrier}")
            print(f"Dia: {delivery.day}, Período: {delivery.period}, Doca: {delivery.dock}")
            print(f"Morfologia total: {delivery.total_morphology}")
            print(f"Distância total: {delivery.total_distance}")
            print(f"Custo da rota: {delivery.cost:.2f}")
            print("Carros:")
            for car in delivery.cars:
                print(f"  VIN: {car['vin']}, Modelo: {car['model']}, Destino: {car['destination']}")
        save_solution_to_json(best_solution)
    else:
        print("Não foi possível encontrar uma solução válida.")
        print("Detalhes do problema:")
        print(f"Número total de carros: {len(outbound_data)}")
        print(f"Número de cidades: {len(outbound_data['Cidade'].unique())}")
        print(f"Número de transportadoras: {sum(len(carriers) for carriers in carriers.values())}")

if __name__ == "__main__":
    main()