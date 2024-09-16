import pandas as pd
import numpy as np
import random
from collections import defaultdict
import logging
import json
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def add_car(self, vin, model, origin, destination, city, morphology):
        if self.total_morphology + morphology <= MAX_MORPHOLOGY:
            self.cars.append({
                'vin': vin,
                'model': model,
                'origin': origin,
                'destination': destination,
                'city': city,
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

# Classe para representar uma solução
class Solution:
    def __init__(self, deliveries):
        self.deliveries = deliveries
        self.fitness = float('inf')

    def calculate_fitness(self, distances, total_cars):
        total_distance = sum(delivery.calculate_distance(distances) for delivery in self.deliveries)
        penalty = self.calculate_penalty(total_cars)
        self.fitness = total_distance + penalty
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
            if len(delivery.cars) < 3:
                penalty += (3 - len(delivery.cars)) * 100
        
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
    max_attempts = MAX_ATTEMPTS
    total_cars = len(outbound_data)

    for _ in range(population_size):
        attempts = 0
        while attempts < max_attempts:
            deliveries = []
            unassigned_cars = outbound_data.copy()
            route_id = 1

            while not unassigned_cars.empty:
                available_cities = unassigned_cars['Cidade'].unique()
                possible_carriers = set.union(*[set(carriers.get(city, [])) for city in available_cities])
                if not possible_carriers:
                    logging.warning(f"Não há transportadoras disponíveis para as cidades restantes: {available_cities}")
                    break

                carrier = random.choice(list(possible_carriers))
                
                day = random.randint(1, MAX_DAYS)
                period = random.choice(['morning', 'afternoon'])
                dock = f"Doca{random.randint(1, NUM_DOCAS)}"
                
                delivery = Delivery(f"Route{route_id}", carrier, day, period, dock)
                
                # Criar uma cópia explícita dos carros elegíveis
                eligible_cars = unassigned_cars[unassigned_cars['Cidade'].isin([city for city, carrier_list in carriers.items() if carrier in carrier_list])].copy()
                
                # Adicionar a coluna de morfologia
                eligible_cars['morphology'] = eligible_cars['Modelo'].map(morphologies)
                
                # Ordenar carros por morfologia (do maior para o menor)
                eligible_cars = eligible_cars.sort_values('morphology', ascending=False)
                
                cars_added = 0
                for _, car in eligible_cars.iterrows():
                    if car['Modelo'] not in morphologies:
                        logging.warning(f"Modelo {car['Modelo']} não encontrado em morphologies")
                        continue
                    if delivery.add_car(car['VIN'], car['Modelo'], car['Origem'], car['Destino'], car['Cidade'], car['morphology']):
                        unassigned_cars = unassigned_cars[unassigned_cars['VIN'] != car['VIN']]
                        cars_added += 1
                    
                    if delivery.total_morphology >= IDEAL_MORPHOLOGY or len(delivery.cars) >= 11:
                        break
                
                if delivery.cars:
                    deliveries.append(delivery)
                    route_id += 1
                    logging.debug(f"Rota {route_id-1} criada com {cars_added} carros, morfologia total: {delivery.total_morphology}")
                else:
                    logging.debug(f"Não foi possível adicionar carros à rota {route_id} para a transportadora {carrier}")

            if unassigned_cars.empty:
                solution = Solution(deliveries)
                population.append(solution)
                logging.info(f"Solução válida gerada com {len(deliveries)} rotas após {attempts + 1} tentativas")
                break
            else:
                logging.debug(f"Tentativa {attempts + 1}: {len(unassigned_cars)} carros não alocados")
                attempts += 1

        if attempts >= max_attempts:
            logging.warning(f"Não foi possível gerar uma solução válida após {max_attempts} tentativas.")

    if not population:
        raise ValueError("Não foi possível gerar nenhuma solução válida para a população inicial.")

    return population

# Função de seleção por torneio
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x.fitness)

# Função de crossover
def crossover(parent1, parent2):
    child_deliveries = []
    used_cars = set()
    
    for delivery in parent1.deliveries + parent2.deliveries:
        new_delivery = Delivery(delivery.route_id, delivery.carrier, delivery.day, delivery.period, delivery.dock)
        for car in delivery.cars:
            if car['vin'] not in used_cars and new_delivery.add_car(**car):
                used_cars.add(car['vin'])
        if new_delivery.cars:
            child_deliveries.append(new_delivery)
    
    return Solution(child_deliveries)

# Função de mutação
def mutate(solution, carriers, mutation_rate):
    for delivery in solution.deliveries:
        if random.random() < mutation_rate:
            # Mudar transportadora
            if delivery.cars:
                common_carriers = set.intersection(*[set(carriers[car['city']]) for car in delivery.cars])
                if common_carriers:
                    delivery.carrier = random.choice(list(common_carriers))
                else:
                    # Se não houver transportadora comum, escolha uma aleatória que possa atender pelo menos uma cidade
                    possible_carriers = set.union(*[set(carriers[car['city']]) for car in delivery.cars])
                    if possible_carriers:
                        delivery.carrier = random.choice(list(possible_carriers))
        
        if random.random() < mutation_rate:
            # Mudar dia ou período
            delivery.day = random.randint(1, MAX_DAYS)
            delivery.period = random.choice(['morning', 'afternoon'])
        
        if random.random() < mutation_rate:
            # Mudar doca
            delivery.dock = f"Doca{random.randint(1, NUM_DOCAS)}"
        
        if random.random() < mutation_rate:
            # Tentar adicionar ou remover um carro
            if delivery.cars and len(delivery.cars) > 1:
                removed_car = delivery.cars.pop()
                delivery.total_morphology -= removed_car['morphology']
            elif len(delivery.cars) < len(solution.deliveries):
                other_delivery = random.choice(solution.deliveries)
                if other_delivery.cars:
                    car_to_move = other_delivery.cars.pop()
                    other_delivery.total_morphology -= car_to_move['morphology']
                    if delivery.add_car(**car_to_move):
                        pass
                    else:
                        other_delivery.add_car(**car_to_move)
    
    return solution

# Algoritmo genético principal
def genetic_algorithm(outbound_data, carriers, morphologies, distances, costs, population_size=100, generations=200, tournament_size=5, crossover_rate=0.8, mutation_rate=0.2, reset_threshold=50, reset_percentage=0.3):
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
            solution.calculate_fitness(distances, total_cars)
        
        # Ordenar a população pelo fitness
        population.sort(key=lambda x: x.fitness)
        
        # Verificar se houve melhoria
        if population[0].fitness < best_fitness:
            best_fitness = population[0].fitness
            best_solution = population[0]
            generations_without_improvement = 0
            logging.info(f"Generation {generation + 1}: New Best Fitness = {best_fitness}")
        else:
            generations_without_improvement += 1
            logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        
        # Verificar se é necessário fazer reset parcial
        if generations_without_improvement >= reset_threshold:
            logging.info(f"Resetting {reset_percentage*100}% of the population due to {generations_without_improvement} generations without improvement")
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
                child = local_search(child, distances, total_cars)
            
            new_population.append(child)
        
        population = new_population
    
    return best_solution

def local_search(solution, distances, total_cars):
    
    for _ in range(10):  # Número de iterações da busca local
        route1, route2 = random.sample(solution.deliveries, 2)
        if route1.cars and route2.cars:
            car1 = random.choice(route1.cars)
            car2 = random.choice(route2.cars)
            route1.cars.remove(car1)
            route2.cars.remove(car2)
            route1.cars.append(car2)
            route2.cars.append(car1)
    
    # Recalcular o fitness após a busca local
    solution.calculate_fitness(distances, total_cars)
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
        "total_cars_allocated": sum(len(delivery.cars) for delivery in solution.deliveries),
        "deliveries": [
            {
                "route_id": d.route_id,
                "carrier": d.carrier,
                "day": convert_to_serializable(d.day),
                "period": d.period,
                "dock": d.dock,
                "total_morphology": convert_to_serializable(d.total_morphology),
                "total_distance": convert_to_serializable(d.total_distance),
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
    logging.info(f"Número de morfologias: {len(morphologies)}")
    logging.info(f"Número de cidades com transportadoras: {len(carriers)}")
    logging.info(f"Número de distâncias: {len(distances)}")
    logging.info(f"Número de carros para entrega (após filtragem): {len(outbound_data)}")
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
    logging.info(f"Número mínimo estimado de rotas necessárias: {estimated_min_routes:.2f}")

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
    logging.info("Iniciando o algoritmo genético...")
    best_solution = genetic_algorithm(outbound_data, carriers, morphologies, distances, costs)
    
    if best_solution:
        total_cars = len(outbound_data)
        allocated_cars = sum(len(delivery.cars) for delivery in best_solution.deliveries)
        
        if allocated_cars < total_cars:
            print("AVISO: A solução não alocou todos os carros!")
    
    
        print("Melhor solução encontrada:")
        print(f"Fitness: {best_solution.fitness}")
        print(f"Número total de rotas: {len(best_solution.deliveries)}")
        print(f"Número total de carros alocados: {sum(len(delivery.cars) for delivery in best_solution.deliveries)}")
        for delivery in best_solution.deliveries:
            print(f"\nRota: {delivery.route_id}")
            print(f"Transportadora: {delivery.carrier}")
            print(f"Dia: {delivery.day}, Período: {delivery.period}, Doca: {delivery.dock}")
            print(f"Morfologia total: {delivery.total_morphology}")
            print(f"Distância total: {delivery.total_distance}")
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