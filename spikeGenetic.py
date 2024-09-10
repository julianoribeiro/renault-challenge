import pandas as pd
import numpy as np
import random
import logging
from collections import defaultdict
import json
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Classe Delivery
class Delivery:
    def __init__(self, route_id, vehicle, vehicle_id, day, period):
        self.route_id = route_id
        self.vehicle = vehicle
        self.vehicle_id = vehicle_id
        self.day = day
        self.period = period
        self.stops = []

    def add_stop(self, supplier, material, quantity_m3, quantity_ton, distance):
        self.stops.append({
            "supplier": supplier,
            "material": material,
            "quantity_m3": quantity_m3,
            "quantity_ton": quantity_ton,
            "distance": distance
        })

    @property
    def total_distance(self):
        return sum(stop['distance'] for stop in self.stops)

# Classe base ClausulaRestritiva
class ClausulaRestritiva:
    def validar(self, solucao):
        raise NotImplementedError("Método validar deve ser implementado na subclasse")

# Subclasse de ClausulaRestritiva para capacidade dos veículos
class CapacidadeVeiculoClausula(ClausulaRestritiva):
    def __init__(self, vehicles, max_trips_per_week):
        self.vehicles = vehicles
        self.max_trips_per_week = max_trips_per_week

    def validar(self, solucao):
        vehicle_usage = defaultdict(lambda: {'count': 0, 'm3': 0, 'ton': 0})
        for delivery in solucao.deliveries:
            vehicle_usage[delivery.vehicle]['count'] += 1
            for stop in delivery.stops:
                vehicle_usage[delivery.vehicle]['m3'] += stop['quantity_m3']
                vehicle_usage[delivery.vehicle]['ton'] += stop['quantity_ton']

        for vehicle, usage in vehicle_usage.items():
            vehicle_row = self.vehicles[self.vehicles['Veiculo'] == vehicle].iloc[0]
            max_trips = vehicle_row['Quantidade'] * self.max_trips_per_week[vehicle]
            if usage['count'] > max_trips:
                logging.debug(f"Falha na restrição de Capacidade: {vehicle} excedeu a quantidade permitida em {usage['count']} / {max_trips}")
                return False
            if usage['m3'] > vehicle_row['Capacidade_m3'] * usage['count']:
                logging.debug(f"Falha na restrição de Capacidade: {vehicle} excedeu a capacidade em m3.")
                return False
            if usage['ton'] > vehicle_row['Capacidade_ton'] * usage['count']:
                logging.debug(f"Falha na restrição de Capacidade: {vehicle} excedeu a capacidade em toneladas.")
                return False

        return True

# Subclasse de ClausulaRestritiva para quantidade de veículos disponíveis e cálculo de tempo
class QuantidadeVeiculoClausula(ClausulaRestritiva):
    def __init__(self, vehicles):
        self.vehicles = vehicles

    def validar(self, solucao):
        vehicle_schedule = defaultdict(list)
        for delivery in solucao.deliveries:
            route_time = calculate_route_time(delivery.total_distance)
            start_time = delivery.day * 2 + (0.5 if delivery.period == 'morning' else 1)
            end_time = start_time + route_time

            # Verificar sobreposição com rotas existentes
            for existing_start, existing_end in vehicle_schedule[delivery.vehicle]:
                if (start_time < existing_end and end_time > existing_start) or \
                   (existing_start < end_time and existing_end > start_time):
                    logging.debug(f"Falha na restrição de Quantidade: {delivery.vehicle} tem sobreposição de horários.")
                    return False

            vehicle_schedule[delivery.vehicle].append((start_time, end_time))

        for vehicle, schedules in vehicle_schedule.items():
            if len(schedules) > self.vehicles[self.vehicles['Veiculo'] == vehicle]['Quantidade'].values[0] * 10:
                logging.debug(f"Falha na restrição de Quantidade: {vehicle} excedeu o número máximo de rotas permitidas.")
                return False

        return True
    
# Classe base ClausulaLimitante
class ClausulaLimitante:
    def penalizar(self, solucao):
        raise NotImplementedError("Método penalizar deve ser implementado na subclasse")

# Subclasse de ClausulaLimitante
class PenalizacaoMuitasEntregasClausula(ClausulaLimitante):
    def penalizar(self, solucao):
        penalidade = 0
        entregas_por_periodo = defaultdict(int)
        for delivery in solucao.deliveries:
            key = (delivery.day, delivery.period)
            entregas_por_periodo[key] += 1
            if entregas_por_periodo[key] > 5:
                penalidade += 10  # Penalidade arbitrária para excesso de entregas
        return penalidade

# Classe Solucao
class Solucao:
    def __init__(self, deliveries=None, vehicles=None, max_trips_per_week=None):
        self.deliveries = deliveries if deliveries is not None else []
        self._vehicles = None
        self._max_trips_per_week = None
        self._fitness = float('inf')
        self._total_cost = float('inf')
        self._total_distance = float('inf')
        self.set_vehicles(vehicles, max_trips_per_week)

    def set_vehicles(self, vehicles, max_trips_per_week):
        if vehicles is not None and max_trips_per_week is not None:
            self._vehicles = vehicles
            self._max_trips_per_week = max_trips_per_week
            self.restricoes = [CapacidadeVeiculoClausula(vehicles, max_trips_per_week), QuantidadeVeiculoClausula(vehicles)]
            self.limitacoes = [PenalizacaoMuitasEntregasClausula()]
        else:
            raise ValueError("Vehicles e max_trips_per_week devem ser fornecidos")

    @property
    def fitness(self):
        return self._fitness

    @property
    def total_cost(self):
        return self._total_cost

    @property
    def total_distance(self):
        return self._total_distance

    def update_fitness(self, fitness, total_cost, total_distance):
        self._fitness = fitness
        self._total_cost = total_cost
        self._total_distance = total_distance

    def calcular_fitness(self, suppliers, costs, materials):
        if self._vehicles is None or self._max_trips_per_week is None:
            raise ValueError("Vehicles e max_trips_per_week não foram definidos para esta solução")

        for restricao in self.restricoes:
            if not restricao.validar(self):
                self.update_fitness(float('inf'), float('inf'), float('inf'))
                return self._fitness, self._total_cost, self._total_distance

        fitness, total_cost, total_distance = calculate_fitness(self, suppliers, costs, materials, self._vehicles)
        self.update_fitness(fitness, total_cost, total_distance)
        return self._fitness, self._total_cost, self._total_distance

    @classmethod
    def copy(cls, other):
        new_solution = cls(deliveries=other.deliveries.copy(), vehicles=other._vehicles, max_trips_per_week=other._max_trips_per_week)
        new_solution.update_fitness(other.fitness, other.total_cost, other.total_distance)
        return new_solution
    
def genetic_algorithm(materials, suppliers, vehicles, costs, max_trips_per_week, population_size=100, generations=100, elite_size=10):
    print("Iniciando algoritmo genético")
    densities = calculate_material_densities(materials)
    
    best_solution_global = None
    best_fitness_global = float('inf')
    generations_without_improvement = 0
    total_restarts = 0
    max_restarts = 40
    
    while total_restarts < max_restarts:
        population = generate_population(materials, suppliers, vehicles, densities, population_size, max_trips_per_week)
        if not population:
            print("Falha ao gerar população inicial. Tentando novamente.")
            total_restarts += 1
            continue
        
        print(f"População inicial gerada com {len(population)} soluções")
        
        for generation in range(generations):
            population_fitness = evaluate_population(population, suppliers, costs, materials)
            
            if not population_fitness:
                print("População vazia após avaliação. Reiniciando.")
                break
            
            current_best = min(population_fitness, key=lambda x: x[1])
            if current_best[1] < best_fitness_global:
                best_solution_global = Solucao.copy(current_best[0])
                best_fitness_global = current_best[1]
                generations_without_improvement = 0
                print(f"Nova melhor solução global encontrada na geração {generation + 1}")
                print(f"Fitness = {best_fitness_global:.2f}, Custo = {best_solution_global.total_cost:.2f}, Distância = {best_solution_global.total_distance:.2f}")
                print_solution(best_solution_global)
            else:
                generations_without_improvement += 1
            
            # Imprimir o progresso a cada geração
            if best_solution_global:
                print(f"Geração {generation + 1}: Melhor Fitness Global = {best_fitness_global:.2f}, Custo = {best_solution_global.total_cost:.2f}, Distância = {best_solution_global.total_distance:.2f}")
            else:
                print(f"Geração {generation + 1}: Ainda não foi encontrada uma solução válida")
            
            if generations_without_improvement >= 50:
                print(f"Estagnação detectada após {generations_without_improvement} gerações. Reiniciando população.")
                break
            
            elite = select_elite(population_fitness, elite_size)
            selected = tournament_selection(population_fitness, population_size - elite_size)
            
            new_population = crossover_and_mutate(selected, materials, suppliers, vehicles, densities, max_trips_per_week)
            new_population.extend(elite)
            
            # Adicionar diversidade
            if generation % 10 == 0:
                new_individuals = generate_population(materials, suppliers, vehicles, densities, population_size // 5, max_trips_per_week)
                if new_individuals:
                    new_population = new_population[:-len(new_individuals)] + new_individuals
            
            population = new_population
        
        total_restarts += 1
        print(f"Reinício {total_restarts} concluído. Melhor fitness global: {best_fitness_global:.2f}")
        
    if best_solution_global is not None and best_solution_global.fitness < float('inf'):
        print(f"Melhor solução global encontrada após {total_restarts} reinícios")
        return best_solution_global
    else:
        print("Não foi possível encontrar uma solução válida após múltiplos reinícios")
        return None
    

def calculate_fitness(solution, suppliers, costs, materials, vehicles):
    total_cost = 0
    total_distance = 0
    penalty = 0

    delivered_materials = defaultdict(lambda: {'m3': 0, 'ton': 0})

    for delivery in solution.deliveries:
        for stop in delivery.stops:
            # Verificar se o fornecedor pode fornecer o material
            if suppliers.loc[suppliers['Fornecedor'] == stop['supplier'], stop['material']].values[0] != 'X':
                penalty += 500  # Penalidade ajustada
                logging.debug(f"Penalidade aplicada por fornecedor inválido: {penalty}")
                continue

            # Verificar capacidade do veículo
            vehicle_row = vehicles[vehicles['Veiculo'] == delivery.vehicle].iloc[0]
            if stop['quantity_m3'] > vehicle_row['Capacidade_m3'] or stop['quantity_ton'] > vehicle_row['Capacidade_ton']:
                penalty += 500 * (max(stop['quantity_m3'] - vehicle_row['Capacidade_m3'], 0) + 
                                  max(stop['quantity_ton'] - vehicle_row['Capacidade_ton'], 0))
                logging.debug(f"Penalidade aplicada por capacidade do veículo: {penalty}")

            # Calcular custo e distância
            try:
                cost_per_km = costs.loc[costs['Km'] >= stop['distance']].iloc[0][delivery.vehicle]
            except IndexError:
                logging.error(f"Erro ao calcular custo: não encontrado custo para veículo {delivery.vehicle} na distância {stop['distance']}km")
                return float('inf'), float('inf'), float('inf')
            
            delivery_cost = stop['distance'] * cost_per_km
            total_cost += delivery_cost
            total_distance += stop['distance']

            # Atualizar materiais entregues
            delivered_materials[stop['material']]['m3'] += stop['quantity_m3']
            delivered_materials[stop['material']]['ton'] += stop['quantity_ton']

    # Verificar se todos os materiais foram entregues
    for material, quantities in delivered_materials.items():
        required_m3 = materials.loc[materials['Material'] == material, 'Quantidade_m3'].values[0]
        required_ton = materials.loc[materials['Material'] == material, 'Quantidade_ton'].values[0]
        
        deviation_m3 = abs(quantities['m3'] - required_m3) / required_m3 if required_m3 > 0 else 0
        deviation_ton = abs(quantities['ton'] - required_ton) / required_ton if required_ton > 0 else 0
        
        penalty += 50 * (deviation_m3 + deviation_ton)  # Penalidade ajustada
        logging.debug(f"Penalidade aplicada por desvio: {penalty}")

    # Adicionar pequena penalidade para número de entregas
    num_deliveries = len(solution.deliveries)
    penalty += num_deliveries * 0.5  # Penalidade menor
    logging.debug(f"Penalidade total acumulada: {penalty}")

    # Ajustando os pesos conforme solicitado
    weighted_fitness = (total_cost * 0.6) + (total_distance * 0.2) + (penalty * 0.2)
    logging.debug(f"Fitness calculado: {weighted_fitness}, Custo total: {total_cost}, Distância total: {total_distance}")

    return weighted_fitness, total_cost, total_distance

def evaluate_population(population, suppliers, costs, materials):
    return [
        (solution, *solution.calcular_fitness(suppliers, costs, materials))
        for solution in population
    ]

def select_elite(population_fitness, elite_size):
    return [x[0] for x in sorted(population_fitness, key=lambda x: x[1])[:elite_size]]

def tournament_selection(population_fitness, num_select, tournament_size=3):
    selected = []
    for _ in range(num_select):
        tournament = random.sample(population_fitness, tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def inject_diversity(population, num_new, materials, suppliers, vehicles):
    new_individuals = generate_population(materials, suppliers, vehicles, num_new)
    population[-num_new:] = new_individuals
    return population
def adaptive_crossover(parent1, parent2, current_gen, max_gen):
    if random.random() < 0.5 + (0.3 * current_gen / max_gen):
        return uniform_crossover(parent1, parent2)
    else:
        return single_point_crossover(parent1, parent2)

def uniform_crossover(parent1, parent2):
    child1_deliveries = []
    child2_deliveries = []
    for d1, d2 in zip(parent1.deliveries, parent2.deliveries):
        if random.random() < 0.5:
            child1_deliveries.append(d1)
            child2_deliveries.append(d2)
        else:
            child1_deliveries.append(d2)
            child2_deliveries.append(d1)
    return Solucao(child1_deliveries), Solucao(child2_deliveries)

def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1.deliveries), len(parent2.deliveries)) - 1)
    child1_deliveries = parent1.deliveries[:crossover_point] + parent2.deliveries[crossover_point:]
    child2_deliveries = parent2.deliveries[:crossover_point] + parent1.deliveries[crossover_point:]
    return Solucao(child1_deliveries), Solucao(child2_deliveries)

def adaptive_mutation(population, current_gen, max_gen, materials, suppliers, vehicles):
    mutation_rate = 0.1 * (1 - current_gen / max_gen)
    return [mutate_solution(individual, mutation_rate, materials, suppliers, vehicles) for individual in population]

def mutate_solution(solution, mutation_rate, materials, suppliers, vehicles):
    if random.random() < mutation_rate:
        if solution.deliveries:
            delivery = random.choice(solution.deliveries)
            mutation_type = random.choice(['supplier', 'vehicle', 'quantity'])
            
            if mutation_type == 'supplier':
                available_suppliers = suppliers[suppliers[delivery.material] == 'X']
                if not available_suppliers.empty:
                    new_supplier = available_suppliers.sample().iloc[0]['Fornecedor']
                    delivery.supplier = new_supplier
                    delivery.distance = suppliers.loc[suppliers['Fornecedor'] == new_supplier, 'DistanciaEntrega_km'].values[0]
            
            elif mutation_type == 'vehicle':
                available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= delivery.quantity_m3) & 
                                              (vehicles['Capacidade_ton'] >= delivery.quantity_ton)]
                if not available_vehicles.empty:
                    new_vehicle = available_vehicles.sample().iloc[0]['Veiculo']
                    delivery.vehicle = new_vehicle
            
            elif mutation_type == 'quantity':
                material_row = materials[materials['Material'] == delivery.material].iloc[0]
                vehicle_row = vehicles[vehicles['Veiculo'] == delivery.vehicle].iloc[0]
                max_m3 = min(material_row['Quantidade_m3'], vehicle_row['Capacidade_m3'])
                max_ton = min(material_row['Quantidade_ton'], vehicle_row['Capacidade_ton'])
                
                new_m3 = random.uniform(0, max_m3)
                new_ton = new_m3 * (material_row['Quantidade_ton'] / material_row['Quantidade_m3'])
                
                if new_ton > max_ton:
                    new_ton = max_ton
                    new_m3 = new_ton / (material_row['Quantidade_ton'] / material_row['Quantidade_m3'])
                
                delivery.quantity_m3 = new_m3
                delivery.quantity_ton = new_ton
    
    return solution

def crossover_and_mutate(selected, materials, suppliers, vehicles, densities, max_trips_per_week):
    new_population = []
    crossover_rate = 0.8
    mutation_rate = 0.2

    for i in range(0, len(selected), 2):
        parent1 = selected[i]
        parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

        if random.random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2, materials)
            logging.debug("Crossover realizado")
        else:
            child1, child2 = Solucao.copy(parent1), Solucao.copy(parent2)
            logging.debug("Cópia direta dos pais")

        child1 = mutate(child1, mutation_rate, materials, suppliers, vehicles)
        child2 = mutate(child2, mutation_rate, materials, suppliers, vehicles)
        logging.debug("Mutação aplicada")

        child1 = repair_solution(child1, materials, suppliers, vehicles, densities, max_trips_per_week)
        child2 = repair_solution(child2, materials, suppliers, vehicles, densities, max_trips_per_week)
        logging.debug("Soluções reparadas")

        if is_valid_solution(child1, materials):
            new_population.append(child1)
            logging.debug("Child 1 adicionado à nova população")
        else:
            logging.warning("Child 1 inválido após crossover, mutação e reparo")

        if is_valid_solution(child2, materials):
            new_population.append(child2)
            logging.debug("Child 2 adicionado à nova população")
        else:
            logging.warning("Child 2 inválido após crossover, mutação e reparo")

    # Se a nova população for muito pequena, adicione soluções aleatórias
    while len(new_population) < len(selected):
        new_solution = gerar_solucao_inicial(materials, suppliers, vehicles, densities, max_trips_per_week)
        if new_solution is not None and is_valid_solution(new_solution, materials):
            new_population.append(new_solution)
            logging.debug("Nova solução aleatória adicionada à população")

    logging.info(f"Nova população gerada com {len(new_population)} soluções válidas")
    return new_population

def adaptive_crossover(parent1, parent2, current_gen, max_gen):
    crossover_prob = 0.5 + (0.3 * current_gen / max_gen)
    if random.random() < crossover_prob:
        return uniform_crossover(parent1, parent2)
    else:
        return single_point_crossover(parent1, parent2)

def adaptive_mutation(population, current_gen, max_gen, materials, suppliers, vehicles):
    mutation_rate = 0.1 * (1 - current_gen / max_gen)
    new_population = []
    for individual in population:
        if random.random() < mutation_rate:
            new_population.append(mutate_solution(individual, mutation_rate, materials, suppliers, vehicles))
        else:
            new_population.append(individual)
    return new_population


def crossover(parent1, parent2, materials):
    child1_deliveries = []
    child2_deliveries = []
    
    for material in materials['Material']:
        p1_deliveries = [d for d in parent1.deliveries if any(stop['material'] == material for stop in d.stops)]
        p2_deliveries = [d for d in parent2.deliveries if any(stop['material'] == material for stop in d.stops)]
        
        if random.random() < 0.5:
            child1_deliveries.extend(p1_deliveries)
            child2_deliveries.extend(p2_deliveries)
        else:
            child1_deliveries.extend(p2_deliveries)
            child2_deliveries.extend(p1_deliveries)
    
    child1 = Solucao.copy(parent1)
    child2 = Solucao.copy(parent2)
    child1.deliveries = child1_deliveries
    child2.deliveries = child2_deliveries
    
    return child1, child2

def mutate(solution, mutation_rate, materials, suppliers, vehicles):
    vehicle_counters = defaultdict(int)
    for delivery in solution.deliveries:
        vehicle_id_parts = delivery.vehicle_id.split('_')
        if len(vehicle_id_parts) == 2:
            vehicle, count = vehicle_id_parts
            vehicle_counters[vehicle] = max(vehicle_counters[vehicle], int(count))

    for delivery in solution.deliveries:
        if random.random() < mutation_rate:
            mutation_type = random.choice(['supplier', 'vehicle', 'quantity', 'remove', 'add'])
            
            if mutation_type == 'supplier':
                for stop in delivery.stops:
                    if stop['material'] not in suppliers.columns:
                        logging.error(f"Material {stop['material']} não encontrado no DataFrame de fornecedores")
                        continue
                    
                    available_suppliers = suppliers[suppliers[stop['material']] == 'X']
                    if not available_suppliers.empty:
                        new_supplier = random.choice(available_suppliers['Fornecedor'].tolist())
                        new_distance = suppliers.loc[suppliers['Fornecedor'] == new_supplier, 'DistanciaEntrega_km'].values[0]
                        stop['supplier'] = new_supplier
                        stop['distance'] = new_distance
                    else:
                        logging.warning(f"Nenhum fornecedor disponível para o material {stop['material']}")

            
            elif mutation_type == 'vehicle':
                total_m3 = sum(stop['quantity_m3'] for stop in delivery.stops)
                total_ton = sum(stop['quantity_ton'] for stop in delivery.stops)
                available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= total_m3) & 
                                            (vehicles['Capacidade_ton'] >= total_ton)]
                if not available_vehicles.empty:
                    new_vehicle = random.choice(available_vehicles['Veiculo'].tolist())
                    old_vehicle = delivery.vehicle
                    delivery.vehicle = new_vehicle
                    vehicle_counters[new_vehicle] += 1
                    delivery.vehicle_id = f"{new_vehicle}_{vehicle_counters[new_vehicle]}"
                    logging.debug(f"Veículo alterado de {old_vehicle} para {new_vehicle} (Novo ID: {delivery.vehicle_id})")

            
            elif mutation_type == 'quantity':
                for stop in delivery.stops:
                    material_row = materials[materials['Material'] == stop['material']].iloc[0]
                    vehicle_row = vehicles[vehicles['Veiculo'] == delivery.vehicle].iloc[0]
                    max_m3 = min(material_row['Quantidade_m3'], vehicle_row['Capacidade_m3'])
                    max_ton = min(material_row['Quantidade_ton'], vehicle_row['Capacidade_ton'])
                    
                    change_factor = random.uniform(0.8, 1.2)
                    new_m3 = stop['quantity_m3'] * change_factor
                    new_ton = new_m3 * (material_row['Quantidade_ton'] / material_row['Quantidade_m3'])
                    
                    if new_ton > max_ton:
                        new_ton = max_ton
                        new_m3 = new_ton / (material_row['Quantidade_ton'] / material_row['Quantidade_m3'])
                    
                    stop['quantity_m3'] = min(new_m3, max_m3)
                    stop['quantity_ton'] = min(new_ton, max_ton)
                    logging.debug(f"Quantidade alterada para {stop['quantity_m3']:.2f} m³ / {stop['quantity_ton']:.2f} ton")
            
            elif mutation_type == 'remove':
                solution.deliveries.remove(delivery)
                logging.debug(f"Entrega removida: {delivery.vehicle_id}")
                break  # Saímos do loop para evitar problemas com a iteração
            
            elif mutation_type == 'add':
                new_delivery = create_random_delivery(materials, suppliers, vehicles, vehicle_counters)
                if new_delivery:
                    solution.deliveries.append(new_delivery)
                    logging.debug(f"Nova entrega adicionada: {new_delivery.vehicle_id}")
                else:
                    logging.warning("Não foi possível criar uma nova entrega aleatória")

    return adjust_quantities(solution, materials)

def create_random_delivery(materials, suppliers, vehicles, vehicle_counters):
    material = random.choice(materials['Material'].tolist())
    available_suppliers = suppliers[suppliers[material] == 'X']
    
    if available_suppliers.empty:
        logging.warning(f"Nenhum fornecedor disponível para o material {material}")
        return None
    
    supplier = random.choice(available_suppliers['Fornecedor'].tolist())
    vehicle = random.choice(vehicles['Veiculo'].tolist())
    
    material_row = materials[materials['Material'] == material].iloc[0]
    vehicle_row = vehicles[vehicles['Veiculo'] == vehicle].iloc[0]
    
    max_quantity_m3 = min(material_row['Quantidade_m3'], vehicle_row['Capacidade_m3'])
    max_quantity_ton = min(material_row['Quantidade_ton'], vehicle_row['Capacidade_ton'])
    
    quantity_m3 = random.uniform(0, max_quantity_m3)
    quantity_ton = min(quantity_m3 * (material_row['Quantidade_ton'] / material_row['Quantidade_m3']), max_quantity_ton)
    
    # Ajuste final para garantir que a quantidade não exceda a capacidade do veículo
    if quantity_ton > vehicle_row['Capacidade_ton']:
        quantity_ton = vehicle_row['Capacidade_ton']
        quantity_m3 = min(vehicle_row['Capacidade_m3'], quantity_ton / (material_row['Quantidade_ton'] / material_row['Quantidade_m3']))

    day = random.randint(1, 5)
    period = random.choice(['morning', 'afternoon'])
    distance = suppliers.loc[suppliers['Fornecedor'] == supplier, 'DistanciaEntrega_km'].values[0]
    
    # Gerar um vehicle_id único
    vehicle_counters[vehicle] += 1
    vehicle_id = f"{vehicle}_{vehicle_counters[vehicle]}"
    
    new_delivery = Delivery(f"Route{vehicle_counters[vehicle]}", vehicle, vehicle_id, day, period)
    new_delivery.add_stop(supplier, material, quantity_m3, quantity_ton, distance)
    
    return new_delivery


def adjust_quantities(solution, materials):
    for material in materials['Material']:
        delivered = sum(stop['quantity_m3'] for delivery in solution.deliveries for stop in delivery.stops if stop['material'] == material)
        required = materials.loc[materials['Material'] == material, 'Quantidade_m3'].values[0]
        
        if delivered > 0:
            scale_factor = required / delivered
            for delivery in solution.deliveries:
                for stop in delivery.stops:
                    if stop['material'] == material:
                        stop['quantity_m3'] *= scale_factor
                        stop['quantity_ton'] *= scale_factor

    return solution

def adjust_material_quantity(deliveries, material, required_m3, required_ton):
    material_deliveries = [d for d in deliveries if d.material == material]
    
    if not material_deliveries:
        return deliveries
    
    total_m3 = sum(d.quantity_m3 for d in material_deliveries)
    
    ton_per_m3 = required_ton / required_m3
    
    for delivery in material_deliveries:
        if total_m3 > 0:
            scale_factor = required_m3 / total_m3
            delivery.quantity_m3 *= scale_factor
            delivery.quantity_ton = delivery.quantity_m3 * ton_per_m3
        else:
            delivery.quantity_m3 = 0
            delivery.quantity_ton = 0
    
    return deliveries

def consolidate_deliveries(deliveries):
    consolidated = {}
    for delivery in deliveries:
        key = (delivery.supplier, delivery.material, delivery.vehicle)
        if key not in consolidated:
            consolidated[key] = delivery
        else:
            existing = consolidated[key]
            existing.quantity_m3 += delivery.quantity_m3
            existing.quantity_ton += delivery.quantity_ton
    
    return list(consolidated.values())

def check_vehicle_utilization(solution, vehicles):
    vehicle_usage = defaultdict(int)
    for delivery in solution.deliveries:
        vehicle_usage[delivery.vehicle] += 1
    
    for _, vehicle_row in vehicles.iterrows():
        vehicle = vehicle_row['Veiculo']
        available = vehicle_row['Quantidade']
        used = vehicle_usage[vehicle]
        utilization = (used / available) * 100 if available > 0 else 0
        logging.info(f"Veículo {vehicle}: Usado {used}/{available} ({utilization:.2f}%)")
    
    total_available = vehicles['Quantidade'].sum()
    total_used = sum(vehicle_usage.values())
    total_utilization = (total_used / total_available) * 100 if total_available > 0 else 0
    logging.info(f"Utilização total de veículos: {total_used}/{total_available} ({total_utilization:.2f}%)")


def generate_population(materials, suppliers, vehicles, densities, population_size, max_trips_per_week):
    population = []
    attempts = 0
    max_attempts = population_size * 30  # Aumentar o número máximo de tentativas

    while len(population) < population_size and attempts < max_attempts:
        solution = gerar_solucao_inicial(materials, suppliers, vehicles, densities, max_trips_per_week)
        
        if solution is not None:
            solution = repair_solution(solution, materials, suppliers, vehicles, densities, max_trips_per_week)
            is_valid = is_valid_solution(solution, materials)
            if is_valid:
                population.append(solution)
                logging.info(f"Indivíduo válido {len(population)} adicionado à população")
            else:
                logging.warning(f"Solução gerada é inválida mesmo após reparo. Tentativa {attempts + 1}")
        else:
            logging.warning(f"Falha ao gerar solução. Tentativa {attempts + 1}")
        
        attempts += 1
        if attempts % 10 == 0:
            logging.info(f"Tentativa {attempts}: {len(population)} soluções válidas geradas")

    if len(population) < population_size:
        logging.warning(f"Gerados apenas {len(population)} indivíduos válidos após {attempts} tentativas.")
    
    if not population:
        logging.error("Não foi possível gerar nenhuma solução válida. Verifique os dados de entrada.")
    else:
        logging.info(f"População inicial completa com {len(population)} soluções válidas.")

    return population


def calculate_material_densities(materials):
    densities = {}
    for _, row in materials.iterrows():
        material = row['Material']
        density = row['Quantidade_ton'] / row['Quantidade_m3']
        densities[material] = density
    return densities

LOADING_TIME = 0.5  # Tempo de carga (meio período)
UNLOADING_TIME = 0.5  # Tempo de descarga (meio período)
TRANSPORT_TIME_PER_300KM = 0.5  # Tempo de transporte para cada 300km (meio período)

def calculate_route_time(distance):
    transport_time = (distance / 300) * TRANSPORT_TIME_PER_300KM
    return LOADING_TIME + transport_time + UNLOADING_TIME

def allocate_direct_load(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries, materials, densities, max_trips_per_week, vehicle_usage):
    if remaining_m3 <= 1e-6:
        return []

    new_deliveries = []
    density = densities[material]

    available_suppliers = suppliers[suppliers[material] == 'X']['Fornecedor'].tolist()
    if not available_suppliers:
        logging.warning(f"Nenhum fornecedor disponível para o material {material}")
        return new_deliveries

    sorted_vehicles = vehicles.sort_values(['Capacidade_m3', 'Capacidade_ton'], ascending=False)
    vehicle_counters = defaultdict(int)

    while remaining_m3 > 1e-6:
        allocated = False
        for vehicle_row in sorted_vehicles.itertuples():
            max_trips = vehicle_row.Quantidade * max_trips_per_week[vehicle_row.Veiculo]
            if len(vehicle_usage[vehicle_row.Veiculo]) >= max_trips:
                continue

            load_m3 = min(remaining_m3, vehicle_row.Capacidade_m3)
            load_ton = load_m3 * density

            if load_m3 > 1e-6 and load_ton <= vehicle_row.Capacidade_ton:
                supplier = random.choice(available_suppliers)
                distance = suppliers.loc[suppliers['Fornecedor'] == supplier, 'DistanciaEntrega_km'].values[0]
                route_time = calculate_route_time(distance)

                available_time_slot = False
                for day in range(1, 6):
                    for period in ['morning', 'afternoon']:
                        start_time = day * 2 + (0.5 if period == 'morning' else 1)
                        end_time = start_time + route_time

                        existing_times = [(slot[0], slot[0] + calculate_route_time(slot[2])) 
                                          for slot in vehicle_usage[vehicle_row.Veiculo]]

                        if all(not (start_time < exist_end and end_time > exist_start)
                               for exist_start, exist_end in existing_times):
                            available_time_slot = True
                            break
                    if available_time_slot:
                        break

                if not available_time_slot:
                    continue

                vehicle_counters[vehicle_row.Veiculo] += 1
                vehicle_id = f"{vehicle_row.Veiculo}_{vehicle_counters[vehicle_row.Veiculo]}"
                
                new_delivery = Delivery(f"Route{vehicle_counters[vehicle_row.Veiculo]}", vehicle_row.Veiculo, vehicle_id, day, period)
                new_delivery.add_stop(supplier, material, load_m3, load_ton, distance)
                new_deliveries.append(new_delivery)
                remaining_m3 -= load_m3
                remaining_ton -= load_ton
                vehicle_usage[vehicle_row.Veiculo].append((start_time, period, distance))
                
                logging.debug(f"Alocado {load_m3:.2f} m³ / {load_ton:.2f} ton de {material} para {supplier} usando {vehicle_row.Veiculo} (ID: {vehicle_id})")
                allocated = True
                break

        if not allocated:
            logging.warning(f"Não foi possível alocar mais carga para {material}. Restante: {remaining_m3:.2f} m³ / {remaining_ton:.2f} ton")
            break

    return new_deliveries



def allocate_load_with_distance_check(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries, materials, densities):
    if remaining_m3 <= 1e-6 or remaining_ton <= 1e-6:
        return []

    new_deliveries = []
    density = densities[material]

    available_suppliers = suppliers[suppliers[material] == 'X'].sort_values('DistanciaEntrega_km')
    if available_suppliers.empty:
        logging.warning(f"Nenhum fornecedor disponível para o material {material}")
        return new_deliveries

    vehicle_usage = defaultdict(list)
    vehicle_counters = defaultdict(int)
    for delivery in existing_deliveries:
        start_time = delivery.day * 2 + (0.5 if delivery.period == 'morning' else 1)
        end_time = start_time + calculate_route_time(delivery.total_distance)
        vehicle_usage[delivery.vehicle].append((start_time, end_time))
        vehicle_counters[delivery.vehicle] = max(vehicle_counters[delivery.vehicle], int(delivery.vehicle_id.split('_')[1]))

    max_routes_per_vehicle = 10

    for _ in range(max_routes_per_vehicle):
        if remaining_m3 <= 1e-6 or remaining_ton <= 1e-6:
            break

        available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= remaining_m3) & 
                                      (vehicles['Capacidade_ton'] >= remaining_ton) &
                                      (vehicles['Veiculo'].apply(lambda v: len(vehicle_usage[v]) < vehicles.loc[vehicles['Veiculo'] == v, 'Quantidade'].values[0] * max_routes_per_vehicle))]
        
        if available_vehicles.empty:
            break

        vehicle = random.choice(available_vehicles['Veiculo'].tolist())
        vehicle_row = available_vehicles[available_vehicles['Veiculo'] == vehicle].iloc[0]

        vehicle_capacity_m3 = vehicle_row['Capacidade_m3']
        vehicle_capacity_ton = vehicle_row['Capacidade_ton']
        
        route = []
        total_distance = 0
        total_load_m3 = 0
        total_load_ton = 0
        total_stops = 0

        for _, supplier_row in available_suppliers.iterrows():
            if total_stops > 0:
                total_distance += suppliers.loc[suppliers['Fornecedor'] == route[-1][0], f'Fornecedor{supplier_row.name}'].values[0]

            direct_distance = supplier_row['DistanciaEntrega_km']
            if total_distance + direct_distance > direct_distance * 2:
                break

            available_m3 = min(remaining_m3, vehicle_capacity_m3 - total_load_m3)
            available_ton = min(remaining_ton, vehicle_capacity_ton - total_load_ton)

            if available_m3 <= 1e-6 or available_ton <= 1e-6:
                continue

            route.append((supplier_row['Fornecedor'], available_m3, available_ton, direct_distance))
            total_load_m3 += available_m3
            total_load_ton += available_ton
            total_distance += direct_distance
            total_stops += 1

            remaining_m3 -= available_m3
            remaining_ton -= available_ton

            if remaining_m3 <= 1e-6 or remaining_ton <= 1e-6:
                break

        if route:
            route_time = calculate_route_time(total_distance) + (total_stops - 1) * 0.5  # Adiciona meio período para cada parada adicional

            # Encontrar um slot de tempo disponível para a rota
            available_time_slot = False
            for day in range(1, 6):
                for period in ['morning', 'afternoon']:
                    start_time = day * 2 + (0.5 if period == 'morning' else 1)
                    end_time = start_time + route_time

                    if all(not (start_time < existing_end and end_time > existing_start)
                        for existing_start, existing_end in vehicle_usage[vehicle]):
                        available_time_slot = True
                        break
                if available_time_slot:
                    break

            if available_time_slot:
                vehicle_counters[vehicle] += 1
                vehicle_id = f"{vehicle}_{vehicle_counters[vehicle]}"
                new_delivery = Delivery(f"Route{vehicle_counters[vehicle]}", vehicle, vehicle_id, day, period)
                for supplier, load_m3, load_ton, distance in route:
                    new_delivery.add_stop(supplier, material, load_m3, load_ton, distance)
                new_deliveries.append(new_delivery)
                vehicle_usage[vehicle].append((start_time, end_time))
                logging.debug(f"Alocado {total_load_m3:.2f} m³ / {total_load_ton:.2f} ton de {material} usando {vehicle} (ID: {vehicle_id}) com múltiplos fornecedores")
            else:
                logging.debug(f"Não foi possível encontrar um slot de tempo para a rota com múltiplos fornecedores para {vehicle}")

    return new_deliveries



def gerar_solucao_inicial(materials, suppliers, vehicles, densities, max_trips_per_week):
    deliveries = []
    route_counter = 1

    for _, material_row in materials.iterrows():
        material = material_row['Material']
        remaining_m3 = material_row['Quantidade_m3']
        remaining_ton = material_row['Quantidade_ton']

        while remaining_m3 > 0 and remaining_ton > 0:
            vehicle = random.choice(vehicles['Veiculo'].tolist())
            vehicle_row = vehicles[vehicles['Veiculo'] == vehicle].iloc[0]
            day = random.randint(1, 5)
            period = random.choice(['morning', 'afternoon'])

            route = Delivery(f"Route{route_counter}", vehicle, f"{vehicle}_{route_counter}", day, period)
 
            route_m3 = 0
            route_ton = 0

            available_suppliers = suppliers[suppliers[material] == 'X']
            supplier_row = random.choice(available_suppliers.index)
            supplier = suppliers.loc[supplier_row, 'Fornecedor']
            distance = suppliers.loc[supplier_row, 'DistanciaEntrega_km']
            
            load_m3 = min(remaining_m3, vehicle_row['Capacidade_m3'])
            load_ton = min(remaining_ton, vehicle_row['Capacidade_ton'])

            if load_m3 > 0 and load_ton > 0:
                route.add_stop(supplier, material, load_m3, load_ton, distance)
                remaining_m3 -= load_m3
                remaining_ton -= load_ton

            if route.stops:
                deliveries.append(route)
                route_counter += 1

    return Solucao(deliveries, vehicles, max_trips_per_week)


def allocate_load(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries, materials, densities, max_trips_per_week):
    vehicle_usage = defaultdict(list)
    all_deliveries = []

    # Primeira tentativa: allocate_load_with_distance_check
    distance_check_deliveries = allocate_load_with_distance_check(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries, materials, densities)
    all_deliveries.extend(distance_check_deliveries)

    remaining_m3 -= sum(sum(stop['quantity_m3'] for stop in d.stops) for d in distance_check_deliveries)
    remaining_ton -= sum(sum(stop['quantity_ton'] for stop in d.stops) for d in distance_check_deliveries)

    # Se ainda houver carga restante, tenta alocação direta
    if remaining_m3 > 1e-6 and remaining_ton > 1e-6:
        direct_deliveries = allocate_direct_load(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries + all_deliveries, materials, densities, max_trips_per_week, vehicle_usage)
        all_deliveries.extend(direct_deliveries)

    # Ajuste final das quantidades alocadas para respeitar a capacidade do veículo
    for delivery in all_deliveries:
        vehicle_row = vehicles[vehicles['Veiculo'] == delivery.vehicle].iloc[0]
        max_m3 = vehicle_row['Capacidade_m3']
        max_ton = vehicle_row['Capacidade_ton']

        total_m3 = sum(stop['quantity_m3'] for stop in delivery.stops)
        total_ton = sum(stop['quantity_ton'] for stop in delivery.stops)

        if total_m3 > max_m3:
            scale_factor = max_m3 / total_m3
            for stop in delivery.stops:
                stop['quantity_m3'] *= scale_factor
                stop['quantity_ton'] *= scale_factor
            logging.debug(f"Ajustando quantidade m³ de {delivery.stops[0]['material']} de {total_m3:.2f} para {max_m3:.2f} devido à capacidade do veículo {delivery.vehicle}.")

        if total_ton > max_ton:
            scale_factor = max_ton / total_ton
            for stop in delivery.stops:
                stop['quantity_m3'] *= scale_factor
                stop['quantity_ton'] *= scale_factor
            logging.debug(f"Ajustando quantidade ton de {delivery.stops[0]['material']} de {total_ton:.2f} para {max_ton:.2f} devido à capacidade do veículo {delivery.vehicle}.")

    return all_deliveries

def log_solution_stats(solution, total_allocated_m3, total_allocated_ton, total_required_m3, total_required_ton, vehicles):
    logging.info(f"Solução inicial gerada com {len(solution.deliveries)} entregas")
    logging.info(f"Total alocado: {total_allocated_m3:.2f} m³ / {total_allocated_ton:.2f} ton")
    logging.info(f"Total requerido: {total_required_m3:.2f} m³ / {total_required_ton:.2f} ton")
    logging.info(f"Percentual alocado: {(total_allocated_m3/total_required_m3)*100:.2f}% (m³), {(total_allocated_ton/total_required_ton)*100:.2f}% (ton)")

    vehicle_usage = defaultdict(int)
    for delivery in solution.deliveries:
        vehicle_usage[delivery.vehicle] += 1
    
    for vehicle, count in vehicle_usage.items():
        vehicle_capacity = vehicles.loc[vehicles['Veiculo'] == vehicle, 'Quantidade'].values[0]
        utilization = (count / (vehicle_capacity * 10)) * 100  # Assumindo 10 rotas máximas por veículo
        logging.info(f"Utilização do veículo {vehicle}: {count} rotas ({utilization:.2f}%)")


def is_valid_solution(solution, materials):
    if not solution.deliveries:
        logging.warning("Solução inválida: sem entregas")
        return False
    
    delivered_materials = defaultdict(lambda: {'m3': 0, 'ton': 0})
    
    for delivery in solution.deliveries:
        for stop in delivery.stops:
            delivered_materials[stop['material']]['m3'] += stop['quantity_m3']
            delivered_materials[stop['material']]['ton'] += stop['quantity_ton']
    
    total_deviation = 0
    for material, quantities in delivered_materials.items():
        required_m3 = materials.loc[materials['Material'] == material, 'Quantidade_m3'].values[0]
        required_ton = materials.loc[materials['Material'] == material, 'Quantidade_ton'].values[0]
        
        deviation_m3 = abs(quantities['m3'] - required_m3) / required_m3 if required_m3 > 0 else 0
        deviation_ton = abs(quantities['ton'] - required_ton) / required_ton if required_ton > 0 else 0
        
        total_deviation += deviation_m3 + deviation_ton
        
        logging.info(f"Material {material}: Entregue {quantities['m3']:.2f} m³ / {quantities['ton']:.2f} ton, "
                     f"Requerido {required_m3:.2f} m³ / {required_ton:.2f} ton, "
                     f"Desvio: {deviation_m3:.2%} (m³), {deviation_ton:.2%} (ton)")
    
    average_deviation = total_deviation / (2 * len(materials))
    is_valid = average_deviation <= 0.3  # Aumentando a tolerância para 30%
    
    if is_valid:
        logging.info(f"Solução válida encontrada com desvio médio de {average_deviation:.2%}")
    else:
        logging.warning(f"Solução inválida com desvio médio de {average_deviation:.2%}")
    
    return is_valid

def check_time_availability(vehicle, day, period, existing_deliveries):
    for existing_delivery in existing_deliveries:
        if isinstance(existing_delivery, tuple) and len(existing_delivery) >= 2:
            existing_day, existing_period = existing_delivery[:2]
            if existing_day == day and existing_period == period:
                return False
        elif isinstance(existing_delivery, Delivery):
            if existing_delivery.day == day and existing_delivery.period == period:
                return False
    return True

def allocate_load_with_time_check(material, remaining_m3, remaining_ton, suppliers, vehicles, existing_deliveries, materials, densities):
    if remaining_m3 <= 1e-6 or remaining_ton <= 1e-6:
        return []

    new_deliveries = []
    density = densities[material]

    available_suppliers = suppliers[suppliers[material] == 'X']
    if available_suppliers.empty:
        logging.warning(f"Nenhum fornecedor disponível para o material {material}")
        return new_deliveries

    vehicle_usage = defaultdict(list)
    for delivery in existing_deliveries:
        vehicle_usage[delivery.vehicle].append((delivery.day, delivery.period))

    max_routes_per_vehicle = 10

    for _ in range(max_routes_per_vehicle):
        if remaining_m3 <= 1e-6 or remaining_ton <= 1e-6:
            break

        available_vehicles = vehicles[vehicles['Veiculo'].apply(lambda v: len(vehicle_usage[v]) < vehicles.loc[vehicles['Veiculo'] == v, 'Quantidade'].values[0] * max_routes_per_vehicle)]
        
        if available_vehicles.empty:
            break

        vehicle = random.choice(available_vehicles['Veiculo'].tolist())
        vehicle_row = available_vehicles[available_vehicles['Veiculo'] == vehicle].iloc[0]

        vehicle_capacity_m3 = vehicle_row['Capacidade_m3']
        vehicle_capacity_ton = vehicle_row['Capacidade_ton']

        available_m3 = min(remaining_m3, vehicle_capacity_m3)
        available_ton = min(remaining_ton, vehicle_capacity_ton)

        if available_m3 > vehicle_capacity_m3:
            available_m3 = vehicle_capacity_m3
            available_ton = available_m3 * density

        if available_ton > vehicle_capacity_ton:
            available_ton = vehicle_capacity_ton
            available_m3 = available_ton / density

        if available_m3 <= 1e-6 or available_ton <= 1e-6:
            continue

        supplier = random.choice(available_suppliers['Fornecedor'].tolist())
        distance = suppliers.loc[suppliers['Fornecedor'] == supplier, 'DistanciaEntrega_km'].values[0]
        day = random.randint(1, 5)
        period = random.choice(['morning', 'afternoon'])

        if not any(d == day and p == period for d, p in vehicle_usage[vehicle]):
            new_deliveries.append(Delivery(supplier, material, vehicle, available_m3, available_ton, day, period, distance))
            vehicle_usage[vehicle].append((day, period))
            remaining_m3 -= available_m3
            remaining_ton -= available_ton

            logging.debug(f"Alocado {available_m3:.2f} m³ / {available_ton:.2f} ton de {material} para {supplier} usando {vehicle} com verificação de tempo")

    return new_deliveries


def generate_valid_delivery(material, vehicle, supplier, day, period, vehicles, materials, suppliers):
    # Verificar capacidade do veículo e disponibilidade de horário
    vehicle_row = vehicles[vehicles['Veiculo'] == vehicle].iloc[0]
    material_row = materials[materials['Material'] == material].iloc[0]

    max_quantity_m3 = min(material_row['Quantidade_m3'], vehicle_row['Capacidade_m3'])
    max_quantity_ton = min(material_row['Quantidade_ton'], vehicle_row['Capacidade_ton'])

    quantity_m3 = random.uniform(0, max_quantity_m3)
    quantity_ton = min(quantity_m3 * (material_row['Quantidade_ton'] / material_row['Quantidade_m3']), max_quantity_ton)

    if quantity_ton > vehicle_row['Capacidade_ton']:
        quantity_ton = vehicle_row['Capacidade_ton']
        quantity_m3 = min(vehicle_row['Capacidade_m3'], quantity_ton / (material_row['Quantidade_ton'] / material_row['Quantidade_m3']))

    # Verificar disponibilidade de horários
    available_periods = [(day, period) for day in range(1, 6) for period in ['morning', 'afternoon']]
    if (day, period) not in available_periods:
        logging.warning(f"Horário {day}-{period} não disponível para o veículo {vehicle}")
        return None

    distance = suppliers.loc[suppliers['Fornecedor'] == supplier, 'DistanciaEntrega_km'].values[0]

    return Delivery(supplier, material, vehicle, quantity_m3, quantity_ton, day, period, distance)


def repair_solution(solution, materials, suppliers, vehicles, densities, max_trips_per_week):
    delivered_materials = defaultdict(lambda: {'m3': 0, 'ton': 0})
    vehicle_usage = defaultdict(list)

    valid_deliveries = []
    for delivery in solution.deliveries:
        if check_time_availability(delivery.vehicle, delivery.day, delivery.period, vehicle_usage[delivery.vehicle]):
            valid_deliveries.append(delivery)
            for stop in delivery.stops:
                delivered_materials[stop['material']]['m3'] += stop['quantity_m3']
                delivered_materials[stop['material']]['ton'] += stop['quantity_ton']
            vehicle_usage[delivery.vehicle].append((delivery.day, delivery.period))

    solution.deliveries = valid_deliveries

    for material, quantities in delivered_materials.items():
        required_m3 = materials.loc[materials['Material'] == material, 'Quantidade_m3'].values[0]
        required_ton = materials.loc[materials['Material'] == material, 'Quantidade_ton'].values[0]

        if quantities['m3'] < required_m3 * 0.99 or quantities['ton'] < required_ton * 0.99:
            remaining_m3 = max(0, required_m3 - quantities['m3'])
            remaining_ton = max(0, required_ton - quantities['ton'])

            logging.info(f"Reparando déficit para {material}: Adicionando {remaining_m3:.2f} m³ / {remaining_ton:.2f} ton")
            new_deliveries = allocate_load(material, remaining_m3, remaining_ton, suppliers, vehicles, solution.deliveries, materials, densities, max_trips_per_week)
            solution.deliveries.extend(new_deliveries)

    return solution


def print_solution(solution):
    # Ordenar as entregas por veículo_id, dia e período
    sorted_deliveries = sorted(solution.deliveries, key=lambda d: (d.vehicle_id, d.day, d.period))
    
    current_vehicle = None
    for delivery in sorted_deliveries:
        # Imprimir informações do veículo se mudou
        if delivery.vehicle_id != current_vehicle:
            if current_vehicle is not None:
                print()  # Linha em branco entre veículos
            print(f"Veículo: {delivery.vehicle} (ID: {delivery.vehicle_id})")
            current_vehicle = delivery.vehicle_id
        
        print(f"  Rota {delivery.route_id}: Dia {delivery.day}, {delivery.period.capitalize()}")
        print(f"    Distância Total: {delivery.total_distance:.2f} km")
        print("    Paradas:")
        
        total_m3 = 0
        total_ton = 0
        for i, stop in enumerate(delivery.stops, 1):
            print(f"      Parada {i}: Fornecedor: {stop['supplier']}, Material: {stop['material']}, "
                  f"Quantidade: {stop['quantity_m3']:.2f} m³ / {stop['quantity_ton']:.2f} ton, "
                  f"Distância: {stop['distance']:.2f} km")
            total_m3 += stop['quantity_m3']
            total_ton += stop['quantity_ton']
        
        print(f"    Total da rota: {total_m3:.2f} m³ / {total_ton:.2f} ton")
    
    print("\nTotais Gerais:")
    total_vehicles = len(set(d.vehicle_id for d in solution.deliveries))
    total_m3 = sum(sum(stop['quantity_m3'] for stop in d.stops) for d in solution.deliveries)
    total_ton = sum(sum(stop['quantity_ton'] for stop in d.stops) for d in solution.deliveries)
    total_distance = sum(d.total_distance for d in solution.deliveries)
    
    print(f"Total de Veículos Utilizados: {total_vehicles}")
    print(f"Quantidade Total Entregue: {total_m3:.2f} m³ / {total_ton:.2f} ton")
    print(f"Distância Total Percorrida: {total_distance:.2f} km")
    
# Funções para ler os arquivos CSV
def read_material_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def read_supplier_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def read_cost_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def read_vehicle_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def save_solution_to_json(solution, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_solution_{timestamp}.json"
    
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
        "deliveries": [
            {
                "route_id": d.route_id,
                "vehicle": d.vehicle,
                "vehicle_id": d.vehicle_id,
                "day": convert_to_serializable(d.day),
                "period": d.period,
                "total_distance": convert_to_serializable(d.total_distance),
                "stops": [
                    {
                        "supplier": stop["supplier"],
                        "material": stop["material"],
                        "quantity_m3": convert_to_serializable(stop["quantity_m3"]),
                        "quantity_ton": convert_to_serializable(stop["quantity_ton"]),
                        "distance": convert_to_serializable(stop["distance"])
                    } for stop in d.stops
                ]
            } for d in solution.deliveries
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=4, default=convert_to_serializable)
    
    print(f"Solução salva em {filename}")

def log_input_data(materials, suppliers, vehicles):
    logging.info("Dados de entrada:")
    logging.info(f"Materiais: {len(materials)} tipos")
    for _, material in materials.iterrows():
        logging.info(f"  {material['Material']}: {material['Quantidade_m3']} m³ / {material['Quantidade_ton']} ton")
    
    logging.info(f"Fornecedores: {len(suppliers)}")
    for _, supplier in suppliers.iterrows():
        materials_supplied = [m for m in ['M1', 'M2', 'M3', 'M4'] if supplier[m] == 'X']
        logging.info(f"  {supplier['Fornecedor']}: Fornece {', '.join(materials_supplied)}")
    
    logging.info(f"Veículos: {len(vehicles)} tipos")
    for _, vehicle in vehicles.iterrows():
        logging.info(f"  {vehicle['Veiculo']}: {vehicle['Quantidade']} disponíveis, Capacidade: {vehicle['Capacidade_m3']} m³ / {vehicle['Capacidade_ton']} ton")


def estimate_max_trips_per_week(vehicles, suppliers):
    max_trips = {}
    for _, vehicle in vehicles.iterrows():
        # Assumindo que temos 5 dias úteis e 2 períodos por dia
        total_periods = 5 * 2
        
        # Estimativa de tempo médio para uma viagem (em períodos)
        avg_distance = suppliers['DistanciaEntrega_km'].mean()
        avg_trip_time = (avg_distance / 600.0) * 2  # Ida e volta
        loading_time = 0.5  # Tempo de carga
        unloading_time = 0.5  # Tempo de descarga
        total_trip_time = avg_trip_time + loading_time + unloading_time
        
        # Número máximo de viagens por semana
        max_trips[vehicle['Veiculo']] = min(int(total_periods / total_trip_time), 10)  # Limitando a 10 viagens por semana
    
    return max_trips

def estimate_problem_difficulty(materials, suppliers, vehicles, max_trips_per_week):
    total_m3_required = materials['Quantidade_m3'].sum()
    total_ton_required = materials['Quantidade_ton'].sum()
    
    total_m3_capacity = sum(v['Capacidade_m3'] * v['Quantidade'] * max_trips_per_week[v['Veiculo']] for _, v in vehicles.iterrows())
    total_ton_capacity = sum(v['Capacidade_ton'] * v['Quantidade'] * max_trips_per_week[v['Veiculo']] for _, v in vehicles.iterrows())
    
    m3_utilization = total_m3_required / total_m3_capacity
    ton_utilization = total_ton_required / total_ton_capacity
    
    logging.info(f"Estimativa de utilização de capacidade: {m3_utilization:.2%} (m³), {ton_utilization:.2%} (ton)")
    
    if m3_utilization > 1 or ton_utilization > 1:
        logging.warning("O problema pode ser muito difícil ou impossível de resolver com os recursos disponíveis.")
    elif max(m3_utilization, ton_utilization) > 0.9:
        logging.warning("O problema parece ser desafiador, pode levar mais tempo para encontrar uma solução ótima.")
    else:
        logging.info("O problema parece ser viável com os recursos disponíveis.")
    
    for _, material in materials.iterrows():
        suppliers_for_material = suppliers[suppliers[material['Material']] == 'X']
        if suppliers_for_material.empty:
            logging.error(f"Nenhum fornecedor disponível para o material {material['Material']}")
    
    return max(m3_utilization, ton_utilization)

# Função principal para executar o algoritmo
def main():
    # Leitura dos dados de entrada
    materials = read_material_data('materials.csv')
    suppliers = read_supplier_data('suppliers.csv')
    vehicles = read_vehicle_data('vehicles.csv')
    costs = read_cost_data('costs.csv')

    # Verificar se todos os nomes dos veículos estão nas colunas dos custos
    vehicle_names = vehicles['Veiculo'].tolist()
    cost_columns = costs.columns.tolist()

    max_trips_per_week = estimate_max_trips_per_week(vehicles, suppliers)
    logging.info(f"Estimativa de viagens máximas por semana: {max_trips_per_week}")

    missing_vehicles = [v for v in vehicle_names if v not in cost_columns]
    if missing_vehicles:
        print(f"Veículos ausentes na tabela de custos: {missing_vehicles}")
        return

    # Configuração dos parâmetros do algoritmo genético
    population_size = 100
    generations = 1000
    elite_size = 10
    log_input_data(materials, suppliers, vehicles)
    
    difficulty = estimate_problem_difficulty(materials, suppliers, vehicles, max_trips_per_week)
    if difficulty > 1:
        print("O problema pode ser muito difícil ou impossível de resolver. Considere aumentar os recursos ou relaxar as restrições.")

    # Execução do algoritmo genético
    best_solution = genetic_algorithm(materials, suppliers, vehicles, costs, max_trips_per_week,
                                      population_size=population_size, 
                                      generations=generations, 
                                      elite_size=elite_size)
    if best_solution is None or best_solution.fitness == float('inf'):
        print("Não foi possível encontrar uma solução válida.")
        return

    # Imprimir a melhor solução encontrada
    print("\nMelhor solução encontrada:")
    print_solution(best_solution)

    # Imprimir as estatísticas finais
    print(f"\nEstatísticas finais:")
    print(f"Fitness: {best_solution.fitness}")
    print(f"Custo total: {best_solution.total_cost}")
    print(f"Distância total: {best_solution.total_distance}")

    # Salvar a melhor solução em um arquivo JSON
    save_solution_to_json(best_solution)

# Executar o programa principal
if __name__ == "__main__":
    main()