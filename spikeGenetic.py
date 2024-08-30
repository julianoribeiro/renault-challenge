import pandas as pd
import numpy as np
import random
import logging

# Funções para ler os arquivos CSV
def read_material_data(file_path):
    return pd.read_csv(file_path)

def read_supplier_data(file_path):
    return pd.read_csv(file_path)

def read_vehicle_data(file_path):
    return pd.read_csv(file_path)

def read_cost_data(file_path):
    return pd.read_csv(file_path)

# Configuração de logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Classe Delivery
class Delivery:
    def __init__(self, supplier, material, vehicle, quantity_m3, quantity_ton, day, period):
        self.supplier = supplier
        self.material = material
        self.vehicle = vehicle
        self.quantity_m3 = quantity_m3
        self.quantity_ton = quantity_ton
        self.day = day
        self.period = period

# Classe base ClausulaRestritiva
class ClausulaRestritiva:
    def validar(self, solucao):
        raise NotImplementedError("Método validar deve ser implementado na subclasse")

# Subclasse de ClausulaRestritiva
class CapacidadeVeiculoClausula(ClausulaRestritiva):
    def validar(self, solucao):
        # Valida se a capacidade do veículo não foi excedida em cada entrega
        for delivery in solucao.deliveries:
            if delivery.quantity_m3 > delivery.vehicle.capacity_m3 or delivery.quantity_ton > delivery.vehicle.capacity_ton:
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
        # Penaliza soluções com mais de 5 entregas no mesmo período
        entregas_por_periodo = {}
        for delivery in solucao.deliveries:
            key = (delivery.day, delivery.period)
            if key not in entregas_por_periodo:
                entregas_por_periodo[key] = 0
            entregas_por_periodo[key] += 1
            if entregas_por_periodo[key] > 5:
                penalidade += 10  # Penalidade arbitrária para excesso de entregas
        return penalidade

# Classe Solucao
class Solucao:
    def __init__(self, deliveries=None):
        if deliveries is None:
            deliveries = []
        self.deliveries = deliveries
        self.restricoes = [CapacidadeVeiculoClausula()]  # Lista de instâncias de ClausulaRestritiva
        self.limitacoes = [PenalizacaoMuitasEntregasClausula()]  # Lista de instâncias de ClausulaLimitante

    def calcular_fitness(self):
        # Verifica se a solução é válida
        for restricao in self.restricoes:
            if not restricao.validar(self):
                return float('inf')  # Retorna um valor muito alto para soluções inválidas

        # Calcula a penalidade total
        penalidade_total = 0
        for limitante in self.limitacoes:
            penalidade_total += limitante.penalizar(self)

        # Calcular o fitness base (simples soma de penalidades neste caso)
        fitness = penalidade_total
        return fitness

# Função de fitness
def fitness(solution, suppliers, vehicles, costs):
    total_cost = 0
    total_distance = 0
    penalty = 0

    vehicle_usage = {}
    supplier_routes = {}

    logging.debug("Iniciando avaliação de fitness para a solução.")

    if not solution.deliveries:
        logging.warning("Nenhuma entrega válida na solução.")
        return float('inf'), float('inf'), float('inf')

    for delivery in solution.deliveries:
        supplier = delivery.supplier
        vehicle = delivery.vehicle
        material = delivery.material

        logging.debug(f"Avaliando entrega: Fornecedor {supplier}, Veículo {vehicle}, Material {material}")

        if vehicle not in vehicle_usage:
            vehicle_usage[vehicle] = 0
        vehicle_usage[vehicle] += 1

        if supplier not in supplier_routes:
            supplier_routes[supplier] = []
        supplier_routes[supplier].append(vehicle)

        supplier_row = suppliers[suppliers['Fornecedor'] == supplier].iloc[0]
        distance_to_delivery = supplier_row['DistanciaEntrega_km']

        vehicle_row = vehicles[vehicles['Veiculo'] == vehicle].iloc[0]
        vehicle_capacity_m3 = vehicle_row['Capacidade_m3']
        vehicle_capacity_ton = vehicle_row['Capacidade_ton']

        if delivery.quantity_m3 > vehicle_capacity_m3 or delivery.quantity_ton > vehicle_capacity_ton:
            logging.warning(f"Capacidade do veículo excedida: Veículo {vehicle}, Capacidade {vehicle_capacity_m3} m³ / {vehicle_capacity_ton} ton, "
                            f"Entrega {delivery.quantity_m3} m³ / {delivery.quantity_ton} ton.")
            penalty += 100000  # Aplicar uma penalidade alta em vez de retornar inf
            continue  # Ignorar o custo dessa entrega e continuar com as outras

        try:
            cost_per_km = costs.loc[costs['Km'] >= distance_to_delivery].iloc[0][vehicle]
        except (IndexError, KeyError):
            logging.error(f"Não foi possível encontrar o custo por km para o veículo {vehicle} e distância {distance_to_delivery} km.")
            penalty += 100000  # Penalidade adicional se o custo por km não for encontrado
            continue

        if not isinstance(cost_per_km, (int, float)) or cost_per_km <= 0:
            logging.error(f"Custo por km inválido para o veículo {vehicle}: {cost_per_km}")
            penalty += 100000  # Penalidade adicional se o custo por km for inválido
            continue

        delivery_cost = distance_to_delivery * cost_per_km
        total_cost += delivery_cost

        travel_time = distance_to_delivery
        total_distance += travel_time

    logging.debug(f"Penalidades calculadas: Veículo = {penalty}, Rota = {penalty}")

    if total_cost == 0 or total_distance == 0:
        logging.error(f"Cálculo inválido: total_cost = {total_cost}, total_distance = {total_distance}")
        return float('inf'), float('inf'), float('inf')

    weighted_fitness = (total_cost * 7) + (total_distance * 3) + penalty

    logging.debug(f"Final da avaliação de fitness: Custo Total {total_cost}, Tempo Total {total_distance}, Fitness Ponderado {weighted_fitness}.")

    return weighted_fitness, total_cost, total_distance


# Função para gerar solução inicial
def gerar_solucao_inicial(materials, suppliers, vehicles, direct=True):
    deliveries = []
    
    for index, material_row in materials.iterrows():
        material = material_row['Material']
        quantity_m3 = material_row['Quantidade_m3']
        quantity_ton = material_row['Quantidade_ton']
        
        if direct:
            solution, remaining_m3, remaining_ton = allocate_direct_load(material, quantity_m3, quantity_ton, suppliers, vehicles, deliveries)
        else:
            solution, remaining_m3, remaining_ton = allocate_load_with_distance_check(material, quantity_m3, quantity_ton, suppliers, vehicles, suppliers.iloc[0], deliveries)
        
        if remaining_m3 > 0 or remaining_ton > 0:
            logging.warning(f"Não foi possível alocar completamente o material {material}. Quantidade restante: {remaining_m3} m³, {remaining_ton} ton.")
        
    return Solucao(deliveries)

# Função para gerar a população inicial
def generate_population(materials, suppliers, vehicles, population_size):
    population = []

    for i in range(population_size):
        direct = i % 2 == 0  # Metade das soluções com rotas diretas, metade com múltiplos fornecedores
        solution = gerar_solucao_inicial(materials, suppliers, vehicles, direct=direct)
        
        if solution is None:
            logging.warning(f"Solução inicial na iteração {i} retornou None.")
            continue
        
        population.append(solution)
    
    if not population:
        logging.error("Nenhuma solução válida foi gerada na população inicial.")
    else:
        logging.debug(f"População inicial gerada com {len(population)} soluções.")

    return population

# Função principal do algoritmo genético
def genetic_algorithm(materials, suppliers, vehicles, costs, population_size=50, generations=100):
    population = generate_population(materials, suppliers, vehicles, population_size)
    
    best_solution = None
    best_fitness = float('inf')
    best_total_cost = float('inf')
    best_total_distance = float('inf')
    
    mutation_rate = 0.1  # Taxa de mutação inicial
    stable_generations = 0  # Contador de gerações sem melhoria
    max_stable_generations = 10  # Número máximo de gerações sem melhoria antes de aumentar a mutação
    
    for generation in range(generations):
        population_fitness = []
        for solution in population:
            fitness_value, total_cost, total_distance = fitness(solution, suppliers, vehicles, costs)
            population_fitness.append((solution, fitness_value, total_cost, total_distance))
        
        population_fitness.sort(key=lambda x: x[1])
        
        if population_fitness[0][1] < best_fitness:
            best_solution = population_fitness[0][0]
            best_fitness = population_fitness[0][1]
            best_total_cost = population_fitness[0][2]
            best_total_distance = population_fitness[0][3]
            stable_generations = 0  # Resetar contador se houver melhoria
        else:
            stable_generations += 1
        
        # Ajuste adaptativo da taxa de mutação
        if stable_generations >= max_stable_generations:
            mutation_rate = min(mutation_rate + 0.1, 0.5)  # Aumentar a taxa de mutação (máx 0.5)
            stable_generations = 0  # Resetar contador
        else:
            mutation_rate = max(mutation_rate - 0.05, 0.1)  # Reduzir a taxa de mutação (mín 0.1)
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Total Cost = {best_total_cost}, Total Distance = {best_total_distance}")
        
        selected_population = [x[0] for x in population_fitness[:population_size // 2]]
        
        new_population = []
        while len(new_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            child1, child2 = crossover(parent1.deliveries, parent2.deliveries)
            new_population.extend([Solucao(child1), Solucao(child2)])
        
        # Aplicar mutação adaptativa na nova população
        new_population = mutate_population(new_population, suppliers, vehicles, mutation_rate=mutation_rate)
        
        population = new_population
    
    print("\nMelhor solução encontrada:")
    print_solution(best_solution)
    print(f"Total Cost: {best_total_cost}, Total Time: {best_total_distance}, Weighted Fitness: {best_fitness}")
    
    return best_solution





# Função de alocação de carga direta
def allocate_direct_load(material, remaining_m3, remaining_ton, suppliers, vehicles, solution):
    for supplier in suppliers['Fornecedor']:
        supplier_row = suppliers[suppliers['Fornecedor'] == supplier].iloc[0]
        for vehicle in vehicles['Veiculo']:
            vehicle_row = vehicles[vehicles['Veiculo'] == vehicle].iloc[0]
            vehicle_capacity_m3 = vehicle_row['Capacidade_m3']
            vehicle_capacity_ton = vehicle_row['Capacidade_ton']

            # Verifica se o veículo tem capacidade para carregar o material
            load_m3 = min(remaining_m3, vehicle_capacity_m3)
            load_ton = min(remaining_ton, vehicle_capacity_ton)

            if load_m3 > 0 and load_ton > 0:
                day = random.randint(1, 5)  # Seleciona um dia aleatório (1 a 5)
                period = random.choice(['morning', 'afternoon'])  # Seleciona um período aleatório
                solution.append(Delivery(supplier, material, vehicle, load_m3, load_ton, day, period))
                remaining_m3 -= load_m3
                remaining_ton -= load_ton

                logging.debug(f"Alocando {load_m3} m³ e {load_ton} ton de {material} do fornecedor {supplier} com o veículo {vehicle} no dia {day} durante o período {period}")

            if remaining_m3 <= 0 and remaining_ton <= 0:
                logging.debug(f"Todo o material {material} foi alocado.")
                break
        if remaining_m3 <= 0 and remaining_ton <= 0:
            break

    return solution, remaining_m3, remaining_ton

# Função de alocação com verificação de distância 
def allocate_load_with_distance_check(material, remaining_m3, remaining_ton, suppliers, vehicles, current_supplier, solution):
    for index, supplier_row in suppliers.iterrows():
        if remaining_m3 <= 0 or remaining_ton <= 0:
            break

        available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= remaining_m3) & (vehicles['Capacidade_ton'] >= remaining_ton)]
        if available_vehicles.empty:
            continue
        
        vehicle_row = available_vehicles.sample().iloc[0]
        supply_m3 = min(remaining_m3, vehicle_row['Capacidade_m3'])
        supply_ton = min(remaining_ton, vehicle_row['Capacidade_ton'])

        day = random.randint(1, 5)
        period = random.choice(['morning', 'afternoon'])

        delivery = Delivery(supplier_row['Fornecedor'], material, vehicle_row['Veiculo'], supply_m3, supply_ton, day, period)
        solution.append(delivery)

        remaining_m3 -= supply_m3
        remaining_ton -= supply_ton
    
    return solution, remaining_m3, remaining_ton

# Função de crossover
def crossover(parent1, parent2):
    # Verificar se as listas dos pais são adequadas para crossover
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1, parent2  # Retorna os pais inalterados se não for possível realizar o crossover
    
    # Escolher um ponto de crossover seguro
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    
    # Criar os filhos combinando partes dos pais
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

# Função para mutar a população
def mutate_population(population, suppliers, vehicles, mutation_rate=0.2):
    new_population = []
    for solution in population:
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Mutar cenários diretos
                mutate(solution.deliveries, suppliers, vehicles, direct=True)
            else:
                # Mutar cenários com múltiplos fornecedores
                mutate(solution.deliveries, suppliers, vehicles, direct=False)
        new_population.append(solution)
    return new_population

# Função de mutação
def mutate(solution, suppliers, vehicles, direct=True):
    if not solution:
        return
    
    delivery = random.choice(solution)
    
    mutation_type = random.choice(['supplier', 'vehicle', 'day_period'])
    
    if mutation_type == 'supplier':
        available_suppliers = suppliers[suppliers[delivery.material] == 'X']
        new_supplier_row = available_suppliers.sample().iloc[0]
        delivery.supplier = new_supplier_row['Fornecedor']
    
    elif mutation_type == 'vehicle':
        if direct:
            available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= delivery.quantity_m3) & (vehicles['Capacidade_ton'] >= delivery.quantity_ton)]
        else:
            available_vehicles = vehicles[(vehicles['Capacidade_m3'] >= delivery.quantity_m3 / 2) & (vehicles['Capacidade_ton'] >= delivery.quantity_ton / 2)]
        
        if not available_vehicles.empty:
            new_vehicle_row = available_vehicles.sample().iloc[0]
            delivery.vehicle = new_vehicle_row['Veiculo']
    
    elif mutation_type == 'day_period':
        delivery.day = random.randint(1, 5)
        delivery.period = random.choice(['morning', 'afternoon'])

# Função para exibir uma solução
def print_solution(solution):
    current_vehicle = None
    route = []
    
    for delivery in solution.deliveries:
        if delivery.vehicle != current_vehicle:
            # Se estamos mudando de veículo, devemos imprimir a rota anterior
            if current_vehicle is not None:
                print(f"Vehicle: {current_vehicle}")
                for stop in route:
                    print(f"    Supplier: {stop['supplier']}, Material: {stop['material']}, Quantity (m³): {stop['quantity_m3']}, Quantity (ton): {stop['quantity_ton']}, Day: {stop['day']}, Period: {stop['period']}")
                print()  # Linha em branco para separar as rotas
            # Iniciar a rota para o novo veículo
            current_vehicle = delivery.vehicle
            route = []
        
        # Adicionar a entrega atual à rota
        route.append({
            'supplier': delivery.supplier,
            'material': delivery.material,
            'quantity_m3': delivery.quantity_m3,
            'quantity_ton': delivery.quantity_ton,
            'day': delivery.day,
            'period': delivery.period
        })
    
    # Imprimir a última rota
    if route:
        print(f"Vehicle: {current_vehicle}")
        for stop in route:
            print(f"    Supplier: {stop['supplier']}, Material: {stop['material']}, Quantity (m³): {stop['quantity_m3']}, Quantity (ton): {stop['quantity_ton']}, Day: {stop['day']}, Period: {stop['period']}")

# Main - Execução do algoritmo
materials = read_material_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/materials.csv')
suppliers = read_supplier_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/suppliers.csv')
vehicles = read_vehicle_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/vehicles.csv')
costs = read_cost_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/costs.csv')

# Verificar se todos os nomes dos veículos estão nas colunas dos custos
vehicle_names = vehicles['Veiculo'].tolist()
cost_columns = costs.columns.tolist()

missing_vehicles = [v for v in vehicle_names if v not in cost_columns]
if missing_vehicles:
    print(f"Veículos ausentes na tabela de custos: {missing_vehicles}")

# Executar o algoritmo genético
best_solution = genetic_algorithm(materials, suppliers, vehicles, costs)

