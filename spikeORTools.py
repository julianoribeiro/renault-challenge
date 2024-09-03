import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Funções para ler os arquivos CSV
def read_material_data(file_path):
    return pd.read_csv(file_path)

def read_supplier_data(file_path):
    return pd.read_csv(file_path)

def read_vehicle_data(file_path):
    return pd.read_csv(file_path)

def read_cost_data(file_path):
    return pd.read_csv(file_path)

# Leitura dos dados
materials = read_material_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/materials.csv')
suppliers = read_supplier_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/suppliers.csv')
vehicles = read_vehicle_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/vehicles.csv')
costs = read_cost_data('/home/marcelo/Doutorado/projetos/github/PHD/Renault/costs.csv')

# Inicializar o gerenciador de dados
manager = pywrapcp.RoutingIndexManager(len(suppliers), len(vehicles), 0)

# Criar o modelo de roteamento
routing = pywrapcp.RoutingModel(manager)

# Função de distância (usando as distâncias dos suppliers)
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    if from_node >= len(suppliers) or to_node >= len(suppliers):
        return 0  # Retornar uma distância zero como fallback
    return int(suppliers.iloc[from_node]['DistanciaEntrega_km'])

distance_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

# Função de demanda (usando a quantidade de toneladas dos materiais)
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    if from_node >= len(materials):
        return 0  # Retornar uma demanda zero como fallback
    return int(materials.iloc[from_node]['Quantidade_ton'])

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

# Definir a capacidade dos veículos
capacities = [int(cap) for cap in vehicles['Capacidade_ton'].tolist()]
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # Sem capacidade extra
    capacities,  # Capacidades dos veículos
    True,  # Início cumulativo
    'Capacity'
)

# Ajuste de estratégia de busca
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
search_parameters.time_limit.seconds = 60  # Aumentar o limite de tempo para 60 segundos

# Resolver o problema
solution = routing.SolveWithParameters(search_parameters)

# Imprimir a solução
if solution:
    print('Objective: {}'.format(solution.ObjectiveValue()))
    for vehicle_id in range(len(vehicles)):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
else:
    print('No solution found!')
