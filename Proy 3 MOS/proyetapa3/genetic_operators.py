from typing import Tuple, List, Dict
from solution import CVRPSolution
import random

class GeneticOperators:
    @staticmethod
    def pmx_crossover(parent1: CVRPSolution, parent2: CVRPSolution) -> Tuple[CVRPSolution, CVRPSolution]:
        flat_parent1 = [c for route in parent1.routes for c in route[1:-1]]
        flat_parent2 = [c for route in parent2.routes for c in route[1:-1]]
        
        size = len(flat_parent1)
        point1 = random.randint(0, size-1)
        point2 = random.randint(point1, size-1)
        
        child1_flat = GeneticOperators._pmx_helper(flat_parent1, flat_parent2, point1, point2)
        child2_flat = GeneticOperators._pmx_helper(flat_parent2, flat_parent1, point1, point2)
        
        child1_sol = GeneticOperators._split_to_routes(
            child1_flat, 
            parent1.clients, 
            parent1.depot, 
            parent1.capacity, 
            **{k: v for k, v in parent1.__dict__.items() if k not in ['clients', 'depot', 'capacity', 'routes', 'fitness', '_distance_matrix']}
        )
        child2_sol = GeneticOperators._split_to_routes(
            child2_flat, 
            parent2.clients, 
            parent2.depot, 
            parent2.capacity, 
            **{k: v for k, v in parent2.__dict__.items() if k not in ['clients', 'depot', 'capacity', 'routes', 'fitness', '_distance_matrix']}
        )
        
        return child1_sol, child2_sol
    
    @staticmethod
    def _pmx_helper(parent1: List[int], parent2: List[int], point1: int, point2: int) -> List[int]:
        size = len(parent1)
        child = [-1] * size
        
        child[point1:point2+1] = parent2[point1:point2+1]
        
        for i in list(range(0, point1)) + list(range(point2+1, size)):
            if parent1[i] not in child[point1:point2+1]:
                child[i] = parent1[i]
            else:
                j = parent2.index(parent1[i])
                while parent1[j] in child[point1:point2+1]:
                    j = parent2.index(parent1[j])
                child[i] = parent1[j]
        
        return child
    
    @staticmethod
    def _split_to_routes(flat_solution: List[int], clients: List[Dict], depot: Dict, vehicle_capacity: float, **kwargs) -> CVRPSolution:
        routes = []
        current_route = [0]
        current_load = 0
        
        def get_demand_from_list(client_id: int, clients_list: List[Dict]) -> float:
             if client_id == 0: return 0.0
             return next((c['demand'] for c in clients_list if c['id'] == client_id), 0.0)

        for client_id in flat_solution:
            demand = get_demand_from_list(client_id, clients)
            
            if current_load + demand <= vehicle_capacity:
                current_route.append(client_id)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, client_id]
                current_load = demand
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        return CVRPSolution(
            clients=clients,
            depot=depot,
            vehicle_capacity=vehicle_capacity,
            routes=routes,
            **kwargs
        )
    
    @staticmethod
    def ordered_crossover(parent1: CVRPSolution, parent2: CVRPSolution, **solution_kwargs) -> Tuple[CVRPSolution, CVRPSolution]:
        """Realiza Ordered Crossover (OX1) entre dos padres."""
        # Obtener la secuencia de clientes visitados (excluyendo depósitos)
        flat_parent1 = [node for route in parent1.routes for node in route[1:-1]]
        flat_parent2 = [node for route in parent2.routes for node in route[1:-1]]
        
        size = len(flat_parent1)
        if size == 0: # No hay clientes para cruzar
            return parent1, parent2

        # Seleccionar puntos de corte aleatorios
        point1 = random.randint(0, size - 1)
        point2 = random.randint(point1, size - 1)

        # Crear hijos con el segmento del padre 1
        child1_flat = [None] * size
        child1_flat[point1 : point2 + 1] = flat_parent1[point1 : point2 + 1]

        # Llenar el resto del hijo 1 desde el padre 2 en orden cíclico
        parent2_segment = flat_parent2[point2 + 1 :] + flat_parent2[: point2 + 1]
        current_pos = (point2 + 1) % size
        for node in parent2_segment:
            if node not in child1_flat:
                while child1_flat[current_pos] is not None:
                    current_pos = (current_pos + 1) % size
                child1_flat[current_pos] = node
        
        # Crear hijo 2 (intercambiando roles de los padres)
        child2_flat = [None] * size
        child2_flat[point1 : point2 + 1] = flat_parent2[point1 : point2 + 1]
        
        parent1_segment = flat_parent1[point2 + 1 :] + flat_parent1[: point2 + 1]
        current_pos = (point2 + 1) % size
        for node in parent1_segment:
            if node not in child2_flat:
                while child2_flat[current_pos] is not None:
                    current_pos = (current_pos + 1) % size
                child2_flat[current_pos] = node

        # Convertir secuencias planas a soluciones CVRP con rutas factibles
        # Usamos _split_to_routes que ya considera capacidad y crea nuevas rutas si es necesario
        child1_sol = GeneticOperators._split_to_routes(
            child1_flat,
            parent1.clients,
            parent1.depot,
            parent1.capacity,
            **solution_kwargs # Pasar los kwargs aquí
        )
        child2_sol = GeneticOperators._split_to_routes(
            child2_flat,
            parent2.clients,
            parent2.depot,
            parent2.capacity,
            **solution_kwargs # Pasar los kwargs aquí
        )

        return child1_sol, child2_sol

    @staticmethod
    def swap_mutation(solution: CVRPSolution, **solution_kwargs) -> CVRPSolution:
        """Aplica mutación de intercambio (swap) a dos clientes aleatorios (pueden estar en diferentes rutas)."""
        flat_solution = [node for route in solution.routes for node in route[1:-1]]
        size = len(flat_solution)
        if size < 2: # Necesita al menos dos clientes para intercambiar
            return solution
        
        # Seleccionar dos posiciones aleatorias en la secuencia plana
        pos1, pos2 = random.sample(range(size), 2)
        
        # Realizar el intercambio
        flat_solution[pos1], flat_solution[pos2] = flat_solution[pos2], flat_solution[pos1]
        
        # Convertir secuencia plana mutada de vuelta a solución CVRP
        mutated_sol = GeneticOperators._split_to_routes(
            flat_solution,
            solution.clients,
            solution.depot,
            solution.capacity,
            **solution_kwargs # Pasar los kwargs aquí
        )
        return mutated_sol
        
    @staticmethod
    def inversion_mutation(solution: CVRPSolution, **solution_kwargs) -> CVRPSolution:
        """Aplica mutación de inversión (inversion) a un segmento aleatorio en la secuencia plana de clientes."""
        flat_solution = [node for route in solution.routes for node in route[1:-1]]
        size = len(flat_solution)
        if size < 2: # Necesita al menos dos clientes para invertir un segmento
            return solution

        # Seleccionar puntos de corte aleatorios en la secuencia plana
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)

        # Invertir el segmento
        segment_to_invert = flat_solution[start : end + 1]
        flat_solution[start : end + 1] = segment_to_invert[::-1]

        # Convertir secuencia plana mutada de vuelta a solución CVRP
        mutated_sol = GeneticOperators._split_to_routes(
            flat_solution,
            solution.clients,
            solution.depot,
            solution.capacity,
            **solution_kwargs # Pasar los kwargs aquí
        )
        return mutated_sol
    
    @staticmethod
    def _repair_solution(solution: CVRPSolution) -> CVRPSolution:
        flat_clients = [c for route in solution.routes for c in route[1:-1]]
        unique_clients = list(dict.fromkeys(flat_clients))
        
        return CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity,
            **{k: v for k, v in solution.__dict__.items() if k not in ['clients', 'depot', 'capacity', 'routes', 'fitness', '_distance_matrix']}
        )

def crossover(parent1_routes: List[List[int]], parent2_routes: List[List[int]]) -> List[List[int]]:
    """Realiza el cruce entre dos padres para producir un hijo."""
    # Seleccionar una ruta aleatoria de cada padre
    route1 = random.choice(parent1_routes)
    route2 = random.choice(parent2_routes)
    
    # Obtener los clientes (excluyendo el depósito)
    clients1 = [node for node in route1 if node != 0]
    clients2 = [node for node in route2 if node != 0]
    
    # Crear una nueva ruta combinando los clientes
    new_route = [0]  # Comenzar con el depósito
    
    # Añadir clientes de la primera ruta
    for client in clients1:
        if client not in new_route:
            new_route.append(client)
    
    # Añadir clientes de la segunda ruta que no estén ya en la nueva ruta
    for client in clients2:
        if client not in new_route:
            new_route.append(client)
    
    new_route.append(0)  # Terminar en el depósito
    
    # Devolver una lista con la nueva ruta
    return [new_route]

def mutation(routes: List[List[int]]) -> List[List[int]]:
    """Aplica mutación a las rutas."""
    # Hacer una copia profunda de las rutas
    new_routes = [route.copy() for route in routes]
    
    # Seleccionar una ruta aleatoria
    route_idx = random.randrange(len(new_routes))
    route = new_routes[route_idx]
    
    # Seleccionar dos posiciones aleatorias (excluyendo el depósito)
    if len(route) <= 3:  # Si la ruta solo tiene el depósito y un cliente
        return new_routes
        
    pos1, pos2 = random.sample(range(1, len(route)-1), 2)
    
    # Intercambiar los clientes en esas posiciones
    route[pos1], route[pos2] = route[pos2], route[pos1]
    
    return new_routes