from typing import List, Dict
import random
import numpy as np
import csv

class CVRPSolution:
    def __init__(self, clients: List[Dict], depot: Dict, vehicle_capacity: float, routes: List[List[int]] = None):
        self.clients = clients
        self.depot = depot
        self.capacity = vehicle_capacity
        self.routes = routes if routes else self._initialize_routes()
        self.fitness = self.calculate_fitness()
        
    def _initialize_routes(self) -> List[List[int]]:
        """Crea solución inicial usando Nearest Neighbor con verificación de capacidad"""
        clients_copy = [c for c in self.clients if c['id'] != 0]  # Excluir depósito
        random.shuffle(clients_copy)  # Para diversidad en población inicial
        
        routes = []
        current_route = [0]  # Comenzar en depósito
        current_load = 0
        
        while clients_copy:
            last_pos = current_route[-1]
            min_dist = float('inf')
            next_client = None
            idx = -1
            
            for i, client in enumerate(clients_copy):
                dist = self._distance(last_pos, client['id'])
                if (client['demand'] + current_load <= self.capacity) and (dist < min_dist):
                    min_dist = dist
                    next_client = client
                    idx = i
            
            if next_client:
                current_route.append(next_client['id'])
                current_load += next_client['demand']
                clients_copy.pop(idx)
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
                current_load = 0
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        return routes
    
    def calculate_fitness(self) -> float:
        """Calcula la distancia total de todas las rutas"""
        total_distance = 0
        for route in self.routes:
            for i in range(len(route)-1):
                total_distance += self._distance(route[i], route[i+1])
        return total_distance
    
    def _distance(self, id1: int, id2: int) -> float:
        """Distancia euclidiana entre dos puntos"""
        if id1 == 0:
            pos1 = self.depot
        else:
            pos1 = next(c for c in self.clients if c['id'] == id1)
        
        if id2 == 0:
            pos2 = self.depot
        else:
            pos2 = next(c for c in self.clients if c['id'] == id2)
        
        return np.sqrt((pos1['x']-pos2['x'])**2 + (pos1['y']-pos2['y'])**2)
    
    def is_valid(self) -> bool:
        """Verifica que la solución cumpla con todas las restricciones"""
        all_clients = set(c['id'] for c in self.clients if c['id'] != 0)
        visited = set()
        
        for route in self.routes:
            if route[0] != 0 or route[-1] != 0:
                return False
            
            route_demand = sum(self._get_demand(c) for c in route[1:-1])
            if route_demand > self.capacity:
                return False
            
            for client in route[1:-1]:
                if client in visited:
                    return False
                visited.add(client)
        
        return visited == all_clients
    
    def _get_demand(self, client_id: int) -> float:
        return next(c['demand'] for c in self.clients if c['id'] == client_id)
    
    def to_verification_csv(self, filename: str):
        """Genera archivo CSV de verificación según formato requerido"""
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["cliente", "secuencia_visita", "ruta_asignada", "distancia_acumulada"])
            
            for route_idx, route in enumerate(self.routes):
                accumulated_dist = 0
                for i in range(len(route)-1):
                    client = route[i]
                    next_client = route[i+1]
                    dist = self._distance(client, next_client)
                    accumulated_dist += dist
                    
                    if client != 0:  # Ignorar depósito
                        writer.writerow([
                            client,
                            f"{route_idx}-{i}",
                            route_idx + 1,
                            round(accumulated_dist, 2)
                        ])
    
    def get_route_stats(self) -> Dict:
        """Devuelve estadísticas de balance de carga por ruta"""
        stats = {
            'num_routes': len(self.routes),
            'demands': [],
            'distances': []
        }
        
        for route in self.routes:
            demand = sum(self._get_demand(c) for c in route[1:-1])
            distance = sum(self._distance(route[i], route[i+1]) for i in range(len(route)-1))
            stats['demands'].append(demand)
            stats['distances'].append(distance)
        
        return stats