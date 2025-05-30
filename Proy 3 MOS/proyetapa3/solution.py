from typing import List, Dict, Tuple
import random
import numpy as np
import csv
from distance_service import DistanceService

class CVRPSolution:
    def __init__(self, clients: List[Dict], depot: Dict, vehicle_capacity: float, 
                 routes: List[List[int]] = None, case_type: str = "base", 
                 distance_service: DistanceService = None, **kwargs):
        self.clients = clients
        self.depot = depot
        self.capacity = vehicle_capacity
        self.case_type = case_type.lower()
        self.fuel_prices = kwargs.get('fuel_prices', {})
        self.tolls = kwargs.get('tolls', {})
        self.distance_service = distance_service
        self.routes = routes if routes else self._initialize_routes()
        self.fitness = self.calculate_fitness()
    
    def _distance(self, id1: int, id2: int) -> float:
        """Calcula la distancia entre dos nodos usando el servicio de distancias."""
        if id1 == id2:
            return 0.0
        
        pos1 = self._get_coords(id1)
        pos2 = self._get_coords(id2)
        
        if self.distance_service:
            return self.distance_service.get_distance(pos1, pos2)
        return self._euclidean_distance(pos1, pos2)
    
    def _get_coords(self, node_id: int) -> Tuple[float, float]:
        """Obtiene las coordenadas (lat, lon) para un nodo."""
        if node_id == 0:
            return (self.depot['y'], self.depot['x'])  # (lat, lon)
        client = next(c for c in self.clients if c['id'] == node_id)
        return (client['y'], client['x'])  # (lat, lon)
    
    def _euclidean_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calcula la distancia euclidiana entre dos coordenadas."""
        return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)
    
    def _initialize_routes(self) -> List[List[int]]:
        """Inicializa las rutas asignando cada cliente a una ruta individual."""
        return [[0, client['id'], 0] for client in self.clients]
    
    def calculate_fitness(self) -> float:
        """Calcula el fitness total considerando todos los factores."""
        total_cost = 0
        
        for route in self.routes:
            route_distance = 0
            for i in range(len(route)-1):
                route_distance += self._distance(route[i], route[i+1])
            
            total_cost += route_distance
            
            # Para Caso2: agregar costo de combustible
            if self.case_type == "caso2":
                total_cost += self._calculate_fuel_cost(route)
                
            # Para Caso3: agregar peajes
            if self.case_type == "caso3":
                total_cost += self._calculate_toll_cost(route)
        
        return total_cost
    
    def _calculate_fuel_cost(self, route: List[int]) -> float:
        """Calcula el costo de combustible para una ruta."""
        if not self.fuel_prices:
            return 0.0
            
        total_fuel_cost = 0
        for i in range(len(route)-1):
            from_id = route[i]
            to_id = route[i+1]
            
            # Obtener la distancia entre los nodos
            distance = self._distance(from_id, to_id)
            
            # Obtener el precio del combustible en el nodo de origen
            if from_id == 0:  # Si es el depósito
                fuel_price = self.fuel_prices.get('depot', 0.0)
            else:
                fuel_price = self.fuel_prices.get(from_id, 0.0)
            
            # Calcular costo de combustible (asumiendo consumo de 10 km/l)
            fuel_cost = (distance / 10.0) * fuel_price
            total_fuel_cost += fuel_cost
            
        return total_fuel_cost
    
    def _calculate_toll_cost(self, route: List[int]) -> float:
        """Calcula el costo de peajes para una ruta."""
        if not self.tolls:
            return 0.0
            
        total_toll_cost = 0
        for i in range(len(route)-1):
            from_id = route[i]
            to_id = route[i+1]
            
            # Buscar el peaje entre estos nodos
            toll_key = (from_id, to_id)
            if toll_key in self.tolls:
                total_toll_cost += self.tolls[toll_key]
            
        return total_toll_cost
    
    def to_verification_csv(self, filename: str):
        """Guarda la solución en formato CSV para verificación."""
        with open(filename, 'w') as f:
            f.write("Route,Node\n")
            for i, route in enumerate(self.routes):
                for node in route:
                    f.write(f"{i+1},{node}\n")

    def is_valid(self) -> bool:
        """Verifica que la solución cumpla todas las restricciones."""
        visited = set()
        for route in self.routes:
            if route[0] != 0 or route[-1] != 0:
                return False
            demand = sum(self._get_demand(c) for c in route[1:-1])
            if demand > self.capacity:
                return False
            for client in route[1:-1]:
                if client in visited:
                    return False
                visited.add(client)
        return len(visited) == len([c for c in self.clients if c['id'] != 0])

    def _get_demand(self, client_id: int) -> float:
        return next(c['demand'] for c in self.clients if c['id'] == client_id)

    def get_route_stats(self) -> Dict:
        """Devuelve estadísticas de carga y distancia por ruta."""
        stats = {'demands': [], 'distances': []}
        for route in self.routes:
            stats['demands'].append(sum(self._get_demand(c) for c in route[1:-1]))
            stats['distances'].append(sum(self._distance(route[i], route[i+1]) for i in range(len(route)-1)))
        return stats