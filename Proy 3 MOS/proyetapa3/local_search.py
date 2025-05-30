from solution import CVRPSolution
import random
from typing import List

class LocalSearch:
    @staticmethod
    def two_opt(solution: CVRPSolution) -> CVRPSolution:
        improved = True
        new_routes = [route.copy() for route in solution.routes]
        
        while improved:
            improved = False
            for route_idx in range(len(new_routes)):
                route = new_routes[route_idx]
                best_distance = sum(solution._distance(route[i], route[i+1]) for i in range(len(route)-1))
                
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j-i == 1: continue
                        
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_distance = sum(solution._distance(new_route[k], new_route[k+1]) for k in range(len(new_route)-1))
                        
                        if new_distance < best_distance:
                            new_routes[route_idx] = new_route
                            best_distance = new_distance
                            improved = True
        
        return CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity,
            routes=new_routes,
            **{k: v for k, v in solution.__dict__.items() if k not in ['clients', 'depot', 'capacity', 'routes', 'fitness', '_distance_matrix']}
        )

def local_search(routes: List[List[int]], solution) -> List[List[int]]:
    """Aplica búsqueda local a las rutas."""
    # Seleccionar una ruta aleatoria
    route_idx = random.randrange(len(routes))
    route = routes[route_idx]
    
    # Si la ruta es muy corta, no hacer nada
    if len(route) <= 3:
        return routes
    
    # Intentar intercambiar dos clientes aleatorios
    pos1, pos2 = random.sample(range(1, len(route)-1), 2)
    
    # Calcular la distancia actual
    current_distance = 0
    for i in range(len(route)-1):
        current_distance += solution._distance(route[i], route[i+1])
    
    # Intercambiar y calcular nueva distancia
    route[pos1], route[pos2] = route[pos2], route[pos1]
    new_distance = 0
    for i in range(len(route)-1):
        new_distance += solution._distance(route[i], route[i+1])
    
    # Si la nueva distancia es peor, revertir el cambio
    if new_distance > current_distance:
        route[pos1], route[pos2] = route[pos2], route[pos1]
    
    return routes