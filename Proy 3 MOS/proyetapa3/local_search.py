from solution import CVRPSolution
import random
from typing import List

class LocalSearch:
    @staticmethod
    def two_opt(solution: CVRPSolution, **solution_kwargs) -> CVRPSolution:
        """Aplica búsqueda local 2-opt a cada ruta de la solución."""
        improved = True
        new_routes = [route.copy() for route in solution.routes]

        # Continuar mejorando mientras se encuentren mejoras en alguna ruta
        while improved:
            improved = False # Reiniciar bandera
            for route_idx in range(len(new_routes)):
                route = new_routes[route_idx]
                # Solo aplicar 2-opt si la ruta tiene al menos 4 nodos (Depot - C1 - C2 - Depot)
                if len(route) < 4:
                    continue
                    
                best_current_route_distance = sum(solution._distance(route[i], route[i+1]) for i in range(len(route)-1))
                
                # Iterar sobre todos los pares (i, j) en la ruta (excluyendo depósitos)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # No invertir segmentos de longitud 1
                        if j - i < 1: continue

                        # Crear nueva ruta invirtiendo el segmento entre i y j
                        # Ejemplo: A -> B -> C -> D -> E -> F
                        # Si i=B y j=E, el nuevo orden es A -> B -> E -> D -> C -> F
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]

                        # Calcular la distancia de la nueva ruta
                        new_route_distance = sum(solution._distance(new_route[k], new_route[k+1]) for k in range(len(new_route)-1))
                        
                        # Si la nueva ruta es mejor, actualizarla y marcar mejora
                        if new_route_distance < best_current_route_distance:
                            new_routes[route_idx] = new_route
                            best_current_route_distance = new_route_distance
                            improved = True # Se encontró una mejora, seguir iterando
        
        # Crear una nueva solución con las rutas potencialmente mejoradas
        # Es crucial re-evaluar el fitness de la nueva solución COMPLETA
        return CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity,
            routes=new_routes,
            **solution_kwargs # Pasar los kwargs aquí
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