from solution import CVRPSolution
import random

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
            routes=new_routes
        )