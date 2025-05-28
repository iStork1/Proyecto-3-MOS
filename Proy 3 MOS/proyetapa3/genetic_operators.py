from typing import Tuple, List
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
        
        child1 = GeneticOperators._pmx_helper(flat_parent1, flat_parent2, point1, point2)
        child2 = GeneticOperators._pmx_helper(flat_parent2, flat_parent1, point1, point2)
        
        child1_sol = GeneticOperators._split_to_routes(child1, parent1)
        child2_sol = GeneticOperators._split_to_routes(child2, parent2)
        
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
    def _split_to_routes(flat_solution: List[int], original_sol: CVRPSolution) -> CVRPSolution:
        routes = []
        current_route = [0]
        current_load = 0
        
        for client in flat_solution:
            demand = original_sol._get_demand(client)
            
            if current_load + demand <= original_sol.capacity:
                current_route.append(client)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, client]
                current_load = demand
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        return CVRPSolution(
            clients=original_sol.clients,
            depot=original_sol.depot,
            vehicle_capacity=original_sol.capacity,
            routes=routes
        )
    
    @staticmethod
    def swap_mutation(solution: CVRPSolution) -> CVRPSolution:
        route1_idx = random.randint(0, len(solution.routes)-1)
        route2_idx = random.randint(0, len(solution.routes)-1)
        
        route1 = solution.routes[route1_idx]
        route2 = solution.routes[route2_idx]
        
        if len(route1) > 2:
            client1_idx = random.randint(1, len(route1)-2)
        else:
            return solution
        
        if len(route2) > 2:
            client2_idx = random.randint(1, len(route2)-2)
        else:
            return solution
        
        new_routes = [route.copy() for route in solution.routes]
        new_routes[route1_idx][client1_idx], new_routes[route2_idx][client2_idx] = (
            new_routes[route2_idx][client2_idx], new_routes[route1_idx][client1_idx]
        )
        
        new_solution = CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity,
            routes=new_routes
        )
        
        return GeneticOperators._repair_solution(new_solution)
    
    @staticmethod
    def inversion_mutation(solution: CVRPSolution) -> CVRPSolution:
        route_idx = random.randint(0, len(solution.routes)-1)
        route = solution.routes[route_idx]
        
        if len(route) <= 3:
            return solution
        
        start = random.randint(1, len(route)-3)
        end = random.randint(start+1, len(route)-2)
        
        new_routes = [r.copy() for r in solution.routes]
        new_routes[route_idx][start:end+1] = reversed(new_routes[route_idx][start:end+1])
        
        new_solution = CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity,
            routes=new_routes
        )
        
        return GeneticOperators._repair_solution(new_solution)
    
    @staticmethod
    def _repair_solution(solution: CVRPSolution) -> CVRPSolution:
        if solution.is_valid():
            return solution
        
        flat_clients = [c for route in solution.routes for c in route[1:-1]]
        unique_clients = list(dict.fromkeys(flat_clients))
        
        return CVRPSolution(
            clients=solution.clients,
            depot=solution.depot,
            vehicle_capacity=solution.capacity
        )