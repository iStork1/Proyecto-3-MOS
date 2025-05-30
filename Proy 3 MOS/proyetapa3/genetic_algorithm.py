from typing import List, Dict, Callable, Optional
from solution import CVRPSolution
from genetic_operators import GeneticOperators
from local_search import LocalSearch
import random
import numpy as np
"""
Algoritmo Genético para CVRP:
- Representación: Lista de rutas (cromosoma).
- Operadores:
  * Cruce PMX: Preserva orden parcial entre padres.
  * Mutación por intercambio/inversión.
  * Reparación para factibilidad (capacidad y visitas únicas).
- Selección: Torneo (k=3).
- Búsqueda Local: 2-opt para refinamiento.
"""
class GeneticAlgorithm:
    def __init__(self, clients: List[Dict], depot: Dict, vehicle_capacity: float,
                 pop_size: int = 50, elite_size: int = 5, mutation_rate: float = 0.1,
                 generations: int = 100, local_search_rate: float = 0.3):
        self.clients = clients
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.local_search_rate = local_search_rate
        self.history = []
    
    def _create_initial_population(self, solution_kwargs: Dict) -> List[CVRPSolution]:
        """Crea la población inicial con rutas agrupadas aleatoriamente respetando capacidad."""
        population = []
        client_ids = [client['id'] for client in self.clients]
        
        for _ in range(self.pop_size):
            shuffled_clients = random.sample(client_ids, len(client_ids)) # Shuffle client IDs
            routes = []
            current_route = [0] # Start at depot
            current_load = 0
            
            for client_id in shuffled_clients:
                # Find client object to get demand
                client = next(c for c in self.clients if c['id'] == client_id)
                demand = client['demand']
                
                if current_load + demand <= self.vehicle_capacity:
                    current_route.append(client_id)
                    current_load += demand
                else:
                    # Finish current route and start a new one
                    current_route.append(0) # Return to depot
                    routes.append(current_route)
                    current_route = [0, client_id] # Start new route with this client
                    current_load = demand
                    
            # Add the last route if it's not empty
            if len(current_route) > 1: # Check if it contains more than just the depot
                current_route.append(0) # Return to depot
                routes.append(current_route)
            
            solution = CVRPSolution(
                clients=self.clients,
                depot=self.depot,
                vehicle_capacity=self.vehicle_capacity,
                routes=routes, # Use the generated routes
                **solution_kwargs
            )
            population.append(solution)
            
        return population
    
    def _select_parents(self, population: List[CVRPSolution]) -> List[CVRPSolution]:
        """Selecciona padres usando torneo binario."""
        parents = []
        for _ in range(2):
            # Seleccionar dos soluciones aleatorias
            candidates = random.sample(population, 2)
            # Seleccionar la mejor
            winner = min(candidates, key=lambda x: x.fitness)
            parents.append(winner)
        return parents
    
    def run(self, callback: Optional[Callable] = None, solution_kwargs: Dict = None) -> CVRPSolution:
        """Ejecuta el algoritmo genético."""
        if solution_kwargs is None:
            solution_kwargs = {}

        # Crear población inicial
        population = self._create_initial_population(solution_kwargs)

        # Ejecutar generaciones
        for gen in range(self.generations):
            # Crear nueva población
            new_population = []

            # Elitismo: mantener la mejor solución (copia para no modificar el original en futuras operaciones)
            best_solution = min(population, key=lambda x: x.fitness)
            # Añadir una copia para asegurar que la mejor solución no sea modificada in-place
            elite_copy = CVRPSolution(
                clients=self.clients,
                depot=self.depot,
                vehicle_capacity=self.vehicle_capacity,
                routes=[route.copy() for route in best_solution.routes], # Copia profunda de las rutas
                **solution_kwargs # Pasar los kwargs
            )
            new_population.append(elite_copy)

            # Generar resto de la población
            while len(new_population) < self.pop_size:
                # Seleccionar padres usando torneo
                parents = self._select_parents(population)

                # Cruzar para obtener dos hijos
                child1, child2 = GeneticOperators.ordered_crossover(parents[0], parents[1], **solution_kwargs)

                # Aplicar mutación con la tasa especificada a cada hijo
                if random.random() < self.mutation_rate:
                    child1 = GeneticOperators.swap_mutation(child1, **solution_kwargs) # Usar swap_mutation
                    # child1 = GeneticOperators.inversion_mutation(child1, **solution_kwargs) # Opcional: usar inversion_mutation

                if random.random() < self.mutation_rate:
                    child2 = GeneticOperators.swap_mutation(child2, **solution_kwargs) # Usar swap_mutation
                    # child2 = GeneticOperators.inversion_mutation(child2, **solution_kwargs) # Opcional: usar inversion_mutation

                # Aplicar búsqueda local a los hijos con la tasa especificada
                if random.random() < self.local_search_rate:
                     child1 = LocalSearch.two_opt(child1, **solution_kwargs) # Usar LocalSearch.two_opt

                if random.random() < self.local_search_rate:
                     child2 = LocalSearch.two_opt(child2, **solution_kwargs) # Usar LocalSearch.two_opt

                # Añadir hijos a la nueva población (verificando capacidad si aplica, aunque _split_to_routes debería manejarlo)
                # Verificar validez y añadir si es válida o si se permite un porcentaje de inválidas
                # Por ahora, simplemente añadimos. La penalización en fitness se encarga de las inválidas.
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                     new_population.append(child2)

            # Reemplazar la población vieja por la nueva
            population = new_population

            # Calcular estadísticas para la generación actual
            valid_solutions = [p for p in population if p.is_valid()]
            if valid_solutions:
                 best_fitness = min(p.fitness for p in valid_solutions)
                 avg_fitness = sum(p.fitness for p in valid_solutions) / len(valid_solutions)
            else:
                 # Si no hay soluciones válidas, reportar fitness infinito
                 best_fitness = float('inf')
                 avg_fitness = float('inf')

            # Guardar en historial
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })

            # Llamar callback si existe
            if callback:
                callback(gen, best_fitness, avg_fitness)

        # Retornar la mejor solución válida encontrada en la última generación
        # Si no hay ninguna válida, retornamos la mejor (probablemente inf fitness)
        valid_solutions = [p for p in population if p.is_valid()]
        if valid_solutions:
             return min(valid_solutions, key=lambda x: x.fitness)
        else:
             # Si no se encontró ninguna solución válida, retornar la mejor no válida (tendrá inf fitness)
             return min(population, key=lambda x: x.fitness)
    
    def _initialize_population(self) -> List[CVRPSolution]:
        return [
            CVRPSolution(self.clients, self.depot, self.vehicle_capacity)
            for _ in range(self.pop_size)
        ]
    
    def _evolve(self, population: List[CVRPSolution]) -> List[CVRPSolution]:
        population.sort(key=lambda x: x.fitness)
        elites = population[:self.elite_size]
        
        children = []
        while len(children) < self.pop_size - self.elite_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            child1, child2 = crossover(parent1.routes, parent2.routes)
            
            if random.random() < self.mutation_rate:
                child1 = mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = mutation(child2)
            
            if random.random() < self.local_search_rate:
                child1 = local_search(child1, child1)
            if random.random() < self.local_search_rate:
                child2 = local_search(child2, child2)
            
            children.extend([child1, child2])
        
        num_new = int(0.1 * self.pop_size)
        new_individuals = [
            CVRPSolution(self.clients, self.depot, self.vehicle_capacity)
            for _ in range(num_new)
        ]
        
        return elites + children[:self.pop_size - self.elite_size - num_new] + new_individuals
    
    def _tournament_selection(self, population: List[CVRPSolution], k: int = 3) -> CVRPSolution:
        contestants = random.sample(population, k)
        return min(contestants, key=lambda x: x.fitness)