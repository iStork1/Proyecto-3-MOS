from typing import List, Dict, Callable, Optional
from solution import CVRPSolution
from genetic_operators import crossover, mutation
from local_search import local_search
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
        """Crea la población inicial."""
        population = []
        for _ in range(self.pop_size):
            solution = CVRPSolution(
                clients=self.clients,
                depot=self.depot,
                vehicle_capacity=self.vehicle_capacity,
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
            
            # Elitismo: mantener la mejor solución
            best_solution = min(population, key=lambda x: x.fitness)
            new_population.append(best_solution)
            
            # Generar resto de la población
            while len(new_population) < self.pop_size:
                # Seleccionar padres
                parents = self._select_parents(population)
                
                # Cruzar
                child_routes = crossover(parents[0].routes, parents[1].routes)
                
                # Mutar
                if random.random() < self.mutation_rate:
                    child_routes = mutation(child_routes)
                
                # Crear nueva solución
                child = CVRPSolution(
                    clients=self.clients,
                    depot=self.depot,
                    vehicle_capacity=self.vehicle_capacity,
                    routes=child_routes,
                    **solution_kwargs
                )
                
                # Búsqueda local
                if random.random() < self.local_search_rate:
                    child.routes = local_search(child.routes, child)
                
                new_population.append(child)
            
            # Actualizar población
            population = new_population
            
            # Calcular estadísticas
            best_fitness = min(p.fitness for p in population)
            avg_fitness = sum(p.fitness for p in population) / len(population)
            
            # Guardar en historial
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Llamar callback si existe
            if callback:
                callback(gen, best_fitness, avg_fitness)
        
        # Retornar mejor solución
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