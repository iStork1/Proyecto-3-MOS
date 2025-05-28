from typing import List, Dict
from solution import CVRPSolution
from genetic_operators import GeneticOperators
from local_search import LocalSearch
import random

class GeneticAlgorithm:
    def __init__(self, clients: List[Dict], depot: Dict, vehicle_capacity: float,
                 pop_size: int = 50, elite_size: int = 5, mutation_rate: float = 0.1,
                 generations: int = 100, local_search_rate: float = 0.2):
        self.clients = clients
        self.depot = depot
        self.capacity = vehicle_capacity
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.local_search_rate = local_search_rate
        self.history = []
    
    def run(self) -> CVRPSolution:
        population = self._initialize_population()
        
        for gen in range(self.generations):
            population = self._evolve(population)
            
            best = min(population, key=lambda x: x.fitness)
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.history.append({
                'generation': gen,
                'best_fitness': best.fitness,
                'avg_fitness': avg_fitness
            })
            
            print(f"Gen {gen}: Best = {best.fitness:.2f}, Avg = {avg_fitness:.2f}")
        
        return min(population, key=lambda x: x.fitness)
    
    def _initialize_population(self) -> List[CVRPSolution]:
        return [
            CVRPSolution(self.clients, self.depot, self.capacity)
            for _ in range(self.pop_size)
        ]
    
    def _evolve(self, population: List[CVRPSolution]) -> List[CVRPSolution]:
        population.sort(key=lambda x: x.fitness)
        elites = population[:self.elite_size]
        
        children = []
        while len(children) < self.pop_size - self.elite_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            child1, child2 = GeneticOperators.pmx_crossover(parent1, parent2)
            
            if random.random() < self.mutation_rate:
                child1 = GeneticOperators.swap_mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = GeneticOperators.inversion_mutation(child2)
            
            if random.random() < self.local_search_rate:
                child1 = LocalSearch.two_opt(child1)
            if random.random() < self.local_search_rate:
                child2 = LocalSearch.two_opt(child2)
            
            children.extend([child1, child2])
        
        num_new = int(0.1 * self.pop_size)
        new_individuals = [
            CVRPSolution(self.clients, self.depot, self.capacity)
            for _ in range(num_new)
        ]
        
        return elites + children[:self.pop_size - self.elite_size - num_new] + new_individuals
    
    def _tournament_selection(self, population: List[CVRPSolution], k: int = 3) -> CVRPSolution:
        contestants = random.sample(population, k)
        return min(contestants, key=lambda x: x.fitness)