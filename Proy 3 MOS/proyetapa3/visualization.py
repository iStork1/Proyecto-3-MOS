from typing import List, Dict
from solution import CVRPSolution
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_solution(solution: CVRPSolution, title: str = "Mejor Solución"):
        plt.figure(figsize=(10, 8))
        
        x = [c['x'] for c in solution.clients if c['id'] != 0]
        y = [c['y'] for c in solution.clients if c['id'] != 0]
        plt.scatter(x, y, c='blue', label='Clientes')
        plt.scatter(solution.depot['x'], solution.depot['y'], c='red', marker='s', s=100, label='Depósito')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
        for route, color in zip(solution.routes, colors):
            route_x = [solution.depot['x'] if c == 0 else 
                      next(cl['x'] for cl in solution.clients if cl['id'] == c) for c in route]
            route_y = [solution.depot['y'] if c == 0 else 
                      next(cl['y'] for cl in solution.clients if cl['id'] == c) for c in route]
            plt.plot(route_x, route_y, '--', color=color, alpha=0.7, linewidth=2)
            plt.scatter(route_x[1:-1], route_y[1:-1], color=color, s=50)
        
        plt.title(f"{title}\nDistancia Total: {solution.fitness:.2f}")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_convergence(history: List[Dict]):
        gens = [h['generation'] for h in history]
        best = [h['best_fitness'] for h in history]
        avg = [h['avg_fitness'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gens, best, 'b-', label='Mejor Fitness')
        plt.plot(gens, avg, 'r--', label='Fitness Promedio')
        plt.title("Convergencia del Algoritmo Genético")
        plt.xlabel("Generación")
        plt.ylabel("Distancia Total")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_scalability(results: Dict):
        cases = list(results.keys())
        ga_times = [results[case]['ga_time'] for case in cases]
        pyomo_times = [results[case]['pyomo_time'] if results[case]['pyomo_time'] is not None else 0 for case in cases]
        ga_fitness = [results[case]['ga_fitness'] for case in cases]
        pyomo_fitness = [results[case]['pyomo_fitness'] if results[case]['pyomo_fitness'] is not None else 0 for case in cases]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de tiempos
        ax1.bar(cases, ga_times, width=0.4, label='GA', align='center')
        if any(t > 0 for t in pyomo_times):  # Solo mostrar Pyomo si hay datos
            ax1.bar(cases, pyomo_times, width=0.4, label='Pyomo', align='edge')
        ax1.set_title("Tiempo de Ejecución por Caso")
        ax1.set_ylabel("Segundos")
        ax1.legend()
        
        # Gráfico de fitness
        ax2.bar(cases, ga_fitness, width=0.4, label='GA', align='center')
        if any(f > 0 for f in pyomo_fitness):  # Solo mostrar Pyomo si hay datos
            ax2.bar(cases, pyomo_fitness, width=0.4, label='Pyomo', align='edge')
        ax2.set_title("Distancia Total por Caso")
        ax2.set_ylabel("Distancia")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
