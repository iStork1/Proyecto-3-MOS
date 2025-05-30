import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

class Visualizer:
    @staticmethod
    def plot_solution(solution, title: str = "Mejor Solución"):
        """Visualiza las rutas en un mapa."""
        plt.figure(figsize=(10, 8))
        
        # Coordenadas del depósito
        depot_x, depot_y = solution.depot['x'], solution.depot['y']
        plt.scatter(depot_x, depot_y, c='red', marker='s', s=100, label='Depósito')
        
        # Coordenadas de clientes
        clients_x = [c['x'] for c in solution.clients if c['id'] != 0]
        clients_y = [c['y'] for c in solution.clients if c['id'] != 0]
        plt.scatter(clients_x, clients_y, c='blue', label='Clientes')
        
        # Dibujar rutas
        colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
        for i, route in enumerate(solution.routes):
            route_x = [depot_x if node == 0 else 
                      next(c['x'] for c in solution.clients if c['id'] == node) for node in route]
            route_y = [depot_y if node == 0 else 
                      next(c['y'] for c in solution.clients if c['id'] == node) for node in route]
            plt.plot(route_x, route_y, '--', color=colors[i], alpha=0.7, linewidth=2, label=f'Ruta {i+1}')
        
        plt.title(f"{title}\nDistancia Total: {solution.fitness:.2f} km")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_convergence(history: List[Dict]):
        """Grafica la convergencia del GA."""
        gens = [h['generation'] for h in history]
        best = [h['best_fitness'] for h in history]
        avg = [h['avg_fitness'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gens, best, 'b-', label='Mejor Fitness')
        plt.plot(gens, avg, 'r--', label='Fitness Promedio')
        plt.title("Convergencia del Algoritmo Genético")
        plt.xlabel("Generación")
        plt.ylabel("Distancia Total (km)")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_scalability(results: List[Dict]):
        """Compara escalabilidad entre casos."""
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de tiempos
        sns.barplot(data=df, x='case_name', y='ga_time', ax=ax1, color='blue', label='GA')
        if 'pyomo_time' in df.columns:
            sns.barplot(data=df, x='case_name', y='pyomo_time', ax=ax1, color='orange', label='Pyomo')
        ax1.set_title("Tiempo de Ejecución por Caso")
        ax1.set_ylabel("Segundos")
        ax1.legend()
        
        # Gráfico de distancias
        sns.barplot(data=df, x='case_name', y='ga_fitness', ax=ax2, color='blue', label='GA')
        if 'pyomo_fitness' in df.columns:
            sns.barplot(data=df, x='case_name', y='pyomo_fitness', ax=ax2, color='orange', label='Pyomo')
        ax2.set_title("Distancia Total por Caso")
        ax2.set_ylabel("Distancia (km)")
        ax2.legend()
        
        plt.tight_layout()