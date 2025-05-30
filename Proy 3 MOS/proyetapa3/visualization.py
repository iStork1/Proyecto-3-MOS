import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from solution import CVRPSolution

class Visualizer:
    @staticmethod
    def plot_solution(solution: CVRPSolution, title: str):
        """Visualiza las rutas de la solución."""
        plt.figure(figsize=(12, 8))
        
        # Coordenadas del depósito
        depot_coords = solution._get_coords(0) # (lat, lon)
        plt.plot(depot_coords[1], depot_coords[0], 's', color='red', markersize=10, label='Depósito') # plot (lon, lat)
        
        # Coordenadas de los clientes
        client_coords = {c['id']: (c['y'], c['x']) for c in solution.clients} # {id: (lat, lon)}
        for client_id, coords in client_coords.items():
            plt.plot(coords[1], coords[0], 'o', color='blue', markersize=7) # plot (lon, lat)
            plt.text(coords[1], coords[0], str(client_id), fontsize=9, ha='right')
            
        # Dibujar rutas
        colors = plt.cm.get_cmap('tab10', len(solution.routes))
        for i, route in enumerate(solution.routes):
            route_coords = []
            for node_id in route:
                if node_id == 0:
                    route_coords.append(depot_coords)
                else:
                    route_coords.append(client_coords[node_id])
            
            # Extraer longitudes y latitudes
            lons = [coord[1] for coord in route_coords]
            lats = [coord[0] for coord in route_coords]
            
            plt.plot(lons, lats, color=colors(i), marker='o', linestyle='-', linewidth=1.5, label=f'Ruta {i+1}')
        
        plt.title(title)
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Asegura que las escalas de los ejes sean iguales
        plt.tight_layout() # Ajusta el layout para evitar solapamiento
        # plt.show() # Eliminar si solo se guarda en archivo

    @staticmethod
    def plot_convergence(history):
        """Visualiza la curva de convergencia (fitness a lo largo de las generaciones)."""
        generations = [entry['generation'] for entry in history]
        best_fitness = [entry['best_fitness'] for entry in history]
        avg_fitness = [entry['avg_fitness'] for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label='Mejor Fitness', marker='.')
        plt.plot(generations, avg_fitness, label='Fitness Promedio', marker='.')
        
        plt.title('Convergencia del Algoritmo Genético')
        plt.xlabel('Generación')
        plt.ylabel('Fitness (km)') # Asegurar la unidad correcta
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout
        # plt.show() # Eliminar si solo se guarda en archivo

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