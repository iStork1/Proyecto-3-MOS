from typing import Dict
from solution import CVRPSolution
import matplotlib.pyplot as plt
import numpy as np

class PyomoComparator:
    @staticmethod
    def load_pyomo_results(case_path: str) -> Dict:
        """Carga resultados de Pyomo desde summary.txt y cvrp_solution_report.csv"""
        try:
            # Leer summary.txt
            with open(f"{case_path}/summary.txt", 'r') as f:
                summary = f.read()
                total_distance = float(summary.split("Total distance: ")[1].split("\n")[0])
                num_vehicles = int(summary.split("Number of routes: ")[1].split("\n")[0])
            
            # Leer tiempos (simulado - ajustar según tu implementación real)
            with open(f"{case_path}/logistico_optimizer.log", 'r') as f:
                log = f.read()
                pyomo_time = float(log.split("Tiempo total: ")[1].split("s")[0]) if "Tiempo total:" in log else 0
            
            return {
                'total_distance': total_distance,
                'num_vehicles': num_vehicles,
                'execution_time': pyomo_time
            }
        except:
            return None
    
    @staticmethod
    def compare_results(ga_solution: CVRPSolution, pyomo_data: Dict, ga_time: float, ga_memory: float) -> Dict:
        if not pyomo_data:
            return None
        
        route_stats = ga_solution.get_route_stats()
        
        comparison = {
            'approach': ['GA', 'Pyomo'],
            'distance': [ga_solution.fitness, pyomo_data['total_distance']],
            'num_vehicles': [len(ga_solution.routes), pyomo_data['num_vehicles']],
            'execution_time': [ga_time, pyomo_data['execution_time']],
            'memory_usage_mb': [ga_memory, None],  # Pyomo no medido
            'avg_route_length': [
                ga_solution.fitness / len(ga_solution.routes),
                pyomo_data['total_distance'] / pyomo_data['num_vehicles']
            ],
            'demand_std': [np.std(route_stats['demands']), None],
            'gap_percent': (ga_solution.fitness - pyomo_data['total_distance']) / pyomo_data['total_distance'] * 100
        }
        
        return comparison
    
    @staticmethod
    def plot_comparison(comparison: Dict):
        if not comparison:
            print("No hay datos de Pyomo para comparar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].bar(comparison['approach'], comparison['distance'], color=['blue', 'orange'])
        axes[0, 0].set_title("Distancia Total")
        
        axes[0, 1].bar(comparison['approach'], comparison['num_vehicles'], color=['blue', 'orange'])
        axes[0, 1].set_title("Número de Vehículos")
        
        axes[1, 0].bar(comparison['approach'], comparison['execution_time'], color=['blue', 'orange'])
        axes[1, 0].set_title("Tiempo de Ejecución (s)")
        
        axes[1, 1].bar(['GAP (%)'], [comparison['gap_percent']], color='green' if comparison['gap_percent'] <= 5 else 'red')
        axes[1, 1].set_title("Diferencia Porcentual (GA vs Pyomo)")
        
        plt.tight_layout()
        plt.show()
