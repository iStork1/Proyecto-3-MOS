from typing import Dict
from data_loader import DataLoader
import tracemalloc
import time
import sys
sys.path.append(r'c:\Users\iStork\Desktop\Proy 3 MOS\proyetapa3')
from solution import CVRPSolution
from visualization import Visualizer
from comparator import PyomoComparator
from genetic_algorithm import GeneticAlgorithm

def run_case(case_name: str, case_path: str, config: Dict) -> Dict:
    """Ejecuta GA y compara con Pyomo para un caso específico"""
    print(f"\n{'='*50}")
    print(f"Ejecutando caso: {case_name}")
    print(f"{'='*50}")
    
    # Cargar datos
    clients, capacity, depot = DataLoader.load_data(case_path)
    
    # Ejecutar GA con medición de tiempo/memoria
    tracemalloc.start()
    start_time = time.time()
    
    ga = GeneticAlgorithm(
        clients=clients,
        depot=depot,
        vehicle_capacity=capacity,
        pop_size=config['pop_size'],
        generations=config['generations'],
        mutation_rate=config['mutation_rate'],
        local_search_rate=config['local_search_rate']
    )
    
    best_solution = ga.run()
    ga_time = time.time() - start_time
    _, ga_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Generar archivo de verificación
    best_solution.to_verification_csv(f"verificacion_metaheuristica_GA_{case_name}.csv")
    
    # Visualizar resultados
    Visualizer.plot_solution(best_solution, f"Mejor Solución - {case_name}")
    Visualizer.plot_convergence(ga.history)
    
    # Comparar con Pyomo
    pyomo_data = PyomoComparator.load_pyomo_results(case_path)
    comparison = PyomoComparator.compare_results(best_solution, pyomo_data, ga_time, ga_memory / 10**6)
    
    if comparison:
        PyomoComparator.plot_comparison(comparison)
        
        print("\nResumen Comparativo:")
        print(f"  - Distancia GA: {comparison['distance'][0]:.2f} vs Pyomo: {comparison['distance'][1]:.2f}")
        print(f"  - Gap: {comparison['gap_percent']:.2f}%")
        print(f"  - Tiempo GA: {comparison['execution_time'][0]:.2f}s vs Pyomo: {comparison['execution_time'][1]:.2f}s")
        print(f"  - Memoria GA: {comparison['memory_usage_mb'][0]:.2f} MB")
        print(f"  - Vehículos GA: {comparison['num_vehicles'][0]} vs Pyomo: {comparison['num_vehicles'][1]}")
    
    return {
        'case_name': case_name,
        'ga_fitness': best_solution.fitness,
        'ga_time': ga_time,
        'ga_memory_mb': ga_memory / 10**6,
        'pyomo_fitness': pyomo_data['total_distance'] if pyomo_data else None,
        'pyomo_time': pyomo_data['execution_time'] if pyomo_data else None,
        'comparison': comparison
    }

def main():
    # Configuración del algoritmo (puede calibrarse)
    config = {
        'pop_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'local_search_rate': 0.3
    }
    
    # Definir casos a ejecutar
    cases = {
        "Caso_Base": "./BaseData",
        "Caso2": "./data",  # Ajustar rutas según tu estructura real
        "Caso3": "./data"
    }
    
    # Ejecutar todos los casos y recolectar resultados
    scalability_results = {}
    for case_name, case_path in cases.items():
        case_result = run_case(case_name, case_path, config)
        scalability_results[case_name] = {
            'ga_time': case_result['ga_time'],
            'pyomo_time': case_result['pyomo_time'],
            'ga_fitness': case_result['ga_fitness'],
            'pyomo_fitness': case_result['pyomo_fitness']
        }
    
    # Visualizar escalabilidad
    Visualizer.plot_scalability(scalability_results)

if __name__ == "__main__":
    main()
