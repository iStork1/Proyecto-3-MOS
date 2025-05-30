import pandas as pd
import time
import tracemalloc
from typing import Dict
from data_loader import DataLoader
from genetic_algorithm import GeneticAlgorithm
from solution import CVRPSolution
from visualization import Visualizer
from comparator import PyomoComparator
from distance_service import DistanceService
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def run_case(case_name: str, case_path: str, config: Dict) -> Dict:
    """Ejecuta GA y compara con Pyomo para un caso."""
    print(f"\n{'='*50}")
    print(f"Ejecutando caso: {case_name}")
    print(f"{'='*50}")

    try:
        # Inicializar el servicio de distancias
        distance_service = DistanceService()
        
        # Cargar datos con parámetros adicionales según el caso
        clients, capacity, depot, params = DataLoader.load_data(case_path, case_name)

        # Configurar semilla para reproducibilidad
        random.seed(config['seed'])
        np.random.seed(config['seed'])

        # Ejecutar GA con los parámetros del caso
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
        
        # Configurar callback para mostrar información detallada
        def print_generation_info(gen, best_fitness, avg_fitness):
            print(f"Gen {gen}: Best = {best_fitness:.2f} km, Avg = {avg_fitness:.2f} km")
        
        # Preparar parámetros adicionales para CVRPSolution
        solution_kwargs = {
            'case_type': case_name.lower(),
            'distance_service': distance_service
        }
        if 'fuel_prices' in params:
            solution_kwargs['fuel_prices'] = params['fuel_prices']
        if 'tolls' in params:
            solution_kwargs['tolls'] = params['tolls']

        # Ejecutar GA con callback solo para Caso3
        callback_func = print_generation_info if case_name == "Caso3" else None
        
        best_solution = ga.run(
            callback=callback_func,
            solution_kwargs=solution_kwargs
        )
        
        ga_time = time.time() - start_time
        _, ga_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\nMejor solución encontrada:")
        print(f"  - Distancia total: {best_solution.fitness:.2f} km")
        print(f"  - Número de vehículos: {len(best_solution.routes)}")
        print(f"  - Tiempo de ejecución: {ga_time:.2f} segundos")
        print(f"  - Memoria utilizada: {ga_memory / 10**6:.2f} MB")

        # Crear directorio de salida si no existe
        os.makedirs("output", exist_ok=True)

        # Guardar gráficas
        plot_file = os.path.join("output", f"rutas_{case_name}.png")
        Visualizer.plot_solution(best_solution, f"Mejor Solución - {case_name}")
        plt.savefig(plot_file)
        plt.close()
        print(f"\nGráfica de rutas guardada como: {plot_file}")

        # Gráfica de convergencia
        convergence_file = os.path.join("output", f"convergencia_{case_name}.png")
        Visualizer.plot_convergence(ga.history)
        plt.savefig(convergence_file)
        plt.close()
        print(f"Gráfica de convergencia guardada como: {convergence_file}")

        # Guardar archivo de verificación del GA
        ga_verification_file = os.path.join("output", f"verificacion_metaheuristica_GA_{case_name}.csv")
        best_solution.to_verification_csv(ga_verification_file)
        print(f"\nArchivo de verificación del GA guardado: {ga_verification_file}")

        # Cargar resultados de Pyomo desde el directorio output
        pyomo_file = os.path.join("output", f"verificacion_{case_name.lower()}.csv")
        print(f"Buscando archivo de verificación de Pyomo en: {pyomo_file}")
        pyomo_data = PyomoComparator.load_pyomo_results(pyomo_file)

        # Comparar resultados
        comparison = PyomoComparator.compare_results(best_solution, pyomo_data, ga_time, ga_memory / 10**6)
        
        if comparison:
            # Guardar gráfico de comparación
            plot_file = os.path.join("output", f"comparacion_ga_pyomo_{case_name}.png")
            PyomoComparator.plot_comparison(comparison, plot_file)
            print(f"\nGráfico de comparación guardado como: {plot_file}")

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

    except FileNotFoundError as e:
        print(f"\nError: No se encontró el archivo necesario: {str(e)}")
        return None
    except Exception as e:
        print(f"\nError al procesar el caso {case_name}: {str(e)}")
        return None

def main():
    # Configuración del GA (ajustable)
    config = {
        'pop_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'local_search_rate': 0.3,
        'seed': 42  # Semilla fija para reproducibilidad
    }

    # Casos a ejecutar
    cases = {
        "Caso1": "./BaseData",  # Caso base
        "Caso2": "./data",      # Caso 2 (archivos en data/)
        "Caso3": "./data"       # Caso 3 (archivos en data/)
    }

    # Crear directorio de salida si no existe
    os.makedirs("output", exist_ok=True)

    # Ejecutar todos los casos
    results = []
    for case_name, case_path in cases.items():
        print(f"\nProcesando caso: {case_name}")
        print(f"Ruta de datos: {case_path}")
        
        # Verificar que el directorio existe
        if not os.path.exists(case_path):
            print(f"Error: El directorio de datos {case_path} no existe")
            continue
            
        result = run_case(case_name, case_path, config)
        if result:
            results.append(result)

    if results:
        # Generar reporte
        df = pd.DataFrame(results)
        output_file = os.path.join("output", "resultados_finales.csv")
        df.to_csv(output_file, index=False)
        print(f"\nResultados guardados en '{output_file}'")
    else:
        print("\nNo se pudo generar ningún resultado.")

if __name__ == "__main__":
    main()