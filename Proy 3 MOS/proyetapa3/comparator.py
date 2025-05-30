import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import os

class PyomoComparator:
    @staticmethod
    def load_pyomo_results(filepath: str) -> Optional[Dict]:
        """Carga resultados de Pyomo desde archivos CSV."""
        try:
            if not os.path.exists(filepath):
                # Intentar con nombres alternativos
                alt_filepath = filepath.replace('caso_base', 'caso1')
                if os.path.exists(alt_filepath):
                    filepath = alt_filepath
                else:
                    print(f"No se encontró el archivo de verificación: {filepath}")
                    return None

            df = pd.read_csv(filepath)
            
            # Determinar el caso basado en el nombre del archivo
            if "caso1" in filepath.lower() or "caso_base" in filepath.lower():
                total_distance = df['TotalDistance'].sum() if 'TotalDistance' in df.columns else df['Distance'].sum()
                num_vehicles = len(df)
                pyomo_time = df['TotalTime'].sum() if 'TotalTime' in df.columns else df['Time'].sum()
            elif "caso2" in filepath.lower():
                total_distance = df['Distance'].sum()
                num_vehicles = len(df)
                pyomo_time = df['Time'].sum()
            elif "caso3" in filepath.lower():
                total_distance = df['Distance'].sum()
                num_vehicles = len(df[df['Distance'] > 0])
                pyomo_time = df['Time'].sum()
            else:
                # Formato genérico
                total_distance = df['Distance'].sum() if 'Distance' in df.columns else df['TotalDistance'].sum()
                num_vehicles = len(df)
                pyomo_time = df['Time'].sum() if 'Time' in df.columns else df['TotalTime'].sum()
            
            return {
                'total_distance': total_distance,
                'num_vehicles': num_vehicles,
                'execution_time': pyomo_time
            }
        except Exception as e:
            print(f"Error al cargar {filepath}: {str(e)}")
            return None

    @staticmethod
    def compare_results(ga_solution, pyomo_data: Optional[Dict], ga_time: float, ga_memory: float) -> Optional[Dict]:
        """Compara resultados del GA con Pyomo."""
        if not pyomo_data:
            print("No hay datos de Pyomo para comparar")
            return None

        return {
            'distance': (ga_solution.fitness, pyomo_data['total_distance']),
            'gap_percent': abs(ga_solution.fitness - pyomo_data['total_distance']) / pyomo_data['total_distance'] * 100,
            'execution_time': (ga_time, pyomo_data['execution_time']),
            'memory_usage_mb': (ga_memory, None),  # Pyomo no reporta uso de memoria
            'num_vehicles': (len(ga_solution.routes), pyomo_data['num_vehicles'])
        }

    @staticmethod
    def plot_comparison(comparison: Dict, output_file: str = 'comparacion_ga_pyomo.png'):
        """Genera gráficos comparativos."""
        if not comparison:
            return

        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparación GA vs Pyomo')

        # Distancia
        ax1.bar(['GA', 'Pyomo'], comparison['distance'])
        ax1.set_title('Distancia Total')
        ax1.set_ylabel('Kilómetros')

        # Tiempo de ejecución
        ax2.bar(['GA', 'Pyomo'], comparison['execution_time'])
        ax2.set_title('Tiempo de Ejecución')
        ax2.set_ylabel('Segundos')

        # Número de vehículos
        ax3.bar(['GA', 'Pyomo'], comparison['num_vehicles'])
        ax3.set_title('Número de Vehículos')
        ax3.set_ylabel('Cantidad')

        # Gap porcentual
        ax4.bar(['Gap'], [comparison['gap_percent']])
        ax4.set_title('Gap Porcentual')
        ax4.set_ylabel('Porcentaje')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()