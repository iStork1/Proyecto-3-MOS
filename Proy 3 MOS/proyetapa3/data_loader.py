import csv
from typing import List, Dict, Tuple
import os

class DataLoader:
    @staticmethod
    def load_data(case_path: str, case_name: str) -> Tuple[List[Dict], float, Dict, Dict]:
        """Carga datos según el caso especificado y devuelve parámetros adicionales"""
        params = {}
        
        if case_name == "Caso1":
            # Caso base (CVRP estándar)
            clients = DataLoader._load_clients(os.path.join(case_path, "clients.csv"))
            capacity = DataLoader._load_vehicle_capacity(os.path.join(case_path, "vehicles.csv"))
            depot = DataLoader._load_depot(os.path.join(case_path, "depots.csv"))
            
        elif case_name == "Caso2":
            # Caso intermedio (con recarga)
            clients = DataLoader._load_clients(os.path.join(case_path, "caso2_clientes.csv"))
            capacity = DataLoader._load_vehicle_capacity(os.path.join(case_path, "caso2_vehiculos.csv"))
            depot = DataLoader._load_depot(os.path.join(case_path, "caso2_depositos.csv"))
            # Asumo que las estaciones de recarga también están en este archivo
            params["fuel_prices"] = DataLoader._load_fuel_prices(os.path.join(case_path, "caso2_estaciones.csv"))
            
        elif case_name == "Caso3":
            # Caso complejo (con peajes y peso)
            clients = DataLoader._load_clients(os.path.join(case_path, "caso3_clientes.csv"))
            capacity = DataLoader._load_vehicle_capacity(os.path.join(case_path, "caso3_vehiculos.csv"))
            depot = DataLoader._load_depot(os.path.join(case_path, "caso3_depositos.csv"))
            params.update({
                "fuel_prices": DataLoader._load_fuel_prices(os.path.join(case_path, "caso3_estaciones.csv")),
                "tolls": DataLoader._load_tolls(os.path.join(case_path, "caso3_peajes.csv"))
            })
            
        else:
            raise ValueError(f"Caso no reconocido: {case_name}")

        return clients, capacity, depot, params

    @staticmethod
    def _load_clients(filepath: str) -> List[Dict]:
        """Carga datos de clientes con manejo de diferentes formatos."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            clients = []
            for row in reader:
                client = {
                    'id': int(row.get('ClientID', row.get('ID', 0))),
                    'x': float(row.get('Longitude', row.get('X', 0))),
                    'y': float(row.get('Latitude', row.get('Y', 0))),
                    'demand': float(row.get('Demand', row.get('Demanda', 0)))
                }
                clients.append(client)
            # Agregar el depósito como cliente con demanda 0 y ID 0 para uniformidad
            # clients.append({'id': 0, 'x': depot['x'], 'y': depot['y'], 'demand': 0}) # Esto se maneja mejor por separado
            return clients

    @staticmethod
    def _load_vehicle_capacity(filepath: str) -> float:
        """Carga capacidad del vehículo con manejo de diferentes formatos."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            row = next(reader)
            return float(row.get('Capacity', row.get('Capacidad', 0)))

    @staticmethod
    def _load_depot(filepath: str) -> Dict:
        """Carga datos del depósito con manejo de diferentes formatos."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            depot = next(reader)
            return {
                'id': 0,
                'x': float(depot.get('Longitude', depot.get('X', 0))),
                'y': float(depot.get('Latitude', depot.get('Y', 0)))
            }

    @staticmethod
    def _load_fuel_prices(filepath: str) -> Dict[str, float]:
        """Carga precios de combustible desde CSV"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo de precios no encontrado: {filepath}")
        
        prices = {}
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Asumo que el CSV tiene columnas 'StationID' y 'FuelPrice'
                if 'StationID' in row and 'FuelPrice' in row:
                     prices[row['StationID']] = float(row['FuelPrice'])
                else:
                    print(f"Advertencia: Archivo {filepath} no contiene columnas StationID o FuelPrice")
                    return {}
        return prices

    @staticmethod
    def _load_tolls(filepath: str) -> Dict[Tuple[str, str], float]:
        """Carga peajes desde CSV"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo de peajes no encontrado: {filepath}")
        
        tolls = {}
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Asumo que el CSV tiene columnas 'From', 'To' y 'Cost'
                if 'From' in row and 'To' in row and 'Cost' in row:
                    tolls[(row['From'], row['To'])] = float(row['Cost'])
                else:
                     print(f"Advertencia: Archivo {filepath} no contiene columnas From, To o Cost")
                     return {}

        return tolls