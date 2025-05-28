import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import json
from pathlib import Path
import tracemalloc
import pandas as pd

class DataLoader:
    @staticmethod
    def load_data(case_path: str) -> Tuple[List[Dict], float, Dict]:
        """Carga todos los datos necesarios para un caso"""
        if "BaseData" in case_path:
            clients = DataLoader._load_clients(f"{case_path}/clients.csv")
            capacity = DataLoader._load_vehicle_capacity(f"{case_path}/vehicles.csv")
            depot = DataLoader._load_depot(f"{case_path}/depots.csv")
        else:
            # Para Caso2 y Caso3
            case_num = "caso2" if "Caso2" in case_path else "caso3"
            clients = DataLoader._load_clients(f"{case_path}/{case_num}_clientes.csv")
            capacity = DataLoader._load_vehicle_capacity(f"{case_path}/{case_num}_vehiculos.csv")
            depot = DataLoader._load_depot(f"{case_path}/{case_num}_depositos.csv")
        return clients, capacity, depot

    @staticmethod
    def _load_clients(filepath: str) -> List[Dict]:
        clients = []
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                clients.append({
                    'id': int(row['ClientID']),
                    'x': float(row['Longitude']),
                    'y': float(row['Latitude']),
                    'demand': float(row['Demand'])
                })
        return clients

    @staticmethod
    def _load_vehicle_capacity(filepath: str) -> float:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            return float(next(reader)['Capacity'])  # Asumiendo flota homogénea

    @staticmethod
    def _load_depot(filepath: str) -> Dict:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            depot_data = next(reader)
            return {
                'id': 0,
                'x': float(depot_data['Longitude']),
                'y': float(depot_data['Latitude'])
            }