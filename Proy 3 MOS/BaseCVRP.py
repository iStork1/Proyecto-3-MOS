import pandas as pd
import numpy as np
import openrouteservice
from pyomo.environ import *
import folium
import os
import time
from datetime import timedelta
import logging
import random
from geopy.distance import geodesic
import sys
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
print(f"API Key detectada: {os.getenv('ORS_API_KEY')}")

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logistico_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LogistiCoRouteOptimizer:
    def __init__(self):
        """Inicializa el optimizador con configuración robusta"""
        self.api_key = os.getenv('ORS_API_KEY')
        if not self.api_key:
            logger.error("API key no encontrada. Configúrala en el archivo .env")
            sys.exit(1)
            
        self.client = openrouteservice.Client(key=self.api_key)
        self.max_retries = 7
        self.base_retry_delay = 3
        self.cache_file = 'distance_cache.pkl'
        self.case_data = {
            1: {'name': 'CVRP Estándar', 'model': None, 'solution': None},
            2: {'name': 'Con Recarga', 'model': None, 'solution': None},
            3: {'name': 'Con Peajes y Peso', 'model': None, 'solution': None}
        }
        self.distance_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Carga distancias cacheadas desde archivo"""
        try:
            if os.path.exists(self.cache_file):
                import pickle
                with open(self.cache_file, 'rb') as f:
                    self.distance_cache = pickle.load(f)
                logger.info(f"Cargada caché con {len(self.distance_cache)} entradas")
        except Exception as e:
            logger.warning(f"Error cargando caché: {str(e)}")
            self.distance_cache = {}

    def _save_cache(self):
        """Guarda la caché de distancias a archivo"""
        try:
            import pickle
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.distance_cache, f)
        except Exception as e:
            logger.warning(f"Error guardando caché: {str(e)}")

    def _api_request_with_retry(self, func, *args, **kwargs):
        """Manejo robusto de peticiones a la API con reintentos"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except openrouteservice.exceptions.ApiError as e:
                last_exception = e
                if 'Rate limit exceeded' in str(e):
                    wait_time = self.base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit excedido. Intento {attempt + 1}/{self.max_retries}. Esperando {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                elif 'Timeout' in str(e):
                    logger.warning(f"Timeout en API. Reintentando...")
                    time.sleep(2)
                    continue
                break
            except Exception as e:
                last_exception = e
                break
        
        logger.error(f"Fallo en petición API después de {self.max_retries} intentos")
        raise last_exception if last_exception else Exception("Error desconocido en API")

    def _get_distance_duration(self, origin, dest):
        """Obtiene distancia y tiempo con caché y fallback"""
        cache_key = f"{origin['Latitude']},{origin['Longitude']}-{dest['Latitude']},{dest['Longitude']}"
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        try:
            coords = (
                (origin['Longitude'], origin['Latitude']),
                (dest['Longitude'], dest['Latitude'])
            )
            routes = self._api_request_with_retry(
                self.client.directions,
                coords,
                profile='driving-hgv',
                units='km',
                validate=False,
                options={'avoid_borders': 'all'}
            )
            distance = routes['routes'][0]['summary']['distance'] / 1000  # km
            duration = routes['routes'][0]['summary']['duration'] / 60  # minutos
            
            self.distance_cache[cache_key] = (distance, duration)
            return distance, duration
            
        except Exception as e:
            logger.warning(f"API falló para {cache_key}, usando haversine. Error: {str(e)}")
            distance = geodesic(
                (origin['Latitude'], origin['Longitude']),
                (dest['Latitude'], dest['Longitude'])
            ).km
            duration = distance * 60 / 80  # asumiendo 80 km/h
            return distance, duration

    def load_data(self, case, data_config):
        """Carga datos específicos para cada caso"""
        data = {}
        
        if case == 1:
            # Caso 1 - Archivos base
            try:
                # Clientes
                clients = pd.read_csv(data_config['clients_path'])
                clients['NodeID'] = 'C' + clients.index.astype(str)
                clients['Latitude'] = clients['Latitude']  # Asegurar que existe
                clients['Longitude'] = clients['Longitude']
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                depots['Latitude'] = depots['Latitude']
                depots['Longitude'] = depots['Longitude']
                data['depots'] = depots
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                vehicles = vehicles.rename(columns={'Range': 'Autonomy'})
                vehicles['VehicleID'] = 'V' + vehicles.index.astype(str)
                data['vehicles'] = vehicles
                
            except Exception as e:
                raise ValueError(f"Error cargando datos caso 1: {str(e)}")

        elif case == 2:
            # Caso 2 - Con recarga
            try:
                # Clientes (necesitan lat/long que no están en el archivo - esto fallará)
                clients = pd.read_csv(data_config['clients_path'])
                clients['NodeID'] = 'C' + clients.index.astype(str)
                # Necesitarías añadir lat/long aquí o modificar el archivo
                raise ValueError("Archivo caso2_clientes.csv necesita columnas Latitude y Longitude")
                
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                data['depots'] = depots
                
                # Estaciones
                stations = pd.read_csv(data_config['stations_path'])
                stations = stations.rename(columns={
                    'EstationID': 'StationID',
                    'FuelCost': 'FuelPrice'
                })
                stations['NodeID'] = 'S' + stations.index.astype(str)
                data['stations'] = stations
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                vehicles = vehicles.rename(columns={
                    'Range': 'Autonomy',
                    'Type': 'VehicleType'
                })
                vehicles['VehicleID'] = 'V' + vehicles.index.astype(str)
                data['vehicles'] = vehicles
                
            except Exception as e:
                raise ValueError(f"Error cargando datos caso 2: {str(e)}")

        elif case == 3:
            # Caso 3 - Con peajes y peso
            try:
                # Clientes
                clients = pd.read_csv(data_config['clients_path'])
                clients['NodeID'] = 'C' + clients.index.astype(str)
                clients = clients.rename(columns={'MaxWeight': 'WeightLimit'})
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                data['depots'] = depots
                
                # Estaciones
                stations = pd.read_csv(data_config['stations_path'])
                stations = stations.rename(columns={
                    'EstationID': 'StationID',
                    'FuelCost': 'FuelPrice'
                })
                stations['NodeID'] = 'S' + stations.index.astype(str)
                data['stations'] = stations
                
                # Peajes
                tolls = pd.read_csv(data_config['tolls_path'])
                tolls = tolls.rename(columns={
                    'BaseRate': 'BaseFee',
                    'RatePerTon': 'VariableFee',
                    'ClientID': 'FromNode'
                })
                tolls['ToNode'] = 'C' + tolls.index.astype(str)
                tolls['FromNode'] = 'D1'  # Asumiendo que los peajes son desde el depósito
                data['tolls'] = tolls
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                vehicles = vehicles.rename(columns={
                    'Range': 'Autonomy',
                    'Type': 'VehicleType'
                })
                vehicles['VehicleID'] = 'V' + vehicles.index.astype(str)
                data['vehicles'] = vehicles
                
            except Exception as e:
                raise ValueError(f"Error cargando datos caso 3: {str(e)}")

        else:
            raise ValueError("Caso no válido")

        # Asignar datos al objeto
        for key, value in data.items():
            setattr(self, key, value)
        
        self.current_case = case
        logger.info(f"Datos cargados para caso {case}")
        return self

    def _prepare_nodes(self):
        """Prepara nodos con estructura optimizada"""
        nodes = []
        
        # Clientes
        self.client_ids = []
        for _, row in self.clients.iterrows():
            node_data = {
                'NodeID': row['NodeID'],
                'Type': 'client',
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'Demand': row['Demand']
            }
            if hasattr(self, 'tolls') and 'WeightLimit' in row:
                node_data['WeightLimit'] = row['WeightLimit']
            if 'Municipality' in row:
                node_data['Municipality'] = row['Municipality']
            nodes.append(node_data)
            self.client_ids.append(row['NodeID'])
        
        # Depósitos
        self.depot_ids = []
        for _, row in self.depots.iterrows():
            nodes.append({
                'NodeID': row['NodeID'],
                'Type': 'depot',
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'Demand': 0
            })
            self.depot_ids.append(row['NodeID'])
        
        # Estaciones (casos 2 y 3)
        if hasattr(self, 'stations'):
            self.station_ids = []
            for _, row in self.stations.iterrows():
                nodes.append({
                    'NodeID': row['NodeID'],
                    'Type': 'station',
                    'Latitude': row['Latitude'],
                    'Longitude': row['Longitude'],
                    'FuelPrice': row['FuelPrice']
                })
                self.station_ids.append(row['NodeID'])
        
        self.nodes = pd.DataFrame(nodes)
        self.demands = {row['NodeID']: row['Demand'] for _, row in self.nodes.iterrows()}
        return self

    def _compute_distance_matrix(self):
        """Calcula matriz de distancias optimizada"""
        num_nodes = len(self.nodes)
        self.distance_matrix = np.zeros((num_nodes, num_nodes))
        self.time_matrix = np.zeros((num_nodes, num_nodes))
        
        logger.info(f"Calculando matriz para {num_nodes} nodos...")
        start_time = time.time()
        
        # Pre-cache de coordenadas
        coords = self.nodes[['Latitude', 'Longitude']].to_dict('records')
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                
                distance, duration = self._get_distance_duration(coords[i], coords[j])
                self.distance_matrix[i][j] = distance
                self.time_matrix[i][j] = duration
            
            # Log progreso
            if (i + 1) % max(1, num_nodes // 10) == 0:
                elapsed = time.time() - start_time
                remaining = (num_nodes - (i + 1)) * (elapsed / (i + 1))
                logger.info(
                    f"Progreso: {i + 1}/{num_nodes} ({(i + 1)/num_nodes*100:.1f}%) | "
                    f"Tiempo restante: {timedelta(seconds=int(remaining))}"
                )
        
        self._save_cache()
        logger.info(f"Matriz calculada en {timedelta(seconds=time.time()-start_time)}")
        return self

    def build_model(self, case):
        """Construye modelo de optimización"""
        self.current_case = case
        self._prepare_nodes()._compute_distance_matrix()
        
        if case == 1:
            return self._build_base_model()
        elif case == 2:
            return self._build_refuel_model()
        elif case == 3:
            return self._build_toll_model()
        else:
            raise ValueError("Caso no válido")

    def _build_base_model(self):
        """Modelo base CVRP"""
        model = ConcreteModel()
        
        # Conjuntos
        model.N = Set(initialize=self.nodes['NodeID'].tolist())
        model.V = Set(initialize=[v for v in self.vehicles['VehicleID']])
        model.C = Set(initialize=self.client_ids)
        model.D = Set(initialize=self.depot_ids)
        
        # Parámetros
        model.demand = Param(model.N, initialize=self.demands)
        model.distance = Param(model.N, model.N, initialize=lambda m, i, j: 
            self.distance_matrix[
                self.nodes[self.nodes['NodeID'] == i].index[0],
                self.nodes[self.nodes['NodeID'] == j].index[0]
            ])
        
        capacities = {row['VehicleID']: row['Capacity'] for _, row in self.vehicles.iterrows()}
        model.capacity = Param(model.V, initialize=capacities)
        
        # Variables
        model.x = Var(model.V, model.N, model.N, within=Binary)
        model.u = Var(model.V, within=NonNegativeReals)
        
        # Objetivo
        def total_distance(m):
            return sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N)
        model.obj = Objective(rule=total_distance, sense=minimize)
        
        # Restricciones
        def depart_from_depot(m, v):
            return sum(m.x[v,d,j] for d in m.D for j in m.N if j != d) == 1
        model.departure = Constraint(model.V, rule=depart_from_depot)
        
        def flow_conservation(m, v, i):
            if i in m.C:
                return sum(m.x[v,j,i] for j in m.N) == sum(m.x[v,i,k] for k in m.N)
            return Constraint.Skip
        model.flow = Constraint(model.V, model.N, rule=flow_conservation)
        
        def serve_client(m, c):
            return sum(m.x[v,i,c] for v in m.V for i in m.N) == 1
        model.serve = Constraint(model.C, rule=serve_client)
        
        def vehicle_capacity(m, v):
            return m.u[v] <= m.capacity[v]
        model.cap = Constraint(model.V, rule=vehicle_capacity)
        
        # MTZ
        model.pi = Var(model.V, model.N, within=NonNegativeIntegers, bounds=(0, len(model.C)))
        
        def no_subtours(m, v, i, j):
            if i != j and i in m.C and j in m.C:
                return m.pi[v,i] - m.pi[v,j] + len(m.C) * m.x[v,i,j] <= len(m.C) - 1
            return Constraint.Skip
        model.subtour = Constraint(model.V, model.N, model.N, rule=no_subtours)
        
        self.case_data[1]['model'] = model
        return model

    def _build_refuel_model(self):
        """Modelo con recarga de combustible"""
        model = self._build_base_model()
        
        # Componentes adicionales
        model.S = Set(initialize=self.station_ids)
        
        autonomies = {row['VehicleID']: row['Autonomy'] for _, row in self.vehicles.iterrows()}
        model.autonomy = Param(model.V, initialize=autonomies)
        
        fuel_prices = {row['NodeID']: row['FuelPrice'] for _, row in self.nodes[self.nodes['Type'] == 'station'].iterrows()}
        model.fuel_price = Param(model.S, initialize=fuel_prices)
        
        # Variables adicionales
        model.y = Var(model.V, model.N, model.S, within=Binary)  # Recarga en e después de i
        model.r = Var(model.V, model.S, within=NonNegativeReals)  # Cantidad recargada
        
        # Actualizar objetivo
        def total_cost(m):
            distance_cost = sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N)
            fuel_cost = sum(m.fuel_price[e] * m.r[v,e] for v in m.V for e in m.S)
            return distance_cost + fuel_cost
        model.obj = Objective(rule=total_cost, sense=minimize)
        
        # Restricciones de recarga
        def refuel_only_if_visited(m, v, i, e):
            return m.y[v,i,e] <= m.x[v,i,e]
        model.refuel_link = Constraint(model.V, model.N, model.S, rule=refuel_only_if_visited)
        
        def max_segment_distance(m, v):
            return sum(m.distance[i,j] * m.x[v,i,j] for i in m.N for j in m.N) <= m.autonomy[v] * (
                1 + sum(m.r[v,e] for e in m.S) / 100)
        model.range_limit = Constraint(model.V, rule=max_segment_distance)
        
        self.case_data[2]['model'] = model
        return model

    def _build_toll_model(self):
        """Modelo con peajes y restricciones de peso"""
        model = self._build_refuel_model()
        
        # Componentes adicionales
        toll_arcs = [(row['FromNode'], row['ToNode']) for _, row in self.tolls.iterrows()]
        model.T = Set(initialize=toll_arcs)
        
        toll_base = {(row['FromNode'], row['ToNode']): row['BaseFee'] for _, row in self.tolls.iterrows()}
        model.toll_base = Param(model.T, initialize=toll_base)
        
        toll_var = {(row['FromNode'], row['ToNode']): row['VariableFee'] for _, row in self.tolls.iterrows()}
        model.toll_var = Param(model.T, initialize=toll_var)
        
        # Restricciones de peso
        weight_limits = {}
        for _, row in self.nodes[self.nodes['Type'] == 'client'].iterrows():
            if 'WeightLimit' in row:
                weight_limits[row['NodeID']] = row['WeightLimit']
        
        model.weight_limit = Param(Set(initialize=weight_limits.keys()), 
                                 initialize=weight_limits)
        model.u_m = Var(model.V, model.weight_limit.index_set(), within=NonNegativeReals)
        
        # Actualizar objetivo
        def total_cost_with_tolls(m):
            base_cost = sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N)
            fuel_cost = sum(m.fuel_price[e] * m.r[v,e] for v in m.V for e in m.S)
            toll_cost = sum((m.toll_base[i,j] + m.toll_var[i,j] * m.u[v]) * m.x[v,i,j] 
                       for v in m.V for (i,j) in m.T)
            penalty = sum(m.u_m[v,m] * 1000 for v in m.V for m in m.weight_limit)
            return base_cost + fuel_cost + toll_cost + penalty
        model.obj = Objective(rule=total_cost_with_tolls, sense=minimize)
        
        # Restricción de peso
        def enforce_weight_limits(m, v, m_id):
            return m.u[v] <= m.weight_limit[m_id] + m.u_m[v,m_id]
        model.weight_constraint = Constraint(model.V, model.weight_limit.index_set(), 
                                           rule=enforce_weight_limits)
        
        self.case_data[3]['model'] = model
        return model

    def solve_model(self, case, solver='glpk', timeout=600):
        """Resuelve el modelo con manejo de errores"""
        if self.case_data[case]['model'] is None:
            raise ValueError("Modelo no construido")
        
        logger.info(f"\n{'='*40}\nResolviendo caso {case} ({self.case_data[case]['name']})\n{'='*40}")
        
        solver_obj = SolverFactory(solver)
        if solver == 'glpk':
            solver_obj.options['tmlim'] = timeout
        elif solver == 'cplex':
            solver_obj.options['timelimit'] = timeout
        
        model = self.case_data[case]['model']
        start_time = time.time()
        
        try:
            results = solver_obj.solve(model, tee=True)
            solve_time = time.time() - start_time
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                logger.info(f"Solución óptima encontrada en {timedelta(seconds=solve_time)}")
                self._extract_solution(case)
                return True
            else:
                logger.warning(f"Solución no óptima: {results.solver.termination_condition}")
                return False
        except Exception as e:
            logger.error(f"Error al resolver: {str(e)}")
            return False

    def _extract_solution(self, case):
        """Procesa la solución del modelo"""
        model = self.case_data[case]['model']
        solution = {
            'routes': [],
            'total_distance': 0,
            'total_cost': 0,
            'fuel_cost': 0,
            'toll_cost': 0,
            'penalty_cost': 0
        }
        
        for v in model.V:
            route = {
                'vehicle': v,
                'path': [],
                'distance': 0,
                'load': 0,
                'fuel_used': 0,
                'tolls_paid': 0
            }
            
            # Encontrar inicio de ruta
            current_node = next((d for d in model.D 
                               if any(value(model.x[v,d,j]) > 0.9 for j in model.N)), None)
            if not current_node:
                continue
                
            route['path'].append(current_node)
            
            # Reconstruir ruta
            while True:
                next_node = next((j for j in model.N 
                                if value(model.x[v,current_node,j]) > 0.9), None)
                
                if not next_node or next_node in model.D:
                    break
                
                route['path'].append(next_node)
                route['distance'] += value(model.distance[current_node,next_node])
                
                if next_node in model.C:
                    route['load'] += value(model.demand[next_node])
                
                current_node = next_node
            
            # Costos específicos
            if case >= 2:
                route['fuel_used'] = sum(
                    value(model.r[v,e]) * value(model.fuel_price[e]) 
                    for e in model.S if hasattr(model, 'r')
                )
            
            if case >= 3:
                route['tolls_paid'] = sum(
                    (value(model.toll_base[i,j]) + value(model.toll_var[i,j]) * value(model.u[v])) * 
                    value(model.x[v,i,j]) 
                    for (i,j) in model.T if hasattr(model, 'T')
                )
            
            solution['routes'].append(route)
            solution['total_distance'] += route['distance']
            solution['total_cost'] += route['distance'] + route['fuel_used'] + route['tolls_paid']
            solution['fuel_cost'] += route['fuel_used']
            solution['toll_cost'] += route['tolls_paid']
        
        self.case_data[case]['solution'] = solution
        return solution

    def generate_verification_file(self, case, output_dir='output'):
        """Genera archivo de verificación CSV"""
        os.makedirs(output_dir, exist_ok=True)
        solution = self.case_data[case]['solution']
        if not solution:
            raise ValueError("No hay solución disponible")
        
        filename = os.path.join(output_dir, f'verificacion_caso{case}.csv')
        
        with open(filename, 'w', encoding='utf-8') as f:
            if case == 1:
                headers = [
                    'VehicleId', 'DepotId', 'InitialLoad', 'RouteSequence', 
                    'ClientsServed', 'DemandsSatisfied', 'TotalDistance', 
                    'TotalTime', 'FuelCost'
                ]
                f.write(','.join(headers) + '\n')
                
                for route in solution['routes']:
                    clients = [n for n in route['path'] if n.startswith('C')]
                    demands = [str(int(self.demands[c])) for c in clients]
                    
                    row = [
                        route['vehicle'],
                        route['path'][0],
                        int(route['load']),
                        '-'.join(route['path']),
                        len(clients),
                        '-'.join(demands),
                        f"{route['distance']:.2f}",
                        f"{route['distance']*60/80:.2f}",
                        "0"
                    ]
                    f.write(','.join(map(str, row)) + '\n')
            
            elif case == 2:
                headers = [
                    'VehicleId', 'VehicleType', 'InitialLoad', 'RouteSequence',
                    'ClientsServed', 'DemandSatisfied', 'ArrivalTimes',
                    'Resup', 'ResupAmounts', 'Distance', 'Time', 'Cost'
                ]
                f.write(','.join(headers) + '\n')
                
                for route in solution['routes']:
                    clients = [n for n in route['path'] if n.startswith('C')]
                    demands = [str(int(self.demands[c])) for c in clients]
                    stations = [n for n in route['path'] if n.startswith('S')]
                    
                    # Estimación de tiempos de llegada
                    time_per_km = 60 / 80  # minutos por km a 80 km/h
                    current_time = 0
                    arrival_times = []
                    for i in range(len(route['path']) - 1):
                        from_node = route['path'][i]
                        to_node = route['path'][i+1]
                        dist = self.distance_matrix[
                            self.nodes[self.nodes['NodeID'] == from_node].index[0],
                            self.nodes[self.nodes['NodeID'] == to_node].index[0]
                        ]
                        current_time += dist * time_per_km
                        if to_node.startswith('C'):
                            arrival_times.append(f"{int(current_time//60):02d}:{int(current_time%60):02d}")
                    
                    # Obtener tipo de vehículo
                    vehicle_type = self.vehicles[self.vehicles['VehicleID'] == route['vehicle']]['Type'].iloc[0]
                    
                    row = [
                        route['vehicle'],
                        vehicle_type,
                        int(route['load']),
                        '-'.join(route['path']),
                        len(clients),
                        '-'.join(demands),
                        '-'.join(arrival_times),
                        len(stations),
                        str(int(route['fuel_used']/50)) if route['fuel_used'] > 0 else "0",
                        f"{route['distance']:.2f}",
                        f"{route['distance']*time_per_km:.2f}",
                        f"{route['distance'] + route['fuel_used']:.2f}"
                    ]
                    f.write(','.join(map(str, row)) + '\n')
            
            elif case == 3:
                headers = [
                    'VehicleId', 'LoadCap', 'FuelCap', 'RouteSeq', 'Municipalities',
                    'Demand', 'InitLoad', 'InitFuel', 'RefuelStops', 'RefuelAmounts',
                    'TollVisited', 'TollCosts', 'VehicleWeights', 'Distance',
                    'Time', 'FuelCost', 'TollCost', 'TotalCost'
                ]
                f.write(','.join(headers) + '\n')
                
                for route in solution['routes']:
                    clients = [n for n in route['path'] if n.startswith('C')]
                    demands = [str(int(self.demands[c])) for c in clients]
                    stations = [n for n in route['path'] if n.startswith('S')]
                    toll_points = [n for n in route['path'] if any(n in pair for pair in model.T)]
                    
                    # Obtener información del vehículo
                    vehicle_info = self.vehicles[self.vehicles['VehicleID'] == route['vehicle']].iloc[0]
                    
                    # Calcular pesos en municipios
                    weights = []
                    current_load = route['load']
                    for node in route['path']:
                        if node.startswith('C'):
                            mun = self.nodes[self.nodes['NodeID'] == node]['Municipality'].iloc[0]
                            if mun in ['Guasca', 'Cogua']:
                                weights.append(str(int(current_load)))
                            current_load -= self.demands[node]
                    
                    row = [
                        route['vehicle'],
                        int(vehicle_info['Capacity']),
                        int(vehicle_info['Autonomy']),
                        '-'.join(route['path']),
                        len(clients),
                        '-'.join(demands),
                        int(route['load']),
                        int(vehicle_info['Autonomy']),
                        len(stations),
                        str(int(route['fuel_used']/50)) if route['fuel_used'] > 0 else "0",
                        len(toll_points),
                        str(int(route['tolls_paid'])),
                        '-'.join(weights),
                        f"{route['distance']:.2f}",
                        f"{route['distance']*60/80:.2f}",
                        f"{route['fuel_used']:.2f}",
                        f"{route['tolls_paid']:.2f}",
                        f"{route['distance'] + route['fuel_used'] + route['tolls_paid']:.2f}"
                    ]
                    f.write(','.join(map(str, row)) + '\n')
        
        logger.info(f"Archivo de verificación generado: {filename}")
        return filename

    def visualize_routes(self, case, output_dir='output'):
        """Genera mapa interactivo de rutas"""
        os.makedirs(output_dir, exist_ok=True)
        solution = self.case_data[case]['solution']
        if not solution:
            raise ValueError("No hay solución disponible")
        
        # Crear mapa centrado en el primer depósito
        depot = self.nodes[self.nodes['Type'] == 'depot'].iloc[0]
        m = folium.Map(
            location=[depot['Latitude'], depot['Longitude']],
            zoom_start=10,
            tiles='cartodbpositron'
        )
        
        # Añadir depósitos
        for _, row in self.nodes[self.nodes['Type'] == 'depot'].iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa'),
                popup=f"Depósito {row['NodeID']}",
                tooltip=f"Depósito {row['NodeID']}"
            ).add_to(m)
        
        # Añadir clientes
        for _, row in self.nodes[self.nodes['Type'] == 'client'].iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                icon=folium.Icon(color='blue', icon='user', prefix='fa'),
                popup=f"Cliente {row['NodeID']}<br>Demanda: {row['Demand']}",
                tooltip=f"Cliente {row['NodeID']}"
            ).add_to(m)
        
        # Añadir estaciones (casos 2 y 3)
        if case >= 2 and hasattr(self, 'stations'):
            for _, row in self.nodes[self.nodes['Type'] == 'station'].iterrows():
                folium.Marker(
                    [row['Latitude'], row['Longitude']],
                    icon=folium.Icon(color='green', icon='gas-pump', prefix='fa'),
                    popup=f"Estación {row['NodeID']}<br>Precio: ${row['FuelPrice']}/L",
                    tooltip=f"Estación {row['NodeID']}"
                ).add_to(m)
        
        # Dibujar rutas con colores distintos
        colors = [
            'darkblue', 'orange', 'darkgreen', 'purple', 
            'red', 'lightblue', 'pink', 'lightgreen'
        ]
        
        for i, route in enumerate(solution['routes']):
            path_coords = []
            for node_id in route['path']:
                node = self.nodes[self.nodes['NodeID'] == node_id].iloc[0]
                path_coords.append([node['Latitude'], node['Longitude']])
            
            # Crear línea con información de la ruta
            folium.PolyLine(
                path_coords,
                color=colors[i % len(colors)],
                weight=3,
                opacity=0.8,
                popup=(
                    f"Vehículo: {route['vehicle']}<br>"
                    f"Distancia: {route['distance']:.2f} km<br>"
                    f"Carga: {route['load']} kg"
                ),
                tooltip=f"Ruta {i+1}"
            ).add_to(m)
        
        filename = os.path.join(output_dir, f'rutas_caso{case}.html')
        m.save(filename)
        logger.info(f"Mapa de rutas generado: {filename}")
        return filename

def main():
    try:
        # Configuración de rutas de archivos
        DATA_DIR = {
            1: {
                'clients_path': 'BaseData/clients.csv',
                'vehicles_path': 'BaseData/vehicles.csv',
                'depots_path': 'BaseData/depots.csv'
            },
            2: {
                'clients_path': 'data/caso2_clientes.csv',
                'vehicles_path': 'data/caso2_vehiculos.csv',
                'depots_path': 'data/caso2_depositos.csv',
                'stations_path': 'data/caso2_estaciones.csv'
            },
            3: {
                'clients_path': 'data/caso3_clientes.csv',
                'vehicles_path': 'data/caso3_vehiculos.csv',
                'depots_path': 'data/caso3_depositos.csv',
                'stations_path': 'data/caso3_estaciones.csv',
                'tolls_path': 'data/caso3_peajes.csv'
            }
        }
        
        # Crear optimizador
        optimizer = LogistiCoRouteOptimizer()
        
        # Procesar cada caso
        for case in [1, 2, 3]:
            try:
                logger.info(f"\n{'='*40}\nPROCESANDO CASO {case}\n{'='*40}")
                
                optimizer.load_data(case, DATA_DIR[case])
                optimizer.build_model(case)
                
                solver = 'cplex' if case == 3 else 'glpk'
                if optimizer.solve_model(case, solver=solver, timeout=600):
                    optimizer.generate_verification_file(case)
                    optimizer.visualize_routes(case)
                    logger.info(f"Caso {case} completado con éxito")
                else:
                    logger.warning(f"Caso {case} no obtuvo solución óptima")
                
            except Exception as e:
                logger.error(f"Error procesando caso {case}: {str(e)}", exc_info=True)
                continue
        
        logger.info("\n" + "="*50)
        logger.info("PROCESO COMPLETADO".center(50))
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"ERROR CRÍTICO: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"\nTiempo total de ejecución: {timedelta(seconds=time.time()-start_time)}")