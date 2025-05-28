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
        self.case_data = {
            1: {'name': 'CVRP Estándar', 'model': None, 'solution': None},
            2: {'name': 'Con Recarga', 'model': None, 'solution': None},
            3: {'name': 'Con Peajes y Peso', 'model': None, 'solution': None}
        }
        self.distance_cache = {}
        self.cache_file = 'distance_cache.pkl'
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

    def _get_distance_duration(self, origin, dest):
        """Obtiene distancia y tiempo entre dos puntos usando haversine"""
        cache_key = f"{origin['Latitude']},{origin['Longitude']}-{dest['Latitude']},{dest['Longitude']}"
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        try:
            # Calcular distancia usando haversine
            distance = geodesic(
                (origin['Latitude'], origin['Longitude']),
                (dest['Latitude'], dest['Longitude'])
            ).km
            
            # Calcular duración asumiendo velocidad promedio de 80 km/h
            duration = distance * 60 / 80  # minutos
            
            self.distance_cache[cache_key] = (distance, duration)
            return distance, duration
            
        except Exception as e:
            logger.error(f"Error calculando distancia para {origin}-{dest}: {str(e)}")
            raise

    def load_data(self, case, data_config):
        """Carga datos específicos para cada caso"""
        data = {}
        
        def validate_coordinates(df, name):
            """Valida que existan las columnas de coordenadas"""
            required_cols = ['Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Archivo {name} necesita las columnas: {', '.join(missing_cols)}")
            return df
        
        def validate_required_columns(df, required_cols, file_name):
            """Valida que existan las columnas requeridas"""
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas en {file_name}: {missing_cols}")
            return df
        
        if case == 1:
            # Caso 1 - Archivos base
            try:
                # Clientes
                clients = pd.read_csv(data_config['clients_path'])
                required_cols = ['ClientID', 'Demand', 'Latitude', 'Longitude']
                clients = validate_required_columns(clients, required_cols, 'clients.csv')
                clients['NodeID'] = 'C' + clients.index.astype(str)
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                required_cols = ['DepotID', 'Latitude', 'Longitude']
                depots = validate_required_columns(depots, required_cols, 'depots.csv')
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                data['depots'] = depots
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                required_cols = ['Capacity', 'Range']
                vehicles = validate_required_columns(vehicles, required_cols, 'vehicles.csv')
                vehicles = vehicles.rename(columns={'Range': 'Autonomy'})
                vehicles['VehicleID'] = 'V' + vehicles.index.astype(str)
                data['vehicles'] = vehicles
                
            except Exception as e:
                raise ValueError(f"Error cargando datos caso 1: {str(e)}")

        elif case == 2:
            # Caso 2 - Con recarga
            try:
                # Clientes
                clients = pd.read_csv(data_config['clients_path'])
                required_cols = ['ClientID', 'Demand', 'Latitude', 'Longitude']
                clients = validate_required_columns(clients, required_cols, 'caso2_clientes.csv')
                clients['NodeID'] = 'C' + clients.index.astype(str)
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                required_cols = ['DepotID', 'Latitude', 'Longitude']
                depots = validate_required_columns(depots, required_cols, 'caso2_depositos.csv')
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                data['depots'] = depots
                
                # Estaciones
                stations = pd.read_csv(data_config['stations_path'])
                required_cols = ['StationID', 'Latitude', 'Longitude', 'FuelPrice']
                stations = validate_required_columns(stations, required_cols, 'caso2_estaciones.csv')
                stations = stations.rename(columns={
                    'StationID': 'StationID',
                    'FuelPrice': 'FuelPrice'
                })
                stations['NodeID'] = 'S' + stations.index.astype(str)
                data['stations'] = stations
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                required_cols = ['Capacity', 'Range', 'VehicleType']
                vehicles = validate_required_columns(vehicles, required_cols, 'caso2_vehiculos.csv')
                vehicles = vehicles.rename(columns={
                    'Range': 'Autonomy',
                    'VehicleType': 'VehicleType'
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
                required_cols = ['ClientID', 'Demand', 'MaxWeight', 'Latitude', 'Longitude']
                clients = validate_required_columns(clients, required_cols, 'caso3_clientes.csv')
                clients['NodeID'] = 'C' + clients.index.astype(str)
                clients = clients.rename(columns={'MaxWeight': 'WeightLimit'})
                data['clients'] = clients
                
                # Depósitos
                depots = pd.read_csv(data_config['depots_path'])
                required_cols = ['DepotID', 'Latitude', 'Longitude']
                depots = validate_required_columns(depots, required_cols, 'caso3_depositos.csv')
                depots['NodeID'] = 'D' + depots['DepotID'].astype(str)
                data['depots'] = depots
                
                # Estaciones
                stations = pd.read_csv(data_config['stations_path'])
                required_cols = ['StationID', 'Latitude', 'Longitude', 'FuelPrice']
                stations = validate_required_columns(stations, required_cols, 'caso3_estaciones.csv')
                stations = stations.rename(columns={
                    'StationID': 'StationID',
                    'FuelPrice': 'FuelPrice'
                })
                stations['NodeID'] = 'S' + stations.index.astype(str)
                data['stations'] = stations
                
                # Peajes
                tolls = pd.read_csv(data_config['tolls_path'])
                required_cols = ['FromNode', 'BaseFee', 'VariableFee']
                tolls = validate_required_columns(tolls, required_cols, 'caso3_peajes.csv')
                tolls = tolls.rename(columns={
                    'BaseFee': 'BaseFee',
                    'VariableFee': 'VariableFee'
                })
                data['tolls'] = tolls
                
                # Vehículos
                vehicles = pd.read_csv(data_config['vehicles_path'])
                required_cols = ['Capacity', 'Range', 'VehicleType']
                vehicles = validate_required_columns(vehicles, required_cols, 'caso3_vehiculos.csv')
                vehicles = vehicles.rename(columns={
                    'Range': 'Autonomy',
                    'VehicleType': 'VehicleType'
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
        """Prepara los nodos para el modelo"""
        self.nodes = pd.DataFrame(columns=['NodeID', 'Type', 'Latitude', 'Longitude', 'Demand', 'FuelPrice'])
        
        # Agregar depósitos
        for _, row in self.depots.iterrows():
            self.nodes = pd.concat([self.nodes, pd.DataFrame([{
                'NodeID': row['NodeID'],
                'Type': 'depot',
                'Latitude': float(row['Latitude']),
                'Longitude': float(row['Longitude']),
                'Demand': 0,
                'FuelPrice': 0
            }])], ignore_index=True)
        
        # Agregar clientes
        for _, row in self.clients.iterrows():
            self.nodes = pd.concat([self.nodes, pd.DataFrame([{
                'NodeID': row['NodeID'],
                'Type': 'client',
                'Latitude': float(row['Latitude']),
                'Longitude': float(row['Longitude']),
                'Demand': float(row['Demand']),
                'FuelPrice': 0
            }])], ignore_index=True)
        
        # Agregar estaciones si existen
        if hasattr(self, 'stations'):
            for _, row in self.stations.iterrows():
                self.nodes = pd.concat([self.nodes, pd.DataFrame([{
                    'NodeID': row['NodeID'],
                    'Type': 'station',
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                    'Demand': 0,
                    'FuelPrice': float(row['FuelPrice'])
                }])], ignore_index=True)
        
        return self

    def _compute_distance_matrix(self):
        """Calcula la matriz de distancias entre todos los nodos"""
        n = len(self.nodes)
        logger.info(f"Calculando matriz para {n} nodos...")
        
        self.distance_matrix = {}
        for i in self.nodes['NodeID'].tolist():
            for j in self.nodes['NodeID'].tolist():
                if i != j:
                    node_i = self.nodes[self.nodes['NodeID'] == i]
                    node_j = self.nodes[self.nodes['NodeID'] == j]
                    if not node_i.empty and not node_j.empty:
                        distance, duration = self._get_distance_duration(
                            node_i.iloc[0],
                            node_j.iloc[0]
                        )
                        self.distance_matrix[i, j] = distance
                else:
                    self.distance_matrix[i, j] = 0
        
        return self

    def build_model(self, case):
        """Construye el modelo de optimización según el caso"""
        logger.info(f"Construyendo modelo para caso {case}")
        
        # Preparar nodos y matriz de distancias
        self._prepare_nodes()
        self._compute_distance_matrix()
        
        # Construir modelo base
        model = self._build_base_model()
        
        if case == 2:
            # Agregar restricciones de recarga
            self._build_refuel_model()
        elif case == 3:
            # Agregar restricciones de peajes y peso
            self._build_toll_model()
            
        self.case_data[case]['model'] = model
        return model

    def _build_base_model(self):
        """Modelo base CVRP usando formulación GG (Gavish-Graves)"""
        model = ConcreteModel()
        
        # Conjuntos
        model.N = Set(initialize=self.nodes['NodeID'].tolist())
        model.V = Set(initialize=[v for v in self.vehicles['VehicleID']])
        model.C = Set(initialize=self.nodes[self.nodes['Type'] == 'client']['NodeID'].tolist())
        model.D = Set(initialize=self.nodes[self.nodes['Type'] == 'depot']['NodeID'].tolist())
        
        # Parámetros
        demand_dict = {row['NodeID']: row['Demand'] for _, row in self.nodes.iterrows()}
        model.demand = Param(model.N, initialize=demand_dict)
        
        distance_dict = {}
        for i in model.N:
            for j in model.N:
                if i != j:
                    distance_dict[i, j] = self.distance_matrix[i, j]
                else:
                    distance_dict[i, j] = 0
        model.distance = Param(model.N, model.N, initialize=distance_dict)
        
        capacities = {row['VehicleID']: row['Capacity'] for _, row in self.vehicles.iterrows()}
        model.capacity = Param(model.V, initialize=capacities)
        
        # Variables
        model.x = Var(model.V, model.N, model.N, within=Binary)  # Variable de ruta
        model.u = Var(model.V, within=NonNegativeReals)  # Carga del vehículo
        
        # Objetivo
        def total_distance(m):
            return sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N if i != j)
        model.obj = Objective(rule=total_distance, sense=minimize)
        
        # Restricciones
        # Cada cliente debe ser visitado exactamente una vez
        def serve_client(m, c):
            return sum(m.x[v,i,c] for v in m.V for i in m.N if i != c) == 1
        model.serve = Constraint(model.C, rule=serve_client)
        
        # Conservación de flujo para cada vehículo
        def flow_conservation(m, v, i):
            if i in m.C:
                return sum(m.x[v,j,i] for j in m.N if j != i) == sum(m.x[v,i,j] for j in m.N if j != i)
            return Constraint.Skip
        model.flow = Constraint(model.V, model.N, rule=flow_conservation)
        
        # Restricción de capacidad
        def capacity_constraint(m, v):
            return m.u[v] <= m.capacity[v]
        model.capacity_constraint = Constraint(model.V, rule=capacity_constraint)
        
        # Restricción de demanda
        def demand_constraint(m, v):
            return m.u[v] >= sum(m.demand[i] * sum(m.x[v,i,j] for j in m.N if j != i) for i in m.C)
        model.demand_constraint = Constraint(model.V, rule=demand_constraint)
        
        # Cada vehículo debe salir del depósito
        def depart_from_depot(m, v):
            return sum(m.x[v,d,j] for d in m.D for j in m.N if j != d) == 1
        model.departure = Constraint(model.V, rule=depart_from_depot)
        
        # Cada vehículo debe regresar al depósito
        def return_to_depot(m, v):
            return sum(m.x[v,i,d] for d in m.D for i in m.N if i != d) == 1
        model.return_constraint = Constraint(model.V, rule=return_to_depot)
        
        return model

    def _build_refuel_model(self):
        """Modelo con recarga de combustible"""
        model = self._build_base_model()
        
        # Componentes adicionales
        model.S = Set(initialize=self.nodes[self.nodes['Type'] == 'station']['NodeID'].tolist())
        
        autonomies = {row['VehicleID']: row['Autonomy'] for _, row in self.vehicles.iterrows()}
        model.autonomy = Param(model.V, initialize=autonomies)
        
        fuel_prices = {row['NodeID']: row['FuelPrice'] for _, row in self.nodes[self.nodes['Type'] == 'station'].iterrows()}
        model.fuel_price = Param(model.S, initialize=fuel_prices)
        
        # Variables adicionales
        model.y = Var(model.V, model.N, model.S, within=Binary)  # Recarga en e después de i
        model.r = Var(model.V, model.S, within=NonNegativeReals)  # Cantidad recargada
        model.fuel_level = Var(model.V, model.N, within=NonNegativeReals)  # Nivel de combustible
        
        # Actualizar objetivo
        def total_cost(m):
            distance_cost = sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N if i != j)
            fuel_cost = sum(m.fuel_price[e] * m.r[v,e] for v in m.V for e in m.S)
            return distance_cost + fuel_cost
        model.obj = Objective(rule=total_cost, sense=minimize)
        
        # Restricciones de recarga
        def refuel_link(m, v, i, e):
            return m.y[v,i,e] <= m.x[v,i,e]
        model.refuel_link = Constraint(model.V, model.N, model.S, rule=refuel_link)
        
        # Nivel de combustible inicial
        def initial_fuel(m, v):
            return m.fuel_level[v, m.D[1]] == m.autonomy[v]
        model.initial_fuel = Constraint(model.V, rule=initial_fuel)
        
        # Consumo de combustible
        def fuel_consumption(m, v, i, j):
            if i != j:
                return m.fuel_level[v,j] == m.fuel_level[v,i] - m.distance[i,j] + sum(m.r[v,e] * m.y[v,i,e] for e in m.S)
            return Constraint.Skip
        model.fuel_consumption = Constraint(model.V, model.N, model.N, rule=fuel_consumption)
        
        # Capacidad de combustible
        def fuel_capacity(m, v, i):
            return m.fuel_level[v,i] <= m.autonomy[v]
        model.fuel_capacity = Constraint(model.V, model.N, rule=fuel_capacity)
        
        # Combustible mínimo
        def min_fuel(m, v, i):
            return m.fuel_level[v,i] >= 0
        model.min_fuel = Constraint(model.V, model.N, rule=min_fuel)
        
        return model

    def _build_toll_model(self):
        """Modelo con peajes y restricciones de peso"""
        model = self._build_base_model()
        
        # Componentes adicionales
        toll_arcs = []
        for _, row in self.tolls.iterrows():
            from_node = row['FromNode']
            to_node = row.get('ToNode', from_node)  # Usar FromNode si ToNode no existe
            if pd.notna(from_node) and pd.notna(to_node):
                toll_arcs.append((from_node, to_node))
        model.T = Set(initialize=toll_arcs)
        
        # Inicializar parámetros de peaje
        toll_base = {}
        toll_var = {}
        for from_node, to_node in toll_arcs:
            toll_data = self.tolls[self.tolls['FromNode'] == from_node]
            if not toll_data.empty:
                base_fee = toll_data['BaseFee'].iloc[0]
                var_fee = toll_data['VariableFee'].iloc[0]
                # Manejar valores NaN
                toll_base[from_node, to_node] = float(base_fee) if pd.notna(base_fee) else 0.0
                toll_var[from_node, to_node] = float(var_fee) if pd.notna(var_fee) else 0.0
            else:
                toll_base[from_node, to_node] = 0.0
                toll_var[from_node, to_node] = 0.0
        
        model.toll_base = Param(model.T, initialize=toll_base, within=NonNegativeReals)
        model.toll_var = Param(model.T, initialize=toll_var, within=NonNegativeReals)
        
        # Restricciones de peso por municipio
        weight_limits = {}
        for _, row in self.nodes[self.nodes['Type'] == 'client'].iterrows():
            if 'WeightLimit' in row and pd.notna(row['WeightLimit']):
                weight_limits[row['NodeID']] = float(row['WeightLimit'])
        
        model.weight_limit = Param(Set(initialize=weight_limits.keys()), 
                                 initialize=weight_limits,
                                 within=NonNegativeReals)
        
        # Actualizar objetivo
        def total_cost_with_tolls(m):
            base_cost = sum(m.distance[i,j] * m.x[v,i,j] for v in m.V for i in m.N for j in m.N if i != j)
            toll_cost = sum((m.toll_base[i,j] + m.toll_var[i,j] * m.u[v]) * m.x[v,i,j] 
                       for v in m.V for (i,j) in m.T if (v,i,j) in m.x)
            return base_cost + toll_cost
        model.obj = Objective(rule=total_cost_with_tolls, sense=minimize)
        
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
        if not model:
            raise ValueError("No hay modelo disponible")
        
        solution = {
            'routes': [],
            'total_distance': 0,
            'total_cost': 0,
            'fuel_cost': 0,
            'toll_cost': 0
        }
        
        try:
            # Obtener rutas activas
            active_routes = []
            for v in model.V:
                for i in model.N:
                    for j in model.N:
                        try:
                            if i != j and value(model.x[v,i,j]) > 0.5:  # Ignorar rutas del mismo nodo
                                active_routes.append((v,i,j))
                        except (ValueError, TypeError):
                            continue
            
            if not active_routes:
                logger.warning("No se encontraron rutas activas en la solución")
                return solution
            
            # Reconstruir rutas
            current_route = None
            for v, i, j in active_routes:
                if i in model.D:  # Nueva ruta
                    if current_route:
                        solution['routes'].append(current_route)
                    try:
                        current_route = {
                            'vehicle': v,
                            'path': [i],
                            'distance': 0,
                            'load': value(model.u[v]),
                            'fuel_used': 0,
                            'tolls_paid': 0
                        }
                    except (ValueError, TypeError):
                        logger.warning(f"No se pudo obtener la carga para el vehículo {v}")
                        continue
                
                if current_route:
                    current_route['path'].append(j)
                    try:
                        current_route['distance'] += value(model.distance[i,j])
                    except (ValueError, TypeError):
                        logger.warning(f"No se pudo obtener la distancia entre {i} y {j}")
                        continue
            
            if current_route:
                solution['routes'].append(current_route)
            
            # Calcular métricas adicionales
            for route in solution['routes']:
                # Calcular combustible usado
                if case >= 2 and hasattr(model, 'fuel_price'):
                    try:
                        route['fuel_used'] = sum(
                            value(model.fuel_price[e]) * value(model.r[v,e])
                            for v in model.V for e in model.S
                            if value(model.y[v,i,e]) > 0.5
                        )
                    except (ValueError, TypeError):
                        logger.warning("No se pudo calcular el combustible usado")
                        route['fuel_used'] = 0
                
                # Calcular peajes
                if case >= 3 and hasattr(model, 'toll_base'):
                    try:
                        route['tolls_paid'] = sum(
                            (value(model.toll_base[i,j]) + value(model.toll_var[i,j]) * value(model.u[v])) * 
                            value(model.x[v,i,j])
                            for v in model.V for i in model.N for j in model.N
                            if value(model.x[v,i,j]) > 0.5
                        )
                    except (ValueError, TypeError):
                        logger.warning("No se pudo calcular los peajes")
                        route['tolls_paid'] = 0
                
                solution['total_distance'] += route['distance']
                solution['total_cost'] += route['distance'] + route['fuel_used'] + route['tolls_paid']
                solution['fuel_cost'] += route['fuel_used']
                solution['toll_cost'] += route['tolls_paid']
            
            self.case_data[case]['solution'] = solution
            return solution
            
        except Exception as e:
            logger.error(f"Error al extraer la solución: {str(e)}")
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
                    demands = [str(int(self.nodes[self.nodes['NodeID'] == c]['Demand'].iloc[0])) for c in clients]
                    
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
                    demands = [str(int(self.nodes[self.nodes['NodeID'] == c]['Demand'].iloc[0])) for c in clients]
                    stations = [n for n in route['path'] if n.startswith('S')]
                    
                    # Estimación de tiempos de llegada
                    time_per_km = 60 / 80  # minutos por km a 80 km/h
                    current_time = 0
                    arrival_times = []
                    for i in range(len(route['path']) - 1):
                        from_node = route['path'][i]
                        to_node = route['path'][i+1]
                        dist = self.distance_matrix[from_node, to_node]
                        current_time += dist * time_per_km
                        if to_node.startswith('C'):
                            arrival_times.append(f"{int(current_time//60):02d}:{int(current_time%60):02d}")
                    
                    # Obtener tipo de vehículo
                    vehicle_type = self.vehicles[self.vehicles['VehicleID'] == route['vehicle']]['VehicleType'].iloc[0]
                    
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
                    demands = [str(int(self.nodes[self.nodes['NodeID'] == c]['Demand'].iloc[0])) for c in clients]
                    stations = [n for n in route['path'] if n.startswith('S')]
                    
                    # Identificar puntos de peaje
                    toll_points = []
                    for i in range(len(route['path']) - 1):
                        from_node = route['path'][i]
                        to_node = route['path'][i+1]
                        if any(from_node == row['FromNode'] for _, row in self.tolls.iterrows()):
                            toll_points.append(from_node)
                    
                    # Obtener información del vehículo
                    vehicle_info = self.vehicles[self.vehicles['VehicleID'] == route['vehicle']].iloc[0]
                    
                    # Calcular pesos en municipios
                    weights = []
                    current_load = route['load']
                    for node in route['path']:
                        if node.startswith('C'):
                            weights.append(str(int(current_load)))
                            current_load -= self.nodes[self.nodes['NodeID'] == node]['Demand'].iloc[0]
                    
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
                solver = 'glpk'
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