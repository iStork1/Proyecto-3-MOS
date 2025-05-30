import requests
import os
import json
from typing import Dict, Tuple
import time
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
env_path = Path(__file__).parent.parent / '.env' # Asumiendo .env está en la raíz del proyecto
load_dotenv(dotenv_path=env_path)

class DistanceService:
    _instance = None
    
    def __new__(cls, api_key: str = None, cache_file: str = "distance_cache.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Usar la API key del .env si no se pasa explícitamente
            cls._instance.api_key = api_key if api_key is not None else os.getenv('ORS_API_KEY')
            if not cls._instance.api_key:
                print("Advertencia: ORS_API_KEY no configurada. Usando distancia euclidiana.")

            cls._instance.cache_file = cache_file
            cls._instance.cache = cls._instance._load_cache()
            cls._instance._save_cache() # Guardar inicialmente si estaba vacío
        return cls._instance
    
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
             print(f"Error cargando caché {self.cache_file}: {e}")
             # Si hay error cargando, empezar con caché vacío
             return {}
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            print(f"Error guardando caché {self.cache_file}: {e}")

    
    def get_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        # Asegurarse de que las coordenadas estén en formato (Longitud, Latitud) para la API de ORS
        # El formato común es (lat, lon), pero ORS usa (lon, lat)
        ors_coord1 = (coord1[1], coord1[0]) # Convertir (lat, lon) a (lon, lat)
        ors_coord2 = (coord2[1], coord2[0])

        # Crear clave de caché consistente (usando el mismo orden de coordenadas)
        cache_key = f"{ors_coord1[0]},{ors_coord1[1]}_{ors_coord2[0]},{ors_coord2[1]}"
        reverse_key = f"{ors_coord2[0]},{ors_coord2[1]}_{ors_coord1[0]},{ors_coord1[1]}"

        # Verificar caché (en ambos órdenes posibles si no normalizamos la clave)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if reverse_key in self.cache:
             # Si encontramos la distancia en orden inverso, la guardamos en el orden normal para futuras búsquedas rápidas
             self.cache[cache_key] = self.cache[reverse_key]
             return self.cache[reverse_key]
        
        # Si no hay API key configurada, usar fallback euclidiano directamente
        if not self.api_key:
             # print("Usando fallback euclidiano (API key no configurada).")
             return self._euclidean_distance(coord1, coord2) # Usar coords originales (lat, lon) para euclidiana

        # Llamar a OpenRouteService si no está en caché y hay API key
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {
            'Authorization': self.api_key,
            'Accept': 'application/json, application/geo+json'
        }
        # Las coordenadas en params deben ser (longitude, latitude)
        params = {
            'start': f"{ors_coord1[0]},{ors_coord1[1]}", 
            'end': f"{ors_coord2[0]},{ors_coord2[1]}"
        }
        
        try:
            # print(f"Llamando a ORS para {cache_key}...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status() # Lanzar excepción para códigos de estado de error (4xx o 5xx)
            data = response.json()
            # La distancia está en metros, convertir a kilómetros
            distance = data['features'][0]['properties']['segments'][0]['distance'] / 1000.0  
            
            # Guardar en caché y en ambos órdenes para simetría (aunque ORS es direccional, a menudo es similar)
            self.cache[cache_key] = distance
            # Opcional: guardar también en orden inverso si asumimos simetría o si la API lo devuelve así
            # self.cache[reverse_key] = distance # Solo si estamos seguros de que la API devuelve distancias simétricas o muy similares

            self._save_cache()
            time.sleep(1.1)  # Espera para respetar límites de tasa (ej: 1 segundo entre llamadas)
            # print(f"Distancia ORS: {distance:.2f} km")
            return distance
        except requests.exceptions.RequestException as e:
            print(f"Error HTTP al obtener distancia de ORS: {e}")
            print("Usando fallback euclidiano.")
            # Fallback a distancia euclidiana si la API falla
            return self._euclidean_distance(coord1, coord2) # Usar coords originales (lat, lon)
        except Exception as e:
            print(f"Error inesperado al obtener distancia de ORS: {e}")
            print("Usando fallback euclidiano.")
            # Fallback a distancia euclidiana si ocurre otro error
            return self._euclidean_distance(coord1, coord2) # Usar coords originales (lat, lon)
    
    # === NOTA ===
    # La distancia euclidiana debe operar en las MISMAS unidades de coordenadas que vienen en los datos.
    # Si las coordenadas son grados decimales, la euclidiana será en grados.
    # Si necesitamos convertir a km, la fórmula de Haversine es más adecuada o usar un factor de conversión aproximado.
    # Sin embargo, dado que el fallback es solo eso (un fallback), usar la euclidiana simple sobre las coordenadas en bruto es aceptable si no se usa activamente.
    # Si necesitas un fallback en KM, la función Haversine anterior o la euclidiana * 111.0 sería mejor.
    # Mantendremos la euclidiana simple por ahora como último recurso si la API falla y no se quiere Haversine.

    def _euclidean_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calcula distancia euclidiana entre dos coordenadas (en las unidades originales)."""
        # Las coordenadas esperadas aquí son (Latitude, Longitude) como en los archivos CSV
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

# Ejemplo de uso (para depuración o si se ejecuta distance_service.py directamente)
# if __name__ == "__main__":
#     # Necesitas tener ORS_API_KEY en tu archivo .env en la raíz del proyecto
#     service = DistanceService()
# 
#     coord_depot = (4.743359, -74.153536) # Lat, Lon del depósito de BaseData
#     coord_client1 = (4.59795431125545, -74.09893796560621) # Lat, Lon del cliente 1 de BaseData
# 
#     print(f"Coordenada 1 (Lat, Lon): {coord_depot}")
#     print(f"Coordenada 2 (Lat, Lon): {coord_client1}")
# 
#     # Obtener distancia (usará ORS si hay API key, sino euclidiana)
#     dist = service.get_distance(coord_depot, coord_client1)
#     print(f"Distancia calculada: {dist:.2f} km (o en unidades euclidianas si falló la API)")
# 
#     # Probar con otra coordenada (ej: un cliente diferente)
#     coord_client2 = (4.687820646838871, -74.07557103763986) # Lat, Lon del cliente 2 de BaseData
#     dist_depot_c2 = service.get_distance(coord_depot, coord_client2)
#     print(f"Distancia Depósito a Cliente 2: {dist_depot_c2:.2f} km (o euclidiana)")
# 
#     dist_c1_c2 = service.get_distance(coord_client1, coord_client2)
#     print(f"Distancia Cliente 1 a Cliente 2: {dist_c1_c2:.2f} km (o euclidiana)")
# 
#     # Intentar obtener una distancia ya cacheadada
#     dist_cached = service.get_distance(coord_depot, coord_client1)
#     print(f"Distancia cacheadada (Depósito a Cliente 1): {dist_cached:.2f} km (o euclidiana)")
#     
#     # Verificar el contenido del caché
#     print(f"\nContenido final de la caché ({len(service.cache)} entradas): {service.cache}") 