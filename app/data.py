import pandas as pd
import os
import json

# --- FUNCIONES DE CARGA DE DATOS ---

def load_data():
    """Carga y procesa los datos del Excel con caché local."""
    print("Cargando datos...")
    cache_file = "base_datos_cache.xlsx"
    
    try:
        if os.path.exists(cache_file):
            print(f"Usando caché local: {cache_file}")
            df = pd.read_excel(cache_file)
        else:
            print("Descargando desde GitHub...")
            url = "https://github.com/OscarT231/Proyecto-deep-/raw/refs/heads/main/Base_filtrada.xlsx"
            df = pd.read_excel(url)
            print("Guardando caché local...")
            df.to_excel(cache_file, index=False)
        
        # Limpieza básica igual que en los scripts de análisis
        df.columns = df.columns.astype(str).str.strip()
        columnas = [
            "bodega", "producto", "calificacion_abc",
            "2024-09-01 00:00:00","2024-10-01 00:00:00","2024-11-01 00:00:00","2024-12-01 00:00:00",
            "2025-01-01 00:00:00","2025-02-01 00:00:00","2025-03-01 00:00:00","2025-04-01 00:00:00",
            "2025-05-01 00:00:00","2025-06-01 00:00:00","2025-07-01 00:00:00","2025-08-01 00:00:00"
        ]
        df = df[[col for col in columnas if col in df.columns]].copy()
        df = df[~df["calificacion_abc"].isin(["O", "N"])].copy()
        
        # Formato long
        id_cols = ["bodega", "producto", "calificacion_abc"]
        date_cols = [c for c in df.columns if c not in id_cols]
        df_long = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="fecha", value_name="stock")
        df_long['fecha'] = pd.to_datetime(df_long['fecha'])
        df_long = df_long.sort_values(["bodega", "producto", "fecha"])
        
        return df_long
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return str(e)

# Variable global para caché de datos
DF_GLOBAL = load_data()

def get_bodegas(producto):
    """Retorna las bodegas disponibles para un producto."""
    if isinstance(DF_GLOBAL, str): return []
    return sorted(DF_GLOBAL[DF_GLOBAL['producto'] == producto]['bodega'].unique().tolist())

def load_global_stats():
    """Carga estadísticas globales desde el JSON si existe."""
    path = '02_DATOS_ANALISIS/JSON/estadisticas_globales.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
