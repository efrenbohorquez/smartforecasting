# -*- coding: utf-8 -*-
"""
USO DE MODELOS CON DATOS REALES
================================
Este script muestra como cargar TUS datos reales y hacer predicciones
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================================
# PASO 1: Cargar tus datos REALES
# ============================================

# Cargar el dataset original
url = "https://github.com/OscarT231/Proyecto-deep-/raw/refs/heads/main/Base_filtrada.xlsx"
df = pd.read_excel(url)

# Filtrar columnas
df.columns = df.columns.astype(str).str.strip()
columnas_deseadas = [
    "bodega", "producto", "calificacion_abc",
    "2024-09-01 00:00:00","2024-10-01 00:00:00","2024-11-01 00:00:00","2024-12-01 00:00:00",
    "2025-01-01 00:00:00","2025-02-01 00:00:00","2025-03-01 00:00:00","2025-04-01 00:00:00",
    "2025-05-01 00:00:00","2025-06-01 00:00:00","2025-07-01 00:00:00","2025-08-01 00:00:00"
]

df_sugerido = df[[col for col in columnas_deseadas if col in df.columns]].copy()
df_sugerido = df_sugerido[~df_sugerido["calificacion_abc"].isin(["O", "N"])].copy()

# Convertir a formato long
id_cols = ["bodega", "producto", "calificacion_abc"]
date_cols = [c for c in df_sugerido.columns if c not in id_cols]

df_long = df_sugerido.melt(id_vars=id_cols,
                            value_vars=date_cols,
                            var_name="fecha",
                            value_name="stock_solicitado")

df_long['fecha'] = pd.to_datetime(df_long['fecha'])
df_long = df_long.sort_values(["bodega", "producto", "fecha"])

print("Datos cargados exitosamente!")
print(f"Total de registros: {len(df_long)}")

# ============================================
# PASO 2: Hacer prediccion REAL para una bodega
# ============================================

# Ejemplo: Producto P9933, Bodega BDG-19GNI
bodega_seleccionada = "BDG-19GNI"
producto_seleccionado = "P9933"

# Filtrar datos de esa bodega y producto
datos_bodega = df_long[
    (df_long['bodega'] == bodega_seleccionada) & 
    (df_long['producto'] == producto_seleccionado)
].copy()

print(f"\n{'='*60}")
print(f"Prediccion REAL para: {producto_seleccionado} - {bodega_seleccionada}")
print(f"{'='*60}")

if len(datos_bodega) >= 6:
    # Obtener ultimos 6 meses REALES
    ultimos_6 = datos_bodega['stock_solicitado'].tail(6).values
    
    print("\nDemanda historica REAL (ultimos 6 meses):")
    for i, val in enumerate(ultimos_6, 1):
        print(f"  Mes {i}: {val:.0f} unidades")
    
    # Normalizar
    scaler = MinMaxScaler()
    scaler.fit(ultimos_6.reshape(-1, 1))
    datos_norm = scaler.transform(ultimos_6.reshape(-1, 1))
    
    # Cargar modelo
    modelo = tf.keras.models.load_model(f'modelos_A/bodega_{bodega_seleccionada}/best_model.keras')
    
    # Predecir
    entrada = datos_norm.reshape(1, 6, 1)
    pred_norm = modelo.predict(entrada, verbose=0)
    pred_real = scaler.inverse_transform(pred_norm.reshape(-1, 1))[0][0]
    
    print(f"\n>>> PREDICCION REAL PROXIMO MES: {pred_real:.0f} unidades <<<")
    print(f"{'='*60}")
else:
    print("No hay suficientes datos historicos para esta bodega")

# ============================================
# PASO 3: Predicciones para TODAS las bodegas de un producto
# ============================================

print("\n" + "="*60)
print("PREDICCIONES PARA TODAS LAS BODEGAS - Producto P9933")
print("="*60)

# Filtrar solo producto P9933
df_P9933 = df_long[df_long['producto'] == 'P9933']

resultados_reales = []

for bodega in df_P9933['bodega'].unique():
    datos_bodega = df_P9933[df_P9933['bodega'] == bodega]
    
    if len(datos_bodega) >= 6:
        ultimos_6 = datos_bodega['stock_solicitado'].tail(6).values
        
        # Normalizar
        scaler = MinMaxScaler()
        scaler.fit(ultimos_6.reshape(-1, 1))
        datos_norm = scaler.transform(ultimos_6.reshape(-1, 1))
        
        # Cargar modelo
        try:
            modelo = tf.keras.models.load_model(f'modelos_A/bodega_{bodega}/best_model.keras')
            
            # Predecir
            entrada = datos_norm.reshape(1, 6, 1)
            pred_norm = modelo.predict(entrada, verbose=0)
            pred_real = scaler.inverse_transform(pred_norm.reshape(-1, 1))[0][0]
            
            resultados_reales.append({
                'bodega': bodega,
                'ult_mes_real': ultimos_6[-1],
                'prediccion': pred_real,
                'cambio': pred_real - ultimos_6[-1]
            })
        except:
            pass

# Mostrar resultados
df_resultados = pd.DataFrame(resultados_reales)
df_resultados = df_resultados.sort_values('prediccion', ascending=False)

print("\nTop 10 bodegas con MAYOR demanda predicha:")
print(df_resultados.head(10).to_string(index=False))

print("\n" + "="*60)
print("ESTOS SON TUS DATOS REALES, NO UN DEMO")
print("="*60)
