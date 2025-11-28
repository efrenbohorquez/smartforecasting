"""
EJEMPLO DE USO DE LOS MODELOS ENTRENADOS
==========================================
Este script muestra cómo cargar y usar los modelos LSTM entrenados
para hacer predicciones de demanda de inventario.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==================================================
# EJEMPLO 1: PREDICCIÓN PARA PRODUCTO P9933 (A)
# ==================================================

print("=" * 60)
print("EJEMPLO 1: Predicción para Producto P9933 - Bodega BDG-19GNI")
print("=" * 60)

# 1. Cargar el modelo entrenado
modelo_A = tf.keras.models.load_model('modelos_A/bodega_BDG-19GNI/best_model.keras')
print("\n[OK] Modelo cargado exitosamente")
print(f"  Arquitectura: {modelo_A.summary()}\n")

# 2. Datos de ejemplo: últimos 6 meses de demanda real
# (En producción, estos vendrían de tu base de datos)
demanda_historica = np.array([150, 180, 200, 175, 190, 220])  # Ejemplo en unidades
print("Demanda histórica (últimos 6 meses):")
print(f"  {demanda_historica}")

# 3. Normalizar los datos (el modelo fue entrenado con datos normalizados)
scaler = MinMaxScaler()
# Necesitamos entrenar el scaler con datos históricos completos de esa bodega
# Por simplicidad, aquí uso min-max de estos 6 meses
scaler.fit(demanda_historica.reshape(-1, 1))
demanda_normalizada = scaler.transform(demanda_historica.reshape(-1, 1))

# 4. Preparar formato de entrada: (1, 6, 1)
#    1 muestra, 6 timesteps (meses), 1 feature (demanda)
entrada = demanda_normalizada.reshape(1, 6, 1)
print(f"\nDatos normalizados preparados con forma: {entrada.shape}")

# 5. Hacer la predicción
prediccion_norm = modelo_A.predict(entrada, verbose=0)
print(f"Predicción normalizada: {prediccion_norm[0][0]:.4f}")

# 6. Desnormalizar para obtener la demanda real predicha
prediccion_real = scaler.inverse_transform(prediccion_norm.reshape(-1, 1))[0][0]
print(f"\n>>> PREDICCION PROXIMO MES: {prediccion_real:.0f} unidades")

# ==================================================
# EJEMPLO 2: PREDICCIÓN PARA PRODUCTO P2417 (B)
# ==================================================

print("\n" + "=" * 60)
print("EJEMPLO 2: Predicción para Producto P2417 - Bodega BDG-19GNI")
print("=" * 60)

# 1. Cargar modelo del producto B
modelo_B = tf.keras.models.load_model('modelos_B/bodega_BDG-19GNI/best_model.keras')
print("\n[OK] Modelo cargado exitosamente")

# 2. Datos históricos diferentes
demanda_historica_B = np.array([80, 95, 110, 88, 100, 115])
print("Demanda histórica (últimos 6 meses):")
print(f"  {demanda_historica_B}")

# 3. Normalizar
scaler_B = MinMaxScaler()
scaler_B.fit(demanda_historica_B.reshape(-1, 1))
demanda_normalizada_B = scaler_B.transform(demanda_historica_B.reshape(-1, 1))

# 4. Preparar entrada
entrada_B = demanda_normalizada_B.reshape(1, 6, 1)

# 5. Predecir
prediccion_norm_B = modelo_B.predict(entrada_B, verbose=0)

# 6. Desnormalizar
prediccion_real_B = scaler_B.inverse_transform(prediccion_norm_B.reshape(-1, 1))[0][0]
print(f"\n>>> PREDICCION PROXIMO MES: {prediccion_real_B:.0f} unidades")

# ==================================================
# EJEMPLO 3: PREDICCIÓN PARA TODAS LAS BODEGAS
# ==================================================

print("\n" + "=" * 60)
print("EJEMPLO 3: Predicciones para todas las bodegas del Producto A")
print("=" * 60)

import os

# Listar todas las bodegas disponibles
bodegas_A = os.listdir('modelos_A')
print(f"\nBodegas disponibles: {len(bodegas_A)}")

# Hacer predicción para las primeras 5 bodegas (ejemplo)
resultados = []

for bodega in bodegas_A[:5]:
    modelo_path = f'modelos_A/{bodega}/best_model.keras'
    modelo = tf.keras.models.load_model(modelo_path)
    
    # Usar datos de ejemplo (en producción, usar datos reales por bodega)
    datos_ejemplo = np.array([100, 120, 110, 130, 125, 140])
    scaler_temp = MinMaxScaler()
    scaler_temp.fit(datos_ejemplo.reshape(-1, 1))
    datos_norm = scaler_temp.transform(datos_ejemplo.reshape(-1, 1))
    
    pred_norm = modelo.predict(datos_norm.reshape(1, 6, 1), verbose=0)
    pred_real = scaler_temp.inverse_transform(pred_norm.reshape(-1, 1))[0][0]
    
    resultados.append({
        'bodega': bodega.replace('bodega_', ''),
        'prediccion': pred_real
    })

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)
print("\nPredicciones por bodega:")
print(df_resultados.to_string(index=False))

# ==================================================
# MÉTRICAS DE RENDIMIENTO
# ==================================================

print("\n" + "=" * 60)
print("MÉTRICAS DE RENDIMIENTO DE LOS MODELOS")
print("=" * 60)

# Leer métricas guardadas
metricas_A = pd.read_csv('mejores_modelos_A.csv')
metricas_B = pd.read_csv('mejores_modelos_B.csv')

print("\nTop 5 modelos con MENOR error (MAE) - Producto A:")
print(metricas_A.nsmallest(5, 'mae')[['bodega', 'mae', 'loss']])

print("\nTop 5 modelos con MENOR error (MAE) - Producto B:")
print(metricas_B.nsmallest(5, 'mae')[['bodega', 'mae', 'loss']])

print("\n" + "=" * 60)
print("[OK] Ejemplos completados")
print("=" * 60)
