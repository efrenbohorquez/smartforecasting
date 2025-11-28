# -*- coding: utf-8 -*-
"""
DEMO: Uso de Modelos LSTM Entrenados para Prediccion de Demanda
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EJEMPLO 1: Prediccion para Producto P9933 - Bodega BDG-19GNI")
print("=" * 70)

# Cargar modelo
modelo_A = tf.keras.models.load_model('modelos_A/bodega_BDG-19GNI/best_model.keras')
print("\n[OK] Modelo cargado")

# Datos historicos (ultimos 6 meses)
demanda_historica = np.array([150, 180, 200, 175, 190, 220])
print("\nDemanda historica (ultimos 6 meses):")
print(f"  Mes 1: {demanda_historica[0]} unidades")
print(f"  Mes 2: {demanda_historica[1]} unidades")
print(f"  Mes 3: {demanda_historica[2]} unidades")
print(f"  Mes 4: {demanda_historica[3]} unidades")
print(f"  Mes 5: {demanda_historica[4]} unidades")
print(f"  Mes 6: {demanda_historica[5]} unidades")

# Normalizar
scaler = MinMaxScaler()
scaler.fit(demanda_historica.reshape(-1, 1))
demanda_norm = scaler.transform(demanda_historica.reshape(-1, 1))

# Preparar entrada
entrada = demanda_norm.reshape(1, 6, 1)
print(f"\nForma de entrada al modelo: {entrada.shape}")

# Predecir
prediccion_norm = modelo_A.predict(entrada, verbose=0)
prediccion_real = scaler.inverse_transform(prediccion_norm.reshape(-1, 1))[0][0]

print(f"\n>>> PREDICCION MES 7: {prediccion_real:.0f} unidades <<<")
print("=" * 70)

print("\n" + "=" * 70)
print("EJEMPLO 2: Prediccion para Producto P2417 - Bodega BDG-19GNI")
print("=" * 70)

# Cargar modelo B
modelo_B = tf.keras.models.load_model('modelos_B/bodega_BDG-19GNI/best_model.keras')
print("\n[OK] Modelo cargado")

demanda_historica_B = np.array([80, 95, 110, 88, 100, 115])
print("\nDemanda historica (ultimos 6 meses):")
print(f"  {demanda_historica_B}")

# Normalizar y predecir
scaler_B = MinMaxScaler()
scaler_B.fit(demanda_historica_B.reshape(-1, 1))
demanda_norm_B = scaler_B.transform(demanda_historica_B.reshape(-1, 1))
entrada_B = demanda_norm_B.reshape(1, 6, 1)
prediccion_norm_B = modelo_B.predict(entrada_B, verbose=0)
prediccion_real_B = scaler_B.inverse_transform(prediccion_norm_B.reshape(-1, 1))[0][0]

print(f"\n>>> PREDICCION MES 7: {prediccion_real_B:.0f} unidades <<<")
print("=" * 70)

print("\n" + "=" * 70)
print("METRICAS DE RENDIMIENTO")
print("=" * 70)

# Leer metricas
metricas_A = pd.read_csv('mejores_modelos_A.csv')
metricas_B = pd.read_csv('mejores_modelos_B.csv')

print("\nTop 3 modelos con MENOR error (MAE) - Producto A:")
top_A = metricas_A.nsmallest(3, 'mae')[['bodega', 'mae', 'loss']]
for idx, row in top_A.iterrows():
    print(f"  {row['bodega']}: MAE = {row['mae']:.6f}, Loss = {row['loss']:.6f}")

print("\nTop 3 modelos con MENOR error (MAE) - Producto B:")
top_B = metricas_B.nsmallest(3, 'mae')[['bodega', 'mae', 'loss']]
for idx, row in top_B.iterrows():
    print(f"  {row['bodega']}: MAE = {row['mae']:.6f}, Loss = {row['loss']:.6f}")

print("\n" + "=" * 70)
print("[OK] Demo completada exitosamente")
print("=" * 70)
print("\nArchivos disponibles:")
print(f"  - {len(metricas_A)} modelos para producto P9933")
print(f"  - {len(metricas_B)} modelos para producto P2417")
print(f"  - Total: {len(metricas_A) + len(metricas_B)} modelos entrenados")
