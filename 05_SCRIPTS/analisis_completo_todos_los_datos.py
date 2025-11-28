# -*- coding: utf-8 -*-
"""
ANALISIS COMPLETO DEL CONJUNTO DE DATOS
========================================
Análisis exhaustivo de todos los productos y bodegas
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALISIS COMPLETO - SISTEMA DE PREDICCION DE DEMANDA")
print("="*80)

# ============================================
# CARGAR DATOS REALES
# ============================================
print("\n[1/6] Cargando datos del archivo Excel...")
url = "https://github.com/OscarT231/Proyecto-deep-/raw/refs/heads/main/Base_filtrada.xlsx"
df = pd.read_excel(url)

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

print(f"   Total de registros cargados: {len(df_long):,}")
print(f"   Productos unicos: {df_long['producto'].nunique()}")
print(f"   Bodegas uniques: {df_long['bodega'].nunique()}")

# ============================================
# ANALISIS PRODUCTO A (P9933)
# ============================================
print("\n[2/6] Analizando Producto P9933 (Categoria A)...")
df_P9933 = df_long[df_long['producto'] == 'P9933']

resultados_A = []
for bodega in df_P9933['bodega'].unique():
    datos = df_P9933[df_P9933['bodega'] == bodega]
    if len(datos) >= 6:
        try:
            ultimos_6 = datos['stock'].tail(6).values
            scaler = MinMaxScaler()
            scaler.fit(ultimos_6.reshape(-1, 1))
            norm = scaler.transform(ultimos_6.reshape(-1, 1))
            
            modelo = tf.keras.models.load_model(f'modelos_A/bodega_{bodega}/best_model.keras')
            pred = modelo.predict(norm.reshape(1, 6, 1), verbose=0)
            pred_real = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
            
            resultados_A.append({
                'bodega': bodega,
                'demanda_promedio': ultimos_6.mean(),
                'demanda_min': ultimos_6.min(),
                'demanda_max': ultimos_6.max(),
                'feb_2025': ultimos_6[-1],
                'mar_2025_pred': pred_real,
                'cambio_abs': pred_real - ultimos_6[-1],
                'cambio_pct': ((pred_real - ultimos_6[-1]) / ultimos_6[-1]) * 100
            })
        except Exception as e:
            pass

df_A = pd.DataFrame(resultados_A)
print(f"   Bodegas analizadas: {len(df_A)}")
print(f"   Demanda total predicha: {df_A['mar_2025_pred'].sum():.0f} unidades")

# ============================================
# ANALISIS PRODUCTO B (P2417)
# ============================================
print("\n[3/6] Analizando Producto P2417 (Categoria B)...")
df_P2417 = df_long[df_long['producto'] == 'P2417']

resultados_B = []
for bodega in df_P2417['bodega'].unique():
    datos = df_P2417[df_P2417['bodega'] == bodega]
    if len(datos) >= 6:
        try:
            ultimos_6 = datos['stock'].tail(6).values
            scaler = MinMaxScaler()
            scaler.fit(ultimos_6.reshape(-1, 1))
            norm = scaler.transform(ultimos_6.reshape(-1, 1))
            
            modelo = tf.keras.models.load_model(f'modelos_B/bodega_{bodega}/best_model.keras')
            pred = modelo.predict(norm.reshape(1, 6, 1), verbose=0)
            pred_real = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
            
            resultados_B.append({
                'bodega': bodega,
                'demanda_promedio': ultimos_6.mean(),
                'demanda_min': ultimos_6.min(),
                'demanda_max': ultimos_6.max(),
                'feb_2025': ultimos_6[-1],
                'mar_2025_pred': pred_real,
                'cambio_abs': pred_real - ultimos_6[-1],
                'cambio_pct': ((pred_real - ultimos_6[-1]) / ultimos_6[-1]) * 100
            })
        except Exception as e:
            pass

df_B = pd.DataFrame(resultados_B)
print(f"   Bodegas analizadas: {len(df_B)}")
print(f"   Demanda total predicha: {df_B['mar_2025_pred'].sum():.0f} unidades")

# ============================================
# METRICAS DE RENDIMIENTO
# ============================================
print("\n[4/6] Analizando metricas de rendimiento...")
metricas_A = pd.read_csv('mejores_modelos_A.csv')
metricas_B = pd.read_csv('mejores_modelos_B.csv')

mae_promedio_A = metricas_A['mae'].mean()
mae_promedio_B = metricas_B['mae'].mean()

print(f"   MAE promedio Producto A: {mae_promedio_A:.6f}")
print(f"   MAE promedio Producto B: {mae_promedio_B:.6f}")

# ============================================
# ESTADISTICAS GLOBALES
# ============================================
print("\n[5/6] Calculando estadisticas globales...")

estadisticas = {
    'producto_A': {
        'bodegas': int(len(df_A)),
        'demanda_total_feb': float(df_A['feb_2025'].sum()),
        'demanda_total_mar_pred': float(df_A['mar_2025_pred'].sum()),
        'demanda_promedio': float(df_A['demanda_promedio'].mean()),
        'crecimiento_total': float(df_A['mar_2025_pred'].sum() - df_A['feb_2025'].sum()),
        'mae_promedio': float(mae_promedio_A)
    },
    'producto_B': {
        'bodegas': int(len(df_B)),
        'demanda_total_feb': float(df_B['feb_2025'].sum()),
        'demanda_total_mar_pred': float(df_B['mar_2025_pred'].sum()),
        'demanda_promedio': float(df_B['demanda_promedio'].mean()),
        'crecimiento_total': float(df_B['mar_2025_pred'].sum() - df_B['feb_2025'].sum()),
        'mae_promedio': float(mae_promedio_B)
    },
    'global': {
        'total_bodegas': int(len(df_A) + len(df_B)),
        'total_modelos': 52,
        'registros_analizados': int(len(df_long)),
        'demanda_total_predicha': float(df_A['mar_2025_pred'].sum() + df_B['mar_2025_pred'].sum())
    }
}

# ============================================
# EXPORTAR RESULTADOS
# ============================================
print("\n[6/6] Exportando resultados...")

# Guardar DataFrames
df_A.to_csv('analisis_completo_producto_A.csv', index=False)
df_B.to_csv('analisis_completo_producto_B.csv', index=False)

# Guardar estadísticas
with open('estadisticas_globales.json', 'w', encoding='utf-8') as f:
    json.dump(estadisticas, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print("RESUMEN EJECUTIVO")
print("="*80)
print(f"\nProducto P9933 (Categoria A):")
print(f"  - Bodegas analizadas: {estadisticas['producto_A']['bodegas']}")
print(f"  - Demanda Feb 2025: {estadisticas['producto_A']['demanda_total_feb']:.0f} unidades")
print(f"  - Prediccion Mar 2025: {estadisticas['producto_A']['demanda_total_mar_pred']:.0f} unidades")
print(f"  - Crecimiento: {estadisticas['producto_A']['crecimiento_total']:+.0f} unidades")

print(f"\nProducto P2417 (Categoria B):")
print(f"  - Bodegas analizadas: {estadisticas['producto_B']['bodegas']}")
print(f"  - Demanda Feb 2025: {estadisticas['producto_B']['demanda_total_feb']:.0f} unidades")
print(f"  - Prediccion Mar 2025: {estadisticas['producto_B']['demanda_total_mar_pred']:.0f} unidades")
print(f"  - Crecimiento: {estadisticas['producto_B']['crecimiento_total']:+.0f} unidades")

print(f"\nTOTAL GENERAL:")
print(f"  - Demanda predicha (ambos productos): {estadisticas['global']['demanda_total_predicha']:.0f} unidades")
print(f"  - Modelos entrenados: {estadisticas['global']['total_modelos']}")
print(f"  - Precision promedio (MAE): {(mae_promedio_A + mae_promedio_B)/2:.6f}")

print("\n" + "="*80)
print("Archivos generados:")
print("  - analisis_completo_producto_A.csv")
print("  - analisis_completo_producto_B.csv")
print("  - estadisticas_globales.json")
print("="*80)
