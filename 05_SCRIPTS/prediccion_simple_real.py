# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("Cargando datos reales...")

# Cargar datos del Excel
url = "https://github.com/OscarT231/Proyecto-deep-/raw/refs/heads/main/Base_filtrada.xlsx"
df = pd.read_excel(url)

# Preparar datos
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

print("OK - Datos cargados")
print(f"Total registros: {len(df_long)}")

# Prediccion para bodega especifica
bodega = "BDG-19GNI"
producto = "P9933"

print("\n" + "="*60)
print(f"PREDICCION REAL: {producto} - {bodega}")
print("="*60)

datos = df_long[(df_long['bodega'] == bodega) & (df_long['producto'] == producto)].copy()

if len(datos) >= 6:
    ultimos_6 = datos['stock'].tail(6).values
    
    print("\nDemanda historica REAL:")
    for i, val in enumerate(ultimos_6, 1):
        print(f"  Mes {i}: {int(val)} unidades")
    
    # Normalizar y predecir
    scaler = MinMaxScaler()
    scaler.fit(ultimos_6.reshape(-1, 1))
    norm = scaler.transform(ultimos_6.reshape(-1, 1))
    
    modelo = tf.keras.models.load_model(f'modelos_A/bodega_{bodega}/best_model.keras')
    pred = modelo.predict(norm.reshape(1, 6, 1), verbose=0)
    pred_real = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
    
    print(f"\n>>> PREDICCION REAL: {int(pred_real)} unidades <<<")
    print(f"Cambio vs ultimo mes: {int(pred_real - ultimos_6[-1]):+d} unidades")

# Top bodegas
print("\n" + "="*60)
print("TOP 5 BODEGAS CON MAYOR DEMANDA PREDICHA - Producto P9933")
print("="*60)

df_P9933 = df_long[df_long['producto'] == 'P9933']
resultados = []

for bod in df_P9933['bodega'].unique()[:10]:
    d = df_P9933[df_P9933['bodega'] == bod]
    if len(d) >= 6:
        try:
            u6 = d['stock'].tail(6).values
            sc = MinMaxScaler()
            sc.fit(u6.reshape(-1, 1))
            n = sc.transform(u6.reshape(-1, 1))
            m = tf.keras.models.load_model(f'modelos_A/bodega_{bod}/best_model.keras')
            p = m.predict(n.reshape(1, 6, 1), verbose=0)
            pr = sc.inverse_transform(p.reshape(-1, 1))[0][0]
            resultados.append({'bodega': bod, 'actual': int(u6[-1]), 'prediccion': int(pr)})
        except:
            pass

df_res = pd.DataFrame(resultados).sort_values('prediccion', ascending=False)
print("\n", df_res.head().to_string(index=False))

print("\n" + "="*60)
print("DATOS 100% REALES DE TU EXCEL")
print("="*60)
