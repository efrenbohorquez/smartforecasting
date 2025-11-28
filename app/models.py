import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import gradio as gr
from app.data import DF_GLOBAL

# --- FUNCIONES DE ANÁLISIS GLOBAL ---

def load_model_results():
    """Carga los resultados de los modelos desde CSV."""
    try:
        df_A = pd.read_csv('02_DATOS_ANALISIS/CSV/mejores_modelos_A.csv')
        df_B = pd.read_csv('02_DATOS_ANALISIS/CSV/mejores_modelos_B.csv')
        df_A['Producto'] = 'P9933 (A)'
        df_B['Producto'] = 'P2417 (B)'
        return pd.concat([df_A, df_B], ignore_index=True)
    except Exception as e:
        print(f"Error cargando resultados: {e}")
        return None

# Cargar resultados al inicio
DF_RESULTS = load_model_results()

# --- LÓGICA DE PREDICCIÓN ---

# Cache de modelos para evitar recargas constantes
MODEL_CACHE = {}
MAX_CACHE_SIZE = 5

def get_cached_model(ruta_modelo):
    """Obtiene un modelo del caché o lo carga si no existe."""
    if ruta_modelo in MODEL_CACHE:
        return MODEL_CACHE[ruta_modelo]
    
    # Si el caché está lleno, eliminar el más antiguo (FIFO simple para este ejemplo)
    if len(MODEL_CACHE) >= MAX_CACHE_SIZE:
        first_key = next(iter(MODEL_CACHE))
        del MODEL_CACHE[first_key]
        tf.keras.backend.clear_session() # Liberar memoria de TF
    
    print(f"Cargando modelo: {ruta_modelo}")
    modelo = tf.keras.models.load_model(ruta_modelo)
    MODEL_CACHE[ruta_modelo] = modelo
    return modelo

def get_bodega_mae(producto, bodega):
    """Obtiene el MAE del modelo para una bodega específica."""
    if DF_RESULTS is None: return 0
    
    prod_label = 'P9933 (A)' if producto == 'P9933' else 'P2417 (B)'
    row = DF_RESULTS[(DF_RESULTS['Producto'] == prod_label) & (DF_RESULTS['bodega'] == bodega)]
    if not row.empty:
        return row.iloc[0]['mae']
    return 0

def predict_bodega(producto, bodega, progress=gr.Progress()):
    """Realiza la predicción para una bodega específica."""
    progress(0, desc="Iniciando predicción...")
    
    if isinstance(DF_GLOBAL, str):
        return None, f"Error cargando datos: {DF_GLOBAL}", None, None

    # Filtrar datos
    datos = DF_GLOBAL[(DF_GLOBAL['producto'] == producto) & (DF_GLOBAL['bodega'] == bodega)]
    
    if len(datos) < 6:
        return None, "Datos insuficientes.", None, None

    # Preparar datos históricos
    historia = datos.tail(6).copy()
    valores_historia = historia['stock'].values
    fechas_historia = historia['fecha'].values
    
    # Cargar modelo
    progress(0.3, desc="Cargando modelo...")
    carpeta_modelo = "01_MODELOS/Producto_P9933" if producto == "P9933" else "01_MODELOS/Producto_P2417"
    ruta_modelo = f"{carpeta_modelo}/bodega_{bodega}/best_model.keras"
    
    if not os.path.exists(ruta_modelo):
        return None, f"No hay modelo entrenado para la bodega {bodega}", None, None
    
    try:
        modelo = get_cached_model(ruta_modelo)
        
        # Preprocesamiento
        progress(0.6, desc="Procesando datos...")
        scaler = MinMaxScaler()
        scaler.fit(valores_historia.reshape(-1, 1))
        entrada_norm = scaler.transform(valores_historia.reshape(-1, 1))
        entrada_tensor = entrada_norm.reshape(1, 6, 1)
        
        # Predicción
        progress(0.8, desc="Generando predicción...")
        pred_norm = modelo.predict(entrada_tensor, verbose=0)
        pred_real = scaler.inverse_transform(pred_norm)[0][0]
        
        # Obtener MAE
        mae = get_bodega_mae(producto, bodega)
        
        # Preparar datos para gráfica
        ultima_fecha = pd.to_datetime(fechas_historia[-1])
        fecha_pred = ultima_fecha + pd.DateOffset(months=1)
        
        fechas_plot = list(fechas_historia) + [fecha_pred]
        valores_plot = list(valores_historia) + [pred_real]
        tipos = ['Histórico']*6 + ['Predicción']
        
        df_plot = pd.DataFrame({
            'Fecha': fechas_plot,
            'Demanda': valores_plot,
            'Tipo': tipos
        })
        
        # Métricas
        cambio = pred_real - valores_historia[-1]
        cambio_pct = (cambio / valores_historia[-1]) * 100 if valores_historia[-1] != 0 else 0
        
        metrics_text = f"""
        ### 🔮 Resultados Predicción
        - **Bodega:** {bodega}
        - **Predicción ({fecha_pred.strftime('%b %Y')}):** {pred_real:.0f}
        - **Rango Estimado:** {pred_real-mae:.0f} - {pred_real+mae:.0f} (MAE: {mae:.2f})
        - **Cambio:** {cambio:+.0f} ({cambio_pct:+.1f}%)
        """
        
        stats_text = f"""
        ### 📉 Estadísticas Históricas (6 meses)
        - **Promedio:** {valores_historia.mean():.1f}
        - **Mínimo:** {valores_historia.min():.0f}
        - **Máximo:** {valores_historia.max():.0f}
        - **Desviación:** {valores_historia.std():.1f}
        """
        
        progress(1.0, desc="¡Listo!")
        return df_plot, metrics_text, stats_text, mae
        
    except Exception as e:
        return None, f"Error en predicción: {str(e)}", None, None
