import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

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

def plot_mae_distribution(df_results):
    """Genera histograma de distribución de MAE."""
    if df_results is None: return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_results, x='mae', hue='Producto', kde=True, element="step", ax=ax)
    ax.set_title("Distribución del Error Absoluto Medio (MAE) por Producto")
    ax.set_xlabel("MAE (Menor es mejor)")
    ax.set_ylabel("Cantidad de Modelos")
    return fig

def get_hyperparameter_stats(df_results):
    """Analiza los hiperparámetros más usados."""
    if df_results is None: return None, None
    
    import ast
    
    # Parsear la columna de hiperparámetros
    units = []
    lrs = []
    
    for hp_str in df_results['hp_clean']:
        try:
            hp = ast.literal_eval(hp_str)
            units.append(hp.get('lstm_units'))
            lrs.append(hp.get('learning_rate'))
        except:
            pass
            
    # Crear DataFrames para gráficas
    df_units = pd.DataFrame({'Unidades LSTM': units})
    df_lrs = pd.DataFrame({'Learning Rate': lrs})
    
    # Gráfica Unidades
    fig_units, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_units, x='Unidades LSTM', ax=ax1, palette='viridis')
    ax1.set_title("Distribución de Unidades LSTM")
    ax1.set_ylabel("Frecuencia")
    
    # Gráfica Learning Rates
    fig_lrs, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_lrs, x='Learning Rate', ax=ax2, palette='magma')
    ax2.set_title("Distribución de Learning Rates")
    ax2.set_ylabel("Frecuencia")
    ax2.tick_params(axis='x', rotation=45)
    
    return fig_units, fig_lrs

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

def predict_bodega(producto, bodega, progress=gr.Progress()):
    """Realiza la predicción para una bodega específica."""
    progress(0, desc="Iniciando predicción...")
    
    if isinstance(DF_GLOBAL, str):
        return None, f"Error cargando datos: {DF_GLOBAL}", None

    # Filtrar datos
    datos = DF_GLOBAL[(DF_GLOBAL['producto'] == producto) & (DF_GLOBAL['bodega'] == bodega)]
    
    if len(datos) < 6:
        return None, "Datos insuficientes para esta bodega (se requieren al menos 6 meses).", None

    # Preparar datos históricos (últimos 6 meses conocidos)
    historia = datos.tail(6).copy()
    valores_historia = historia['stock'].values
    fechas_historia = historia['fecha'].values
    
    # Cargar modelo
    progress(0.3, desc="Cargando modelo...")
    carpeta_modelo = "01_MODELOS/Producto_P9933" if producto == "P9933" else "01_MODELOS/Producto_P2417"
    ruta_modelo = f"{carpeta_modelo}/bodega_{bodega}/best_model.keras"
    
    if not os.path.exists(ruta_modelo):
        return None, f"No hay modelo entrenado para la bodega {bodega}", None
    
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
        ### Resultados Predicción
        - **Bodega:** {bodega}
        - **Último Mes Real ({ultima_fecha.strftime('%b %Y')}):** {valores_historia[-1]:.0f}
        - **Predicción ({fecha_pred.strftime('%b %Y')}):** {pred_real:.0f}
        - **Cambio:** {cambio:+.0f} ({cambio_pct:+.1f}%)
        """
        
        progress(1.0, desc="¡Listo!")
        return df_plot, metrics_text, pred_real
        
    except Exception as e:
        return None, f"Error en predicción: {str(e)}", None

import plotly.graph_objects as go

def get_bodega_mae(producto, bodega):
    """Obtiene el MAE del modelo para una bodega específica."""
    if DF_RESULTS is None: return 0
    
    prod_label = 'P9933 (A)' if producto == 'P9933' else 'P2417 (B)'
    row = DF_RESULTS[(DF_RESULTS['Producto'] == prod_label) & (DF_RESULTS['bodega'] == bodega)]
    if not row.empty:
        return row.iloc[0]['mae']
    return 0

def plot_forecast_interactive(df_plot, mae=0):
    """Genera gráfica interactiva con Plotly."""
    if df_plot is None: return None
    
    fig = go.Figure()
    
    # Histórico
    hist = df_plot[df_plot['Tipo'] == 'Histórico']
    fig.add_trace(go.Scatter(
        x=hist['Fecha'], y=hist['Demanda'],
        mode='lines+markers', name='Histórico',
        line=dict(color='#2E86C1', width=3)
    ))
    
    # Predicción
    pred = df_plot.tail(2)
    fig.add_trace(go.Scatter(
        x=pred['Fecha'], y=pred['Demanda'],
        mode='lines+markers', name='Predicción',
        line=dict(color='#E74C3C', width=3, dash='dash')
    ))
    
    # Intervalo de confianza (MAE)
    if mae > 0:
        pred_val = pred.iloc[-1]['Demanda']
        pred_date = pred.iloc[-1]['Fecha']
        
        fig.add_trace(go.Scatter(
            x=[pred_date, pred_date],
            y=[pred_val - mae, pred_val + mae],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Error bars visual hack or just use text
        fig.add_annotation(
            x=pred_date, y=pred_val + mae,
            text=f"+{mae:.2f}", showarrow=False, yshift=10, font=dict(size=10, color="gray")
        )
        fig.add_annotation(
            x=pred_date, y=pred_val - mae,
            text=f"-{mae:.2f}", showarrow=False, yshift=-10, font=dict(size=10, color="gray")
        )

    fig.update_layout(
        title="Pronóstico de Demanda (Interactivo)",
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

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

# --- INTERFAZ GRADIO ---

def actualizar_bodegas(producto):
    return gr.Dropdown(choices=get_bodegas(producto), value=None)

def ejecutar_demo(producto, bodega):
    df_plot, metrics, stats, mae = predict_bodega(producto, bodega)
    if df_plot is None:
        return None, metrics, None # metrics contiene el error
    
    fig = plot_forecast_interactive(df_plot, mae)
    return fig, metrics, stats

# Cargar stats globales para la pestaña 2
STATS = load_global_stats()

with gr.Blocks(title="Sistema de Predicción de Demanda LSTM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Sistema de Predicción de Demanda con LSTM
    **Proyecto de Maestría en Deep Learning**
    """)
    
    with gr.Tabs():
        # TAB 1: ANÁLISIS INDIVIDUAL
        with gr.Tab("📊 Análisis Individual"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuración")
                    in_prod = gr.Dropdown(choices=["P9933", "P2417"], label="Producto", value="P9933")
                    in_bodega = gr.Dropdown(choices=get_bodegas("P9933"), label="Bodega")
                    btn_pred = gr.Button("Generar Predicción", variant="primary")
                    
                    gr.Markdown("---")
                    out_metrics = gr.Markdown("Seleccione una bodega y presione 'Generar Predicción'")
                    out_stats = gr.Markdown("")
                    
                with gr.Column(scale=2):
                    out_plot = gr.Plot(label="Gráfica de Pronóstico")
            
            in_prod.change(fn=actualizar_bodegas, inputs=in_prod, outputs=in_bodega)
            btn_pred.click(fn=ejecutar_demo, inputs=[in_prod, in_bodega], outputs=[out_plot, out_metrics, out_stats])



        # TAB 2: ANÁLISIS GLOBAL
        with gr.Tab("🌍 Análisis Global"):
            if STATS:
                gr.Markdown("### Resumen de Rendimiento del Modelo")
                with gr.Row():
                    with gr.Column():
                        gr.Number(label="Total Modelos", value=STATS['global']['total_modelos'], interactive=False)
                        gr.Number(label="Precisión Promedio (MAE)", value=STATS['producto_A']['mae_promedio'], interactive=False)
                    with gr.Column():
                        gr.Number(label="Demanda Total Predicha (Mar 25)", value=STATS['global']['demanda_total_predicha'], interactive=False)
                        gr.Number(label="Registros Analizados", value=STATS['global']['registros_analizados'], interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 📊 Análisis de Errores y Arquitectura")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Distribución de Errores (MAE)")
                        gr.Plot(value=plot_mae_distribution(DF_RESULTS), label="Distribución MAE")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Hiperparámetros Óptimos")
                        fig_u, fig_lr = get_hyperparameter_stats(DF_RESULTS)
                        gr.Plot(value=fig_u, label="Unidades LSTM")
                        gr.Plot(value=fig_lr, label="Learning Rates")

                gr.Markdown("---")
                gr.Markdown("### Detalles por Producto")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Producto P9933 (A)")
                        gr.JSON(STATS['producto_A'])
                    with gr.Column():
                        gr.Markdown("#### Producto P2417 (B)")
                        gr.JSON(STATS['producto_B'])
            else:
                gr.Markdown("⚠️ No se encontró el archivo `estadisticas_globales.json`. Ejecute primero el script de análisis completo.")

        # TAB 3: DETALLES TÉCNICOS
        with gr.Tab("🛠️ Detalles Técnicos"):
            gr.Markdown("## 🏗️ Arquitectura y Proceso de Entrenamiento")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 1. Topología de la Red (LSTM)
                    El modelo utiliza una arquitectura **Long Short-Term Memory (LSTM)**, diseñada específicamente para problemas de series temporales.
                    
                    - **Input Layer:** Recibe una matriz de `(6, 1)`, representando 6 meses de historia con 1 característica (demanda).
                    - **LSTM Layer:** Extrae patrones temporales complejos (tendencias, estacionalidad). El número de unidades varía entre 32 y 128 según la optimización.
                    - **Dense Layer:** Una capa completamente conectada que condensa la información en un único valor de predicción.
                    - **Output:** Un escalar que representa la demanda estimada para el mes `t+1`.
                    """)
                    gr.Image("04_DOCUMENTACION/assets/network_topology.png", label="Topología de la Red", show_label=True)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 2. Pipeline de Entrenamiento
                    El proceso de construcción del modelo sigue un flujo riguroso para asegurar la robustez:
                    
                    1. **Preprocesamiento:** Normalización MinMax (0-1) y creación de ventanas deslizantes.
                    2. **Split:** División temporal en Train (Entrenamiento), Val (Validación) y Test (Prueba).
                    3. **Optimización (Keras Tuner):** Búsqueda automática de los mejores hiperparámetros (Learning Rate, Unidades LSTM).
                    4. **Entrenamiento:** Uso de `EarlyStopping` para evitar sobreajuste (overfitting).
                    5. **Evaluación:** Cálculo del MAE (Error Absoluto Medio) en el conjunto de prueba.
                    """)
                    gr.Image("04_DOCUMENTACION/assets/training_pipeline.png", label="Flujo de Entrenamiento", show_label=True)
            
            gr.Markdown("---")
            gr.Markdown("""
            ### ⚙️ Configuración de Hiperparámetros
            
            | Parámetro | Configuración | Descripción |
            |-----------|---------------|-------------|
            | **Optimizador** | Adam | Algoritmo de optimización adaptativo. |
            | **Loss Function** | MSE (Mean Squared Error) | Función de pérdida para regresión, penaliza grandes errores. |
            | **Métrica** | MAE (Mean Absolute Error) | Métrica interpretable (unidades de producto). |
            | **Ventana** | 6 Meses | Cantidad de historia usada para cada predicción. |
            | **Epochs** | 100 (con Early Stopping) | Máximo de iteraciones de entrenamiento. |
            """)

        # TAB 4: EXPLICACIÓN DIDÁCTICA
        with gr.Tab("🎓 Explicación Didáctica"):
            gr.Markdown("## 🧠 ¿Cómo funciona la Inteligencia Artificial?")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 1. La Ventana Deslizante (Sliding Window)
                    
                    Para predecir el futuro, la IA necesita "mirar" el pasado. Usamos una técnica llamada **Ventana Deslizante**.
                    
                    Imagina que tienes una ventana que solo te deja ver 6 meses a la vez.
                    - **Entrada (X):** La IA observa los meses 1 al 6.
                    - **Objetivo (y):** Intenta adivinar qué pasó en el mes 7.
                    
                    Luego, la ventana se mueve un mes a la derecha (meses 2 al 7) para predecir el mes 8, y así sucesivamente. De esta forma, transformamos una serie de tiempo en muchos ejemplos de entrenamiento.
                    """)
                    gr.Image("04_DOCUMENTACION/assets/sliding_window.png", label="Concepto de Ventana Deslizante", show_label=True)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 2. Red Neuronal LSTM
                    
                    Usamos un tipo especial de cerebro digital llamado **LSTM (Long Short-Term Memory)**.
                    
                    A diferencia de una red normal, la LSTM tiene "memoria". Puede recordar patrones importantes de hace varios meses (como una temporada alta en diciembre) y olvidar datos irrelevantes.
                    
                    - **Celdas de Memoria:** Guardan información a largo plazo.
                    - **Puertas (Gates):** Deciden qué información entra, sale o se olvida.
                    
                    Es ideal para inventarios porque entiende que la demanda de hoy depende de la tendencia de los últimos meses.
                    """)
                    gr.Image("04_DOCUMENTACION/assets/lstm_architecture.png", label="Arquitectura LSTM Simplificada", show_label=True)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
