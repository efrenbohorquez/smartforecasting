import gradio as gr
from app.data import get_bodegas, load_global_stats
from app.models import predict_bodega, DF_RESULTS
from app.plots import plot_forecast_interactive, plot_mae_distribution, get_hyperparameter_stats

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

def create_demo():
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
    return demo
