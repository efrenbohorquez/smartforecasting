import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import ast

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

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
