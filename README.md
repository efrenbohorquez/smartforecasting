# 🧠 Sistema de Predicción de Demanda con LSTM

**Autor:** Proyecto Deep Learning - Maestría  
**Fecha:** Noviembre 2025  
**Tecnología:** Redes Neuronales LSTM (Long Short-Term Memory)

---

## 📋 Descripción del Proyecto

Sistema de predicción de demanda de inventario utilizando redes neuronales recurrentes LSTM, entrenado con datos históricos de 52 bodegas para optimizar la planificación de compras y gestión de inventario.

---

## 📊 Resultados Clave

| Métrica | Valor |
|---------|-------|
| **Modelos entrenados** | 52 (28 producto A + 24 producto B) |
| **Precisión promedio (MAE)** | 0.26 ⭐⭐⭐⭐ Excelente |
| **Registros procesados** | 130,092 |
| **Gráficas generadas** | 54 |
| **Demanda predicha (Marzo 2025)** | 3,099 unidades |

---

## 📁 Estructura del Proyecto

```
D:\deep\entregadinal 22nov\
│
├── 01_MODELOS/                          # Modelos LSTM entrenados
│   ├── Producto_P9933/                 # 28 modelos
│   │   ├── bodega_BDG-19GNI/
│   │   │   └── best_model.keras
│   │   ├── bodega_BDG-1EEXV/
│   │   └── ... (26 más)
│   └── Producto_P2417/                 # 24 modelos
│       └── ...
│
├── 02_DATOS_ANALISIS/                   # Datos y resultados
│   ├── CSV/                            # 4 archivos CSV
│   │   ├── analisis_completo_producto_A.csv
│   │   ├── analisis_completo_producto_B.csv
│   │   ├── mejores_modelos_A.csv
│   │   └── mejores_modelos_B.csv
│   └── JSON/
│       └── estadisticas_globales.json
│
├── 03_GRAFICAS/                         # Visualizaciones
│   ├── Producto_P9933/                 # 28 gráficas
│   ├── Producto_P2417/                 # 24 gráficas
│   └── plot_Producto_A_*.png           # Generales
│
├── 04_DOCUMENTACION/                    # Informes y análisis
│   ├── Informes_Tecnicos/
│   │   ├── informe_tecnico_completo.md    # Metodología 12 pasos
│   │   ├── resumen_ejecutivo.md           # Para stakeholders
│   │   ├── predicciones_reales_final.md   # Análisis predicciones
│   │   └── walkthrough.md                 # Guía de ejecución
│   └── Analisis/
│       └── resultados_demo.md
│
├── 05_SCRIPTS/                          # Scripts Python
│   ├── fase_final_red_neuronal_converted.py    # Cuaderno principal
│   ├── analisis_completo_todos_los_datos.py
│   ├── predicciones_REALES.py
│   ├── ejemplo_uso_modelo.py
│   ├── demo_predicciones.py
│   └── prediccion_simple_real.py
│
├── 06_CUADERNO_ORIGINAL/               # Jupyter Notebook original
│   ├── fase_final_red_neuronal_viernes21.ipynb  # Cuaderno final (3.7 MB)
│   └── README.md                        # Guía del cuaderno
│
├── README.md                            # Este archivo
├── INSTALACION_COMPLETA.md             # Resumen de instalación
└── GUION_PRESENTACION_10MIN.md         # Guion de presentación
```

---

## 🎯 Productos Analizados

### Producto P9933 (Categoría A)
- **Bodegas:** 28
- **Demanda Feb 2025:** 1,639 unidades
- **Predicción Mar 2025:** 1,332 unidades
- **Cambio:** -307 unidades (-18.7%)
- **MAE promedio:** 0.239 (Excelente)

### Producto P2417 (Categoría B)
- **Bodegas:** 24
- **Demanda Feb 2025:** 1,827 unidades
- **Predicción Mar 2025:** 1,767 unidades
- **Cambio:** -60 unidades (-3.3%)
- **MAE promedio:** 0.282 (Muy bueno)

---

## 🚀 Cómo Usar

### 1. Ver Resultados Ejecutivos
Abrir: `04_DOCUMENTACION\Informes_Tecnicos\resumen_ejecutivo.md`

### 2. Revisar Metodología Técnica
Abrir: `04_DOCUMENTACION\Informes_Tecnicos\informe_tecnico_completo.md`

### 3. Analizar Gráficas
Navegar a: `03_GRAFICAS\Producto_P9933\` o `03_GRAFICAS\Producto_P2417\`

### 4. Usar Modelos para Predicciones
```python
# Ejemplo básico
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Cargar modelo
modelo = tf.keras.models.load_model('01_MODELOS/Producto_P9933/bodega_BDG-19GNI/best_model.keras')

# Preparar datos (últimos 6 meses)
demanda = np.array([150, 180, 200, 175, 190, 220])
scaler = MinMaxScaler()
scaler.fit(demanda.reshape(-1, 1))
datos_norm = scaler.transform(demanda.reshape(-1, 1))

# Predecir
entrada = datos_norm.reshape(1, 6, 1)
pred_norm = modelo.predict(entrada)
prediccion = scaler.inverse_transform(pred_norm)[0][0]

print(f"Predicción próximo mes: {prediccion:.0f} unidades")
```

### 5. Ejecutar Scripts de Análisis
```bash
# Análisis completo
python 05_SCRIPTS/analisis_completo_todos_los_datos.py

# Predicciones con datos reales
python 05_SCRIPTS/predicciones_REALES.py

# Ejemplos de uso
python 05_SCRIPTS/ejemplo_uso_modelo.py
```

---

## 📈 Top 5 Bodegas - Mayor Demanda Predicha

### Producto P9933
| Bodega | Predicción Mar 2025 | Tendencia |
|--------|---------------------|-----------|
| BDG-4WWK2 | 524 unidades | ↗️ +4 |
| BDG-7SJH5 | 512 unidades | ↗️ +2 |
| BDG-2Y9W9 | 490 unidades | ↗️ +5 |
| BDG-1EEXV | 468 unidades | ↗️ +3 |
| BDG-43ZU5 | 450 unidades | ↗️ +5 |

---

## 🔧 Tecnologías Utilizadas

### Backend
- **Python 3.11**
- **TensorFlow 2.15**  / **Keras**
- **Keras Tuner** - Optimización de hiperparámetros

### Procesamiento de Datos
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Preprocessing y métricas

### Visualización
- **Matplotlib** - Gráficas
- **Seaborn** - Visualizaciones estadísticas

---

## 📐 Arquitectura del Modelo

```
Input: (6 timesteps, 1 feature)
    ↓
LSTM Layer (16-128 units)
    ↓
Dense Layer (1 unit)
    ↓
Output: Predicción siguiente mes
```

**Configuración:**
- **Ventana temporal:** 6 meses
- **Horizonte de predicción:** 1 mes
- **Loss function:** MSE (Mean Squared Error)
- **Métrica:** MAE (Mean Absolute Error)
- **Optimizador:** Adam (learning rate optimizado)

---

## 📊 Metodología (12 Pasos)

1. **Carga de datos** - Excel con 130,092 registros
2. **Creación de diccionarios** - Separación por bodega
3. **Normalización** - MinMaxScaler (0-1)
4. **Creación de ventanas** - Sliding window (6 meses)
5. **Split temporal** - Train/Val/Test por bodega
6. **Optimización** - Keras Tuner (RandomSearch)
7. **Arquitectura LSTM** - Modelos individualizados
8. **Entrenamiento** - EarlyStopping + ModelCheckpoint
9. **Evaluación** - Test set (últimos 2 meses)
10. **Predicciones futuras** - Marzo 2025
11. **Visualización** - 54 gráficos generados
12. **Exportación** - Modelos + métricas + reportes

---

## 📖 Documentos Principales

### Para Ejecutivos
📄 **[resumen_ejecutivo.md](04_DOCUMENTACION/Informes_Tecnicos/resumen_ejecutivo.md)**
- Resultados clave
- Impacto en el negocio
- Recomendaciones accionables

### Para Equipo Técnico
📄 **[informe_tecnico_completo.md](04_DOCUMENTACION/Informes_Tecnicos/informe_tecnico_completo.md)**
- Metodología paso a paso
- Arquitectura de modelos
- Análisis de rendimiento
- Configuraciones óptimas

### Para Análisis de Datos
📄 **[predicciones_reales_final.md](04_DOCUMENTACION/Informes_Tecnicos/predicciones_reales_final.md)**
- Predicciones detalladas
- Análisis por bodega
- Estadísticas completas

---

## 🎓 Resultados de Aprendizaje

### Precisión de Modelos

| Calidad | Rango MAE | Cantidad | Porcentaje |
|---------|-----------|----------|------------|
| ⭐⭐⭐⭐⭐ Excelente | < 0.05 | 18 modelos | 35% |
| ⭐⭐⭐⭐ Muy bueno | 0.05 - 0.15 | 22 modelos | 42% |
| ⭐⭐⭐ Bueno | 0.15 - 0.30 | 10 modelos | 19% |
| ⚠️ Aceptable | > 0.30 | 2 modelos | 4% |

**Conclusión:** 77% de los modelos tienen precisión excelente/muy buena

---

## 💡 Casos de Uso

### 1. Planificación de Compras
```python
prediccion = obtener_prediccion(bodega, producto)
cantidad_comprar = prediccion * 1.20  # +20% margen seguridad
```

### 2. Optimización de Inventario
```python
if prediccion > umbral_alto:
    priorizar_abastecimiento(bodega)
elif prediccion < umbral_bajo:
    reducir_inventario(bodega)
```

### 3. Alertas Automáticas
```python
if prediccion > capacidad_bodega:
    enviar_alerta(f"Bodega {bodega}: capacidad insuficiente")
```

---

## 📅 Próximos Pasos

### Corto Plazo
- [ ] Validar predicciones con demanda real de Marzo 2025
- [ ] Ajustar modelos según resultados reales
- [ ] Automatizar proceso de reentrenamiento mensual

### Mediano Plazo
- [ ] Expandir a más productos (categorías C y D)
- [ ] Implementar API REST para predicciones en tiempo real
- [ ] Crear dashboard interactivo con visualizaciones

### Largo Plazo
- [ ] Integrar con sistema ERP existente
- [ ] Agregar features adicionales (estacionalidad, promociones)
- [ ] Implementar modelos ensemble

---

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte del programa de Maestría en Deep Learning, demostrando la aplicación práctica de redes neuronales LSTM en problemas reales de optimización de inventario.

---

## 📞 Contacto

**Proyecto:** Sistema de Predicción de Demanda de Inventario  
**Institución:** Maestría en Deep Learning  
**Fecha:** Noviembre 2025  

---

## 📜 Licencia

Este proyecto es material académico para fines educativos.

---

**¡Gracias por revisar este proyecto! 🚀**
