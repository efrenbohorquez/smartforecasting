# 📓 Cuaderno Original del Proyecto

## Descripción

Este es el **cuaderno Jupyter original** utilizado para desarrollar el sistema de predicción de demanda con LSTM.

---

## Archivo

**Nombre:** `fase_final_red_neuronal_viernes21.ipynb`  
**Tamaño:** ~3.7 MB  
**Formato:** Jupyter Notebook (.ipynb)  
**Versión:** Final - Viernes 21 de Noviembre

---

## Contenido del Cuaderno

El cuaderno está organizado en **12 secciones principales**:

### 1. Instalación de Dependencias
```python
!pip install keras_tuner
```

### 2. Carga de Librerías
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Keras Tuner

### 3. Carga y Preparación de Datos
- Lectura del archivo Excel desde GitHub
- Filtrado de productos y categorías
- Transformación Wide → Long

### 4. Creación de Diccionarios por Bodega
- Separación de datos por bodega
- Creación de diccionarios para P9933 y P2417

### 5. Normalización
- Aplicación de MinMaxScaler
- Escalado 0-1 por bodega

### 6. Creación de Ventanas Temporales
- Sliding window (6 meses input, 1 mes output)
- Formato para LSTM

### 7. Split Temporal
- División Train/Val/Test respetando cronología
- Test: últimos 2 meses
- Val: 2 meses anteriores

### 8. Optimización con Keras Tuner
- RandomSearch para hiperparámetros
- Búsqueda de unidades LSTM y learning rate

### 9. Entrenamiento de Modelos
- 52 modelos LSTM individuales
- EarlyStopping y ModelCheckpoint
- Guardado automático

### 10. Evaluación en Test
- Cálculo de MAE y Loss
- Extracción de hiperparámetros
- Exportación a CSV

### 11. Predicciones Futuras
- Uso de últimos 6 meses
- Predicción para mes siguiente
- Desnormalización de resultados

### 12. Visualizaciones
- Gráficos de predicción vs historia
- 54 gráficas PNG generadas
- Guardado automático

---

## Cómo Usar

### Opción 1: Ejecutar en Jupyter Notebook
```bash
jupyter notebook fase_final_red_neuronal_viernes21.ipynb
```

### Opción 2: Ejecutar en Google Colab
1. Subir el archivo a Google Drive
2. Abrir con Google Colaboratory
3. Ejecutar celdas secuencialmente

### Opción 3: Ejecutar en VS Code
1. Instalar extensión de Jupyter
2. Abrir archivo .ipynb
3. Seleccionar kernel Python 3.11+

---

## Requisitos

### Python
- **Versión:** 3.11 o superior

### Librerías Principales
```
tensorflow >= 2.15.0
keras-tuner >= 1.4.0
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
openpyxl >= 3.1.0
```

### Instalación
```bash
pip install tensorflow keras-tuner pandas numpy scikit-learn matplotlib seaborn openpyxl
```

---

## Tiempo de Ejecución

| Sección | Tiempo Aprox |
|---------|--------------|
| Carga de datos | 1-2 min |
| Preparación | 2-3 min |
| Optimización Keras Tuner | 3-4 horas |
| Entrenamiento final | 3-4 horas |
| Evaluación | 5-10 min |
| Predicciones y gráficas | 10-15 min |
| **TOTAL** | **~8 horas** |

*Nota: Tiempo en GPU. En CPU puede ser 3-5x más lento.*

---

## Datos de Entrada

**Archivo:** `Base_filtrada.xlsx`  
**Ubicación:** GitHub (URL incluida en el cuaderno)  
**Tamaño:** ~10 MB  
**Registros:** 130,092  

---

## Salidas Generadas

### Modelos
- `modelos_entrenados/` - 52 archivos .keras
- `modelos_A/` - 28 carpetas organizadas
- `modelos_B/` - 24 carpetas organizadas

### Datos
- `mejores_modelos_A.csv`
- `mejores_modelos_B.csv`

### Gráficas
- 54 archivos PNG con predicciones

---

## Notas Importantes

⚠️ **El cuaderno original usa comandos de instalación:**
```python
!pip install keras_tuner
```
Estos funcionan en Jupyter/Colab. Para scripts Python, comente estas líneas.

⚠️ **Paths relativos:**
El cuaderno guarda archivos en el directorio actual. Ajuste paths si es necesario.

⚠️ **Recursos de GPU:**
Recomendado ejecutar en GPU para optimización de tiempo.

---

## Script Python Convertido

Si prefiere ejecutar como script Python sin celdas:
- **Ubicación:** `05_SCRIPTS/fase_final_red_neuronal_converted.py`
- **Ventajas:** Más rápido, sin interfaz Jupyter
- **Uso:** `python fase_final_red_neuronal_converted.py`

---

## Diferencias con el Script Convertido

| Aspecto | Cuaderno (.ipynb) | Script (.py) |
|---------|-------------------|--------------|
| Ejecución | Celda por celda | Todo de una vez |
| Visualizaciones | Interactivas | Guardadas como PNG |
| Debugging | Más fácil | Más difícil |
| Producción | No recomendado | Recomendado |
| Documentación | Markdown + código | Solo código |

---

## Soporte

Para dudas sobre el cuaderno:
1. Revisar comentarios en cada celda
2. Consultar `04_DOCUMENTACION/Informes_Tecnicos/informe_tecnico_completo.md`
3. Ver `walkthrough.md` para guía paso a paso

---

## Licencia

Material académico - Proyecto de Maestría en Deep Learning  
Noviembre 2025
