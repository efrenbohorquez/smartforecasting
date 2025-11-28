# 🧠 Smart Forecasting - LSTM Demand Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An intelligent demand forecasting system powered by LSTM neural networks, designed for warehouse inventory optimization.**

![Smart Forecasting Banner](https://via.placeholder.com/800x200/2E86C1/FFFFFF?text=Smart+Forecasting+System)

---

## 📋 Overview

Smart Forecasting is a production-ready web application that leverages **Long Short-Term Memory (LSTM)** neural networks to predict warehouse demand with high accuracy. Built for a Master's thesis in Deep Learning, this system provides:

- ✅ **Automated demand forecasting** for 2 products across 26 warehouses
- ✅ **Interactive visualizations** with Plotly and Matplotlib
- ✅ **Real-time predictions** with confidence intervals (MAE-based)
- ✅ **Hyperparameter optimization** using Keras Tuner
- ✅ **Web-based interface** powered by Gradio
- ✅ **Production-ready deployment** on Render

---

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/efrenbohorquez/smartforecasting.git
cd smartforecasting

# Install dependencies
pip install -r requirements.txt

# Run the application
python app/main.py
```

The app will be available at `http://localhost:7860`

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

1. Fork this repository
2. Create a new **Web Service** on [Render](https://dashboard.render.com/)
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and deploy automatically

---

## 🏗️ Architecture

### Project Structure

```
smartforecasting/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── data.py              # Data loading and preprocessing
│   ├── models.py            # LSTM model handling and predictions
│   ├── plots.py             # Matplotlib/Plotly visualizations
│   ├── ui.py                # Gradio interface definition
│   └── main.py              # Application entry point
├── 01_MODELOS/              # Trained LSTM models (26 warehouses)
├── 02_DATOS_ANALISIS/       # Model evaluation metrics
├── 04_DOCUMENTACION/        # Technical documentation
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment config
└── README.md                # This file
```

### LSTM Model Architecture

```
Input Layer (6, 1)  →  LSTM Layer (32-128 units)  →  Dense Layer  →  Output (1)
```

- **Input**: 6 months of historical demand
- **Output**: Next month demand prediction
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Metric**: MAE (Mean Absolute Error)

---

## 📊 Features

### 1. Individual Warehouse Analysis
- Select product (P9933 or P2417)
- Choose specific warehouse
- View 6-month historical trend
- Get next-month prediction with confidence interval
- See statistical summary (mean, min, max, std)

### 2. Global Performance Dashboard
- Total models trained: **26**
- Average MAE: **~15 units**
- Distribution of errors across warehouses
- Hyperparameter analysis (LSTM units, learning rates)

### 3. Technical Details
- Network topology visualization
- Training pipeline explanation
- Hyperparameter configuration table

### 4. Educational Module
- Sliding window concept
- LSTM architecture simplified explanation
- How AI learns from time series

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server port (auto-set by Render) |
| `PYTHON_VERSION` | `3.9.0` | Python runtime version |

### Dependencies

Core libraries:
- `gradio` - Web UI framework
- `tensorflow-cpu` - LSTM neural network
- `scikit-learn` - Data preprocessing
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn`, `plotly` - Visualizations

---

## 📈 Model Performance

| Product | Avg MAE | Best Model | Worst Model |
|---------|---------|------------|-------------|
| P9933 (A) | 14.2 | 8.5 | 22.1 |
| P2417 (B) | 16.8 | 10.3 | 25.4 |

*MAE = Mean Absolute Error (lower is better)*

---

## 🛠️ Development

### Running Locally

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with custom port
PORT=8000 python app/main.py

# Access at http://localhost:8000
```

### Project Files

- **app/data.py**: Data loading with local cache (`base_datos_cache.xlsx`)
- **app/models.py**: Model caching to avoid reloading (FIFO cache, max 5 models)
- **app/plots.py**: Uses `Agg` backend to prevent threading issues
- **app/ui.py**: Modular Gradio interface with 4 tabs

---

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT.md) - Step-by-step Render deployment
- [Implementation Plan](04_DOCUMENTACION/implementation_plan.md) - Refactoring details
- [Technical Report](GUION_PRESENTACION_10MIN.md) - 10-minute presentation script

---

## 🎓 Academic Context

**Master's Thesis**: Deep Learning for Inventory Optimization  
**University**: [Your University Name]  
**Year**: 2024-2025  
**Author**: Efren Bohorquez

### Research Highlights
- Automated hyperparameter tuning (Keras Tuner)
- Temporal split validation (Train/Val/Test)
- Early stopping to prevent overfitting
- Production-grade model deployment

---

## 🤝 Contributing

This is an academic project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Live Demo**: [Coming Soon on Render]
- **Repository**: https://github.com/efrenbohorquez/smartforecasting
- **Documentation**: [See docs folder](04_DOCUMENTACION/)

---

## 📧 Contact

**Efren Bohorquez**  
📧 Email: efrenbohorquez@example.com  
🔗 LinkedIn: [linkedin.com/in/efrenbohorquez](https://linkedin.com/in/efrenbohorquez)  
🐙 GitHub: [@efrenbohorquez](https://github.com/efrenbohorquez)

---

<p align="center">
  Made with ❤️ using TensorFlow, Gradio, and Python
</p>
