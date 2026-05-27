# 🏠 House Price Prediction - XGBoost

Proyecto de **Machine Learning** para predecir precios de casas utilizando datos tabulares. Uno de mis proyectos principales de portafolio.

## 📊 Resultados del Modelo
- **RMSLE**: 0.13443
- **MAE**: $14,974 USD
- **R²**: 0.9032
- **Mejor modelo**: XGBoost + Feature Engineering avanzado + Target Encoding

## 🚀 Tecnologías Utilizadas
- **Análisis y Visualización**: Pandas, NumPy, Seaborn, Matplotlib
- **Modelado**: XGBoost
- **Optimización**: Optuna (en versiones anteriores)
- **Preprocessing**: Scikit-learn + Target Encoding
- **Despliegue**: Streamlit
- **Interpretabilidad**: SHAP (en notebooks)

## 📁 Estructura del Proyecto

house_price_prediction/
├── data/raw/
├── notebooks/          # EDA, Modelado y Análisis
├── src/                # Código modular
├── app/                # Aplicación Streamlit
├── models/             # Modelos entrenados
└── reports/
text

## 🎯 Características Principales
- Feature Engineering avanzado (edad de casa, áreas combinadas, interacciones de calidad, etc.)
- Preprocesamiento robusto con Target Encoding
- Pipeline completo (entrenamiento → predicción)
- Aplicación web interactiva

## ▶️ Cómo Ejecutar Localmente

```bash
# 1. Clonar el repositorio
git clone <tu-repo>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app/streamlit_app.py

Demo



---

### **B) Versión Más Bonita del Streamlit App**

Reemplaza el contenido de **`app/streamlit_app.py`** con esto:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price AI", layout="wide", page_icon="🏠")

# Header
st.title("🏠 House Price AI Predictor")
st.markdown("### Predicción inteligente de precios de casas usando XGBoost")

st.info("**RMSLE**: 0.13443 | **R²**: 0.9032", icon="📊")

# Cargar modelo
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model_final.pkl")
    preprocessor = joblib.load("models/preprocessor_final.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# Formulario
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Principales")
    overall_qual = st.slider("Calidad General de la Casa", 1, 10, 7)
    gr_liv_area = st.number_input("Área Habitable (ft²)", 500, 8000, 1600)
    total_bsmt_sf = st.number_input("Área del Sótano (ft²)", 0, 5000, 900)
    year_built = st.number_input("Año de Construcción", 1900, 2025, 1995)

with col2:
    st.subheader("Otras Características")
    garage_cars = st.slider("Plazas de Garage", 0, 4, 2)
    full_bath = st.slider("Baños Completos", 0, 5, 2)
    neighborhood = st.selectbox("Vecindario", 
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Gilbert"])
    lot_area = st.number_input("Tamaño del Terreno (ft²)", 1000, 100000, 9500)

# Predicción
if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'TotalBsmtSF': [total_bsmt_sf],
        'YearBuilt': [year_built],
        'GarageCars': [garage_cars],
        'FullBath': [full_bath],
        'Neighborhood': [neighborhood],
        'LotArea': [lot_area]
    })
    
    from src.features import feature_engineering
    input_data = feature_engineering(input_data)
    
    input_processed = preprocessor.transform(input_data)
    pred_log = model.predict(input_processed)
    prediction = np.expm1(pred_log[0])
    
    st.success(f"### Precio Estimado: **${prediction:,.2f} USD**", icon="💰")

st.caption("Proyecto de Portafolio | Machine Learning | Ana Luisa")
