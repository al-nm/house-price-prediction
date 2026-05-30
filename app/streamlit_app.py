import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Configuración de rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.markdown("**Modelo XGBoost** - Predicción de precios de casas")

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/best_model_final.pkl")
        preprocessor = joblib.load("models/preprocessor_final.pkl")
        return model, preprocessor
    except:
        st.error("No se pudo cargar el modelo. Verifica que los archivos estén en la carpeta 'models/'")
        return None, None

model, preprocessor = load_model()

if model is None:
    st.stop()

# Formulario simple
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Calidad General (1-10)", 1, 10, 7)
    gr_liv_area = st.number_input("Área Habitable (ft²)", 500, 8000, 1600)
    year_built = st.number_input("Año de Construcción", 1900, 2025, 2000)

with col2:
    garage_cars = st.slider("Plazas de Garage", 0, 4, 2)
    full_bath = st.slider("Baños Completos", 0, 5, 2)
    neighborhood = st.selectbox("Vecindario", ["NAmes", "CollgCr", "OldTown", "Somerst"])

if st.button("🔮 Predecir Precio", type="primary"):
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'YearBuilt': [year_built],
        'GarageCars': [garage_cars],
        'FullBath': [full_bath],
        'Neighborhood': [neighborhood],
        'YrSold': [2025],
        'TotalBsmtSF': [800],
        '1stFlrSF': [1000],
        '2ndFlrSF': [600],
        'OverallCond': [5]
    })
    
    from src.features import feature_engineering
    input_data = feature_engineering(input_data)
    
    input_processed = preprocessor.transform(input_data)
    pred_log = model.predict(input_processed)
    prediction = np.expm1(pred_log[0])
    
    st.success(f"### Precio Estimado: **${prediction:,.2f} USD**")