import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ==================== CONFIGURACIÓN DE RUTAS ====================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="House Price AI", layout="wide", page_icon="🏠")

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

# ==================== FORMULARIO ====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Principales")
    overall_qual = st.slider("Calidad General", 1, 10, 7)
    gr_liv_area = st.number_input("Área Habitable (ft²)", 500, 8000, 1600)
    total_bsmt_sf = st.number_input("Área del Sótano (ft²)", 0, 5000, 900)
    year_built = st.number_input("Año de Construcción", 1900, 2025, 1995)
    yr_sold = st.number_input("Año de Venta", 2000, 2026, 2025)

with col2:
    st.subheader("Otras Características")
    garage_cars = st.slider("Plazas de Garage", 0, 4, 2)
    full_bath = st.slider("Baños Completos", 0, 5, 2)
    neighborhood = st.selectbox("Vecindario", 
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Gilbert"])
    lot_area = st.number_input("Tamaño del Terreno (ft²)", 1000, 100000, 9500)

# ==================== PREDICCIÓN ====================

if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'TotalBsmtSF': [total_bsmt_sf],
        'YearBuilt': [year_built],
        'YrSold': [yr_sold],
        'GarageCars': [garage_cars],
        'FullBath': [full_bath],
        'Neighborhood': [neighborhood],
        'LotArea': [lot_area],
        '1stFlrSF': [gr_liv_area * 0.6],      # aproximación
        '2ndFlrSF': [gr_liv_area * 0.4],
        'OverallCond': [5],
        'GarageArea': [garage_cars * 200]
    })
    
    from src.features import feature_engineering
    input_data = feature_engineering(input_data)
    
    input_processed = preprocessor.transform(input_data)
    pred_log = model.predict(input_processed)
    prediction = np.expm1(pred_log[0])
    
    st.success(f"### Precio Estimado: **${prediction:,.2f} USD**", icon="💰")
    
