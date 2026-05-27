import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('..'))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score, mean_absolute_error

from features import feature_engineering
from preprocessing import create_preprocessor

print("🚀 Iniciando entrenamiento con XGBoost Mejorado...")

# 1. Cargar datos
df = pd.read_csv('data/raw/train.csv')
print(f"Datos cargados: {df.shape}")

# 2. Feature Engineering
df = feature_engineering(df)

# 3. Target log
df['SalePrice'] = np.log1p(df['SalePrice'])
y = df['SalePrice']
X = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')

# 4. Preprocesamiento
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = create_preprocessor(numeric_features, categorical_features)
X_processed = preprocessor.fit_transform(X, y)

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 6. XGBoost con buenos hiperparámetros
from xgboost import XGBRegressor

print("Entrenando XGBoost...")
model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmsle'
)

model.fit(X_train, y_train)

# 7. Predicciones y evaluación
y_pred = model.predict(X_test)

y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

rmsle = np.sqrt(mean_squared_log_error(y_test_orig, y_pred_orig))
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test, y_pred)

print("="*65)
print("✅ ENTRENAMIENTO FINALIZADO")
print("="*65)
print(f"RMSLE  : {rmsle:.5f}")
print(f"MAE    : ${mae:,.2f}")
print(f"R²     : {r2:.4f}")
print("="*65)

# 8. Guardar
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/best_model_final.pkl')
joblib.dump(preprocessor, 'models/preprocessor_final.pkl')

print("💾 Modelo guardado correctamente")