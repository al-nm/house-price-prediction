import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import VotingRegressor

def train_best_model(X_train, y_train):
    # Mejor XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Mejor LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Ensemble
    ensemble = VotingRegressor([
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ])
    
    ensemble.fit(X_train, y_train)
    return ensemble