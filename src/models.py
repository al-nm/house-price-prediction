import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_log_error')
    return -score.mean()

def train_best_model(X_train, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    return best_model