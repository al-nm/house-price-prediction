import pandas as pd

def feature_engineering(df):
    """Feature Engineering Robusto y Completo para Predicción"""
    df = df.copy()
    
    # ====================== VALORES POR DEFECTO ======================
    defaults = {
        'YrSold': 2025,
        'YearBuilt': 2000,
        'YearRemodAdd': 2005,
        'OverallQual': 6,
        'OverallCond': 5,
        'GrLivArea': 1500,
        'TotalBsmtSF': 800,
        '1stFlrSF': 1000,
        '2ndFlrSF': 500,
        'GarageCars': 2,
        'GarageArea': 400,
        'FullBath': 2,
        'HalfBath': 1,
        'BsmtFullBath': 0,
        'BsmtHalfBath': 0,
        'TotRmsAbvGrd': 6,
        'Fireplaces': 1,
        'LotArea': 9000,
        'LotFrontage': 70,
        'GarageYrBlt': 2000,
        'MasVnrArea': 0,
        'BsmtFinSF1': 0,
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 800,
        'LowQualFinSF': 0,
        'WoodDeckSF': 0,
        'OpenPorchSF': 50,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'MiscVal': 0,
        'MoSold': 6,
        'MSSubClass': 60,
        'Neighborhood': 'NAmes',
        'MSZoning': 'RL',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'LotConfig': 'Inside',
        'LandSlope': 'Gtl',
        'BldgType': '1Fam',
        'HouseStyle': '2Story',
        'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'MasVnrType': 'None',
        'ExterQual': 'TA',
        'ExterCond': 'TA',
        'Foundation': 'PConc',
        'BsmtQual': 'TA',
        'BsmtCond': 'TA',
        'BsmtExposure': 'No',
        'BsmtFinType1': 'Unf',
        'BsmtFinType2': 'Unf',
        'Heating': 'GasA',
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'Electrical': 'SBrkr',
        'KitchenQual': 'TA',
        'Functional': 'Typ',
        'FireplaceQu': 'Gd',
        'GarageType': 'Attchd',
        'GarageFinish': 'Unf',
        'GarageQual': 'TA',
        'GarageCond': 'TA',
        'PavedDrive': 'Y',
        'PoolQC': 'None',
        'Fence': 'None',
        'MiscFeature': 'None',
        'SaleType': 'WD',
        'SaleCondition': 'Normal',
        'Alley': 'None',
        'Street': 'Pave',
        'Condition1': 'Norm',
        'Condition2': 'Norm',
        'Utilities': 'AllPub',
        'KitchenAbvGr': 1,
        'BedroomAbvGr': 3,
    }
    
    # Rellenar columnas que falten
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    
    # ====================== CREACIÓN DE FEATURES ======================
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    
    df['IsNew'] = (df['HouseAge'] <= 5).astype(int)
    df['IsRemod'] = (df['RemodAge'] < df['HouseAge']).astype(int)
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['LivingSF'] = df['GrLivArea']
    df['TotalBath'] = (df['FullBath'] + 0.5*df['HalfBath'] + 
                      df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                         df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    
    df['OverallGrade'] = df['OverallQual'] * df['OverallCond']
    df['Qual_per_SF'] = df['OverallQual'] / (df['TotalSF'] + 1)
    df['Age_times_Qual'] = df['HouseAge'] * df['OverallQual']
    df['Qual_GrLiv'] = df['OverallQual'] * df['GrLivArea']
    
    df['SF_per_Room'] = df['TotalSF'] / (df['TotRmsAbvGrd'] + 1)
    df['GaragePerCar'] = df['GarageArea'] / (df['GarageCars'] + 1)
    df['LotArea_per_SF'] = df['LotArea'] / (df['TotalSF'] + 1)
    
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    # Features de calidad (mapping)
    qual_map = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
    df['KitchenQualScore'] = df['KitchenQual'].map(qual_map).fillna(3)
    df['BsmtQualScore'] = df['BsmtQual'].map(qual_map).fillna(0)
    df['ExteriorGrade'] = df['ExterQual'].map(qual_map).fillna(3) * df['ExterCond'].map(qual_map).fillna(3)
    
    return df