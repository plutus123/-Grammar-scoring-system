"""
Model building and evaluation for grammar scoring engine.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

def prepare_data_for_training(features_df, labels_df=None):
    """
    Prepare data for model training.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        labels_df (pd.DataFrame, optional): DataFrame with labels
        
    Returns:
        tuple: X (features), y (labels), scaler (for feature normalization)
    """
    # Drop filename column from features
    X = features_df.drop(['filename'], axis=1, errors='ignore')
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if labels_df is not None:
        # Merge with labels if provided
        merged_df = pd.merge(features_df[['filename']], labels_df, on='filename', how='left')
        y = merged_df['label'].values
        return X_scaled, y, scaler
    else:
        return X_scaled, None, scaler

def train_evaluate_models(X, y, cv=5):
    """
    Train and evaluate multiple regression models.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Dictionary with trained models and evaluation results
    """
    # Define models to evaluate
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Evaluation results
    results = {}
    
    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        
        # Cross-validation scores
        mse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []
        
        # K-fold cross-validation
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Compute metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            pearson = pearsonr(y_val, y_pred)[0]
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson)
        
        # Calculate average scores
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)
        avg_r2 = np.mean(r2_scores)
        avg_pearson = np.mean(pearson_scores)
        
        # Store results
        results[name] = {
            'model': model,
            'mse': avg_mse,
            'rmse': np.sqrt(avg_mse),
            'mae': avg_mae,
            'r2': avg_r2,
            'pearson': avg_pearson
        }
        
        print(f"{name} - RMSE: {np.sqrt(avg_mse):.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}, Pearson: {avg_pearson:.4f}")
    
    return results

def tune_best_model(X, y, model_type, param_grid, cv=5):
    """
    Perform hyperparameter tuning for the best model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        model_type (str): Type of model to tune
        param_grid (dict): Grid of parameters to search
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Tuned model
    """
    if model_type == 'Ridge':
        base_model = Ridge()
    elif model_type == 'Lasso':
        base_model = Lasso()
    elif model_type == 'ElasticNet':
        base_model = ElasticNet()
    elif model_type == 'RandomForest':
        base_model = RandomForestRegressor(random_state=42)
    elif model_type == 'GradientBoosting':
        base_model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'SVR':
        base_model = SVR()
    elif model_type == 'XGBoost':
        base_model = xgb.XGBRegressor(random_state=42)
    elif model_type == 'LightGBM':
        base_model = lgb.LGBMRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_

def train_final_model(X, y, best_model_type, best_params):
    """
    Train final model with best parameters on all data.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        best_model_type (str): Type of model to use
        best_params (dict): Best parameters for the model
        
    Returns:
        object: Trained model
    """
    # Make a copy of parameters to avoid modifying the original
    params = best_params.copy()
    
    if best_model_type == 'Ridge':
        model = Ridge(**params)
    elif best_model_type == 'Lasso':
        model = Lasso(**params)
    elif best_model_type == 'ElasticNet':
        model = ElasticNet(**params)
    elif best_model_type == 'RandomForest':
        # Remove random_state if it exists in params to avoid conflict
        if 'random_state' in params:
            del params['random_state']
        model = RandomForestRegressor(random_state=42, **params)
    elif best_model_type == 'GradientBoosting':
        # Remove random_state if it exists in params to avoid conflict
        if 'random_state' in params:
            del params['random_state']
        model = GradientBoostingRegressor(random_state=42, **params)
    elif best_model_type == 'SVR':
        model = SVR(**params)
    elif best_model_type == 'XGBoost':
        # Remove random_state if it exists in params to avoid conflict
        if 'random_state' in params:
            del params['random_state']
        model = xgb.XGBRegressor(random_state=42, **params)
    elif best_model_type == 'LightGBM':
        # Remove random_state if it exists in params to avoid conflict
        if 'random_state' in params:
            del params['random_state']
        model = lgb.LGBMRegressor(random_state=42, **params)
    else:
        raise ValueError(f"Unsupported model type: {best_model_type}")
    
    # Train on all data
    model.fit(X, y)
    
    return model

def save_model(model, scaler, filename='grammar_scoring_model.pkl'):
    """
    Save model and scaler to file.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        filename (str): Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"Model saved to {filename}")

def load_model(filename='grammar_scoring_model.pkl'):
    """
    Load model from file.
    
    Args:
        filename (str): Model filename
        
    Returns:
        tuple: model, scaler
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['scaler']

def get_feature_importance(model, feature_names):
    """
    Get feature importance from model if available.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def evaluate_model_performance(model, X, y):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X (np.ndarray): Test features
        y (np.ndarray): Test labels
        
    Returns:
        dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    pearson = pearsonr(y, y_pred)[0]
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson
    }
    
    return metrics, y_pred
