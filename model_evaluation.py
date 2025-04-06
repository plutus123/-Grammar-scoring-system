"""
Model evaluation utilities for grammar scoring engine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def evaluate_model_with_cv(model, X, y, cv=5):
    """
    Evaluate model using cross-validation.
    
    Args:
        model: ML model to evaluate
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Performance metrics
    """
    # Performance metrics
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_scores = []
    
    # K-fold cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Store actual vs. predicted values for all folds
    all_y_true = []
    all_y_pred = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Store actual and predicted values
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        # Compute metrics
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        pearson = pearsonr(y_val, y_pred)[0]
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_scores.append(pearson)
    
    # Calculate average scores
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_pearson = np.mean(pearson_scores)
    
    metrics = {
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'r2_scores': r2_scores,
        'pearson_scores': pearson_scores,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'avg_pearson': avg_pearson,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred
    }
    
    return metrics

def plot_cv_metrics(metrics_dict, figsize=(12, 8)):
    """
    Plot cross-validation metrics.
    
    Args:
        metrics_dict (dict): Dictionary with evaluation metrics
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot RMSE and MAE by fold
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(metrics_dict['rmse_scores']) + 1), metrics_dict['rmse_scores'], 'o-', label='RMSE')
    plt.plot(range(1, len(metrics_dict['mae_scores']) + 1), metrics_dict['mae_scores'], 'o-', label='MAE')
    plt.axhline(y=metrics_dict['avg_rmse'], color='r', linestyle='--', alpha=0.7, label=f'Avg RMSE: {metrics_dict["avg_rmse"]:.4f}')
    plt.axhline(y=metrics_dict['avg_mae'], color='g', linestyle='--', alpha=0.7, label=f'Avg MAE: {metrics_dict["avg_mae"]:.4f}')
    plt.title('Error Metrics by CV Fold')
    plt.xlabel('CV Fold')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot R² and Pearson by fold
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(metrics_dict['r2_scores']) + 1), metrics_dict['r2_scores'], 'o-', label='R²')
    plt.plot(range(1, len(metrics_dict['pearson_scores']) + 1), metrics_dict['pearson_scores'], 'o-', label='Pearson')
    plt.axhline(y=metrics_dict['avg_r2'], color='r', linestyle='--', alpha=0.7, label=f'Avg R²: {metrics_dict["avg_r2"]:.4f}')
    plt.axhline(y=metrics_dict['avg_pearson'], color='g', linestyle='--', alpha=0.7, label=f'Avg Pearson: {metrics_dict["avg_pearson"]:.4f}')
    plt.title('Correlation Metrics by CV Fold')
    plt.xlabel('CV Fold')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot predicted vs. actual from CV
    plt.subplot(2, 1, 2)
    sns.scatterplot(x=metrics_dict['all_y_true'], y=metrics_dict['all_y_pred'], alpha=0.6)
    
    # Add diagonal line
    min_val = min(min(metrics_dict['all_y_true']), min(metrics_dict['all_y_pred']))
    max_val = max(max(metrics_dict['all_y_true']), max(metrics_dict['all_y_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression line
    m, b = np.polyfit(metrics_dict['all_y_true'], metrics_dict['all_y_pred'], 1)
    x_range = np.array([min_val, max_val])
    plt.plot(x_range, m*x_range + b, 'g-')
    
    plt.title('Predicted vs. Actual Values (All CV Folds)')
    plt.xlabel('Actual Grammar Score')
    plt.ylabel('Predicted Grammar Score')
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    pearson_all, _ = pearsonr(metrics_dict['all_y_true'], metrics_dict['all_y_pred'])
    r2_all = r2_score(metrics_dict['all_y_true'], metrics_dict['all_y_pred'])
    rmse_all = np.sqrt(mean_squared_error(metrics_dict['all_y_true'], metrics_dict['all_y_pred']))
    
    plt.text(min_val + 0.1, max_val - 0.3, 
             f'RMSE: {rmse_all:.4f}\nR²: {r2_all:.4f}\nPearson: {pearson_all:.4f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error', figsize=(12, 6)):
    """
    Plot learning curve for model.
    
    Args:
        estimator: ML model
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cv (int): Number of cross-validation folds
        train_sizes (np.ndarray): Training set sizes to plot
        scoring (str): Scoring metric
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes,
        scoring=scoring, n_jobs=-1, shuffle=True, random_state=42
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)
    
    # Convert MSE to RMSE
    train_mean = np.sqrt(train_mean)
    train_std = np.sqrt(train_std)
    test_mean = np.sqrt(test_mean)
    test_std = np.sqrt(test_std)
    
    # Plot learning curve
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training RMSE')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation RMSE')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.title('Learning Curve', fontsize=16)
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def score_breakdown_by_label_range(y_true, y_pred):
    """
    Analyze model performance by label range.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        pd.DataFrame: Performance metrics by label range
    """
    # Create bins for grammar scores
    bins = [1.0, 2.0, 3.0, 4.0, 5.0]
    bin_labels = ['1-2', '2-3', '3-4', '4-5']
    
    # Get bin index for each true value
    bin_indices = np.digitize(y_true, bins, right=True)
    
    # Calculate metrics for each bin
    results = []
    
    for i, label in enumerate(bin_labels):
        # Filter values in this bin
        mask = (bin_indices == i)
        if np.sum(mask) == 0:
            continue
            
        bin_y_true = y_true[mask]
        bin_y_pred = y_pred[mask]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(bin_y_true, bin_y_pred))
        mae = mean_absolute_error(bin_y_true, bin_y_pred)
        pearson = pearsonr(bin_y_true, bin_y_pred)[0] if len(bin_y_true) > 1 else np.nan
        
        # Store results
        results.append({
            'Label Range': label,
            'Count': np.sum(mask),
            'RMSE': rmse,
            'MAE': mae,
            'Pearson': pearson
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def analyze_errors(y_true, y_pred, threshold=0.5):
    """
    Analyze prediction errors.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        threshold (float): Error threshold to consider significant
        
    Returns:
        tuple: DataFrames with error analysis
    """
    # Calculate absolute errors
    errors = np.abs(y_pred - y_true)
    
    # Create DataFrame with all predictions
    all_preds_df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred,
        'Error': errors
    })
    
    # Get high-error predictions
    high_error_df = all_preds_df[all_preds_df['Error'] > threshold].sort_values('Error', ascending=False)
    
    # Get error statistics by true value
    error_by_true = all_preds_df.groupby(np.round(all_preds_df['True'] * 2) / 2).agg({
        'Error': ['mean', 'std', 'count']
    }).reset_index()
    
    error_by_true.columns = ['True Score', 'Mean Error', 'Std Error', 'Count']
    
    return all_preds_df, high_error_df, error_by_true

def plot_error_analysis(all_preds_df, error_by_true, figsize=(15, 10)):
    """
    Plot error analysis.
    
    Args:
        all_preds_df (pd.DataFrame): DataFrame with all predictions
        error_by_true (pd.DataFrame): Error statistics by true value
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot error distribution
    plt.subplot(2, 2, 1)
    sns.histplot(all_preds_df['Error'], bins=20, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Plot error vs. true value
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='True', y='Error', data=all_preds_df, alpha=0.6)
    plt.title('Error vs. True Grammar Score')
    plt.xlabel('True Grammar Score')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Plot mean error by true score
    plt.subplot(2, 2, 3)
    sns.barplot(x='True Score', y='Mean Error', data=error_by_true)
    plt.title('Mean Error by True Grammar Score')
    plt.xlabel('True Grammar Score')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Plot number of samples by score
    plt.subplot(2, 2, 4)
    sns.barplot(x='True Score', y='Count', data=error_by_true)
    plt.title('Number of Samples by Score')
    plt.xlabel('True Grammar Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
