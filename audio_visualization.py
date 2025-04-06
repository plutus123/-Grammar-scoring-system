"""
Visualization utilities for grammar scoring engine.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_waveform_and_spectrogram(file_path, figsize=(14, 5)):
    """
    Plot waveform and spectrogram of an audio file.
    
    Args:
        file_path (str): Path to the audio file
        figsize (tuple): Figure size
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    plt.figure(figsize=figsize)
    
    # Plot waveform
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'Waveform (Duration: {duration:.2f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
    
    return y, sr, duration

def plot_mfcc(file_path, n_mfcc=13, figsize=(12, 4)):
    """
    Plot MFCC features of an audio file.
    
    Args:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients
        figsize (tuple): Figure size
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Plot MFCCs
    plt.figure(figsize=figsize)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title(f'MFCC Features (n_mfcc={n_mfcc})')
    plt.xlabel('Time (frames)')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

def visualize_multiple_audio_examples(audio_dir, file_list, labels_df=None, n=3):
    """
    Visualize multiple audio examples.
    
    Args:
        audio_dir (str): Directory containing audio files
        file_list (list): List of audio filenames
        labels_df (pd.DataFrame): DataFrame with labels
        n (int): Number of examples to visualize
    """
    # Select n random examples
    if n > len(file_list):
        n = len(file_list)
    
    random_indices = np.random.choice(len(file_list), size=n, replace=False)
    selected_files = [file_list[i] for i in random_indices]
    
    for file in selected_files:
        file_path = os.path.join(audio_dir, file)
        
        # Get label if available
        label = None
        if labels_df is not None:
            label_row = labels_df[labels_df['filename'] == file]
            if not label_row.empty:
                label = label_row['label'].values[0]
        
        print(f"File: {file}" + (f", Label: {label}" if label else ""))
        
        # Plot waveform and spectrogram
        plot_waveform_and_spectrogram(file_path)
        
        # Plot MFCC
        plot_mfcc(file_path)
        
        print("-" * 50)

def plot_feature_distributions(features_df, n_features=10, figsize=(15, 10)):
    """
    Plot distributions of top features.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        n_features (int): Number of features to plot
        figsize (tuple): Figure size
    """
    # Select numeric columns only (exclude filename)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Select top n features with highest variance
    variances = features_df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.index[:n_features].tolist()
    
    # Plot distributions
    plt.figure(figsize=figsize)
    for i, feature in enumerate(top_features):
        plt.subplot(int(np.ceil(n_features/3)), 3, i+1)
        sns.histplot(features_df[feature], kde=True)
        plt.title(f'{feature}')
        plt.tight_layout()
    
    plt.suptitle('Distributions of Top Features by Variance', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def visualize_feature_correlations(features_df, labels_df, n_features=20, figsize=(15, 12)):
    """
    Visualize correlations between features and labels.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        labels_df (pd.DataFrame): DataFrame with labels
        n_features (int): Number of top features to show
        figsize (tuple): Figure size
    """
    # Merge features and labels
    merged_df = pd.merge(features_df, labels_df, on='filename', how='inner')
    
    # Calculate correlations with label
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('label')
    
    correlations = {}
    for col in numeric_cols:
        correlations[col] = np.abs(np.corrcoef(merged_df[col], merged_df['label'])[0, 1])
    
    # Sort correlations
    sorted_correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}
    
    # Select top n features
    top_features = list(sorted_correlations.keys())[:n_features]
    
    # Plot correlations
    plt.figure(figsize=figsize)
    
    # Bar plot of top correlations
    plt.subplot(2, 1, 1)
    top_corrs = [sorted_correlations[feat] for feat in top_features]
    sns.barplot(x=top_corrs, y=top_features)
    plt.title(f'Top {n_features} Features Correlated with Grammar Score', fontsize=14)
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    
    # Heatmap of feature intercorrelations
    plt.subplot(2, 1, 2)
    top_feature_df = merged_df[top_features + ['label']]
    correlation_matrix = top_feature_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=False, 
                center=0, square=True, linewidths=.5)
    plt.title('Correlation Matrix of Top Features', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_dimensionality_reduction(features_df, labels_df, method='tsne', figsize=(12, 10)):
    """
    Visualize data using dimensionality reduction.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        labels_df (pd.DataFrame): DataFrame with labels
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        figsize (tuple): Figure size
    """
    # Merge features and labels
    merged_df = pd.merge(features_df, labels_df, on='filename', how='inner')
    
    # Select features
    X = merged_df.drop(['filename', 'label'], axis=1)
    y = merged_df['label']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Audio Features'
    else:
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Audio Features'
    
    # Reduce dimensions
    X_reduced = reducer.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    vis_df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'label': y
    })
    
    # Plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(vis_df['x'], vis_df['y'], c=vis_df['label'], 
                         cmap='viridis', alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(scatter, label='Grammar Score')
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, figsize=(10, 8)):
    """
    Plot predictions vs actual values.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.7, s=100, edgecolors='w')
    
    # Add diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*np.array(y_true) + b, 'g-')
    
    # Add text with metrics
    plt.text(min_val + 0.1, max_val - 0.3, 
             f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nPearson: {pearson_corr:.4f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Predictions vs Actual Values', fontsize=16)
    plt.xlabel('Actual Grammar Score')
    plt.ylabel('Predicted Grammar Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_df, n_features=20, figsize=(12, 8)):
    """
    Plot feature importance.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importance
        n_features (int): Number of features to show
        figsize (tuple): Figure size
    """
    # Select top n features
    top_features = importance_df.head(n_features)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {n_features} Feature Importance', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
