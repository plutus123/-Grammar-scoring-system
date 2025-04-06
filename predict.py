"""
Prediction utilities for grammar scoring engine.
"""

import os
import pandas as pd
import numpy as np
from feature_extraction import extract_features_from_directory
from model_building import load_model

def predict_grammar_scores(model, scaler, audio_dir, file_list):
    """
    Predict grammar scores for audio files.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        audio_dir (str): Directory containing audio files
        file_list (list): List of audio filenames
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Extract features
    print("Extracting features from test audio files...")
    features_df = extract_features_from_directory(audio_dir, file_list)
    
    # Prepare features
    X = features_df.drop(['filename'], axis=1)
    X = X.fillna(0)  # Handle any NaN values
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'filename': features_df['filename'],
        'label': predictions
    })
    
    return submission_df

def create_submission_file(predictions_df, output_path='submission.csv'):
    """
    Create submission file.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
        output_path (str): Path to save submission file
    """
    # Ensure predictions are within the valid range [1, 5]
    predictions_df['label'] = predictions_df['label'].clip(1.0, 5.0)
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

def main():
    """Main prediction function."""
    # Load test file list
    test_df = pd.read_csv('dataset/test.csv')
    test_filenames = test_df['filename'].tolist()
    
    # Load trained model
    model, scaler = load_model('grammar_scoring_model.pkl')
    
    # Predict grammar scores
    predictions_df = predict_grammar_scores(
        model, 
        scaler, 
        'dataset/audios_test', 
        test_filenames
    )
    
    # Create submission file
    create_submission_file(predictions_df, 'submission.csv')

if __name__ == '__main__':
    main()
