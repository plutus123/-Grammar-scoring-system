"""
Main script for the Grammar Scoring Engine.
This script runs the complete pipeline for the grammar scoring engine:
1. Data loading
2. Feature extraction
3. Model training
4. Model evaluation
5. Prediction on test set
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import time

# Import custom modules
import feature_extraction
import model_building
import predict
import audio_visualization
import model_evaluation

# Set random seed for reproducibility
np.random.seed(42)

# Set paths
DATA_PATH = 'dataset/'
TRAIN_AUDIO_PATH = os.path.join(DATA_PATH, 'audios_train')
TEST_AUDIO_PATH = os.path.join(DATA_PATH, 'audios_test')
MODEL_PATH = 'grammar_scoring_model.pkl'
SUBMISSION_PATH = 'submission.csv'
FEATURES_CACHE_PATH = 'features_cache.pkl'

def main():
    """Run the complete pipeline for the grammar scoring engine."""
    print("="*50)
    print("Grammar Scoring Engine for Spoken Data")
    print("="*50)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    print(f"Training data: {train_df.shape[0]} samples")
    print(f"Testing data: {test_df.shape[0]} samples")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    
    # Check if features cache exists
    if os.path.exists(FEATURES_CACHE_PATH):
        print("Loading features from cache...")
        with open(FEATURES_CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        train_features_df = cache['train_features']
        test_features_df = cache['test_features']
    else:
        print("Extracting features from audio files...")
        start_time = time.time()
        
        # Extract training features
        train_filenames = train_df['filename'].tolist()
        train_features_df = feature_extraction.extract_features_from_directory(
            TRAIN_AUDIO_PATH, train_filenames
        )
        
        # Extract testing features
        test_filenames = test_df['filename'].tolist()
        test_features_df = feature_extraction.extract_features_from_directory(
            TEST_AUDIO_PATH, test_filenames
        )
        
        # Cache features
        cache = {
            'train_features': train_features_df,
            'test_features': test_features_df
        }
        with open(FEATURES_CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
        
        end_time = time.time()
        print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    
    print(f"Training features shape: {train_features_df.shape}")
    print(f"Testing features shape: {test_features_df.shape}")
    
    # Step 3: Prepare data for training
    print("\nStep 3: Preparing data for training...")
    X, y, scaler = model_building.prepare_data_for_training(train_features_df, train_df)
    print(f"Training data shape: X: {X.shape}, y: {y.shape}")
    
    # Optional: Visualize features
    print("\nVisualizing feature correlations with grammar score...")
    audio_visualization.visualize_feature_correlations(train_features_df, train_df, n_features=15)
    
    # Step 4: Train and evaluate multiple models
    print("\nStep 4: Training and evaluating models...")
    results = model_building.train_evaluate_models(X, y, cv=5)
    
    # Find best model based on Pearson correlation
    best_model_name = max(results, key=lambda k: results[k]['pearson'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name}")
    print(f"Best model Pearson correlation: {results[best_model_name]['pearson']:.4f}")
    
    # Step 5: Perform detailed evaluation of best model
    print("\nStep 5: Detailed evaluation of best model...")
    evaluation_metrics = model_evaluation.evaluate_model_with_cv(best_model, X, y, cv=5)
    model_evaluation.plot_cv_metrics(evaluation_metrics)
    
    # Optional: Plot learning curve
    print("\nPlotting learning curve...")
    model_evaluation.plot_learning_curve(best_model, X, y, cv=5)
    
    # Step 6: Tune best model
    print("\nStep 6: Tuning best model...")
    if best_model_name == 'Ridge':
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif best_model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'GradientBoosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        }
    elif best_model_name == 'LightGBM':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100]
        }
    else:
        param_grid = {}
    
    if param_grid:
        tuned_model = model_building.tune_best_model(X, y, best_model_name, param_grid)
    else:
        tuned_model = best_model
    
    # Check if the feature importance is available
    if hasattr(tuned_model, 'feature_importances_') or hasattr(tuned_model, 'coef_'):
        feature_names = train_features_df.drop(['filename'], axis=1).columns.tolist()
        importance_df = model_building.get_feature_importance(tuned_model, feature_names)
        if importance_df is not None:
            print("\nTop feature importance:")
            audio_visualization.plot_feature_importance(importance_df)
    
    # Step 7: Train final model on all data
    print("\nStep 7: Training final model on all data...")
    final_model = model_building.train_final_model(X, y, best_model_name, 
                                                tuned_model.get_params())
    
    # Step 8: Save model
    print("\nStep 8: Saving model...")
    model_building.save_model(final_model, scaler, MODEL_PATH)
    
    # Step 9: Predict on test set
    print("\nStep 9: Predicting on test set...")
    test_predictions_df = predict.predict_grammar_scores(
        final_model, scaler, TEST_AUDIO_PATH, test_df['filename'].tolist()
    )
    
    # Step 10: Create submission file
    print("\nStep 10: Creating submission file...")
    predict.create_submission_file(test_predictions_df, SUBMISSION_PATH)
    
    print("\nGrammar Scoring Engine pipeline completed!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Submission file saved to: {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
