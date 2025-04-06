"""
Feature extraction module for grammar scoring engine.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(file_path, n_mfcc=13, n_mels=40, n_chroma=12):
    """
    Extract audio features from a single audio file.
    
    Args:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        n_mels (int): Number of Mel bands to consider
        n_chroma (int): Number of chroma features to extract
        
    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # Mel spectrograms
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_means = np.mean(mel_spec, axis=1)
        mel_spec_vars = np.var(mel_spec, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_means = np.mean(chroma, axis=1)
        chroma_vars = np.var(chroma, axis=1)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_var = np.var(spec_cent)
        
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean = np.mean(spec_bw)
        spec_bw_var = np.var(spec_bw)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Tempo and rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Harmonicity and percussiveness
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_mean = np.mean(y_harmonic)
        harmonic_var = np.var(y_harmonic)
        percussive_mean = np.mean(y_percussive)
        percussive_var = np.var(y_percussive)
        
        # Speech rate estimation (using zero crossings as proxy)
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = np.sum(librosa.zero_crossings(y)) / duration
        
        # Collect all features
        features = {
            # Basic info
            'duration': duration,
            'tempo': tempo,
            'speech_rate': speech_rate,
            
            # MFCC statistics
            **{f'mfcc_mean_{i}': val for i, val in enumerate(mfcc_means)},
            **{f'mfcc_var_{i}': val for i, val in enumerate(mfcc_vars)},
            
            # Mel spectrogram statistics
            **{f'melspec_mean_{i}': val for i, val in enumerate(mel_spec_means)},
            **{f'melspec_var_{i}': val for i, val in enumerate(mel_spec_vars)},
            
            # Chroma statistics
            **{f'chroma_mean_{i}': val for i, val in enumerate(chroma_means)},
            **{f'chroma_var_{i}': val for i, val in enumerate(chroma_vars)},
            
            # Spectral statistics
            'spectral_centroid_mean': spec_cent_mean,
            'spectral_centroid_var': spec_cent_var,
            'spectral_bandwidth_mean': spec_bw_mean,
            'spectral_bandwidth_var': spec_bw_var,
            'rolloff_mean': rolloff_mean,
            'rolloff_var': rolloff_var,
            
            # Other features
            'zero_crossing_rate_mean': zcr_mean,
            'zero_crossing_rate_var': zcr_var,
            'rms_energy_mean': rms_mean,
            'rms_energy_var': rms_var,
            'harmonic_mean': harmonic_mean,
            'harmonic_var': harmonic_var,
            'percussive_mean': percussive_mean,
            'percussive_var': percussive_var
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def extract_features_from_directory(audio_dir, file_list):
    """
    Extract features from all audio files in the directory.
    
    Args:
        audio_dir (str): Directory containing audio files
        file_list (list): List of audio filenames to process
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features_list = []
    
    for filename in tqdm(file_list, desc="Extracting audio features"):
        file_path = os.path.join(audio_dir, filename)
        if os.path.exists(file_path):
            features = extract_audio_features(file_path)
            if features:
                features['filename'] = filename
                features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    return features_df

def extract_speech_based_features(audio_file, use_whisper=True):
    """
    Extract speech-based features using speech-to-text.
    This is a placeholder for actual speech-to-text implementation.
    
    In a real implementation, you would use a speech recognition API like:
    - Google Cloud Speech-to-Text
    - Whisper from OpenAI
    - Mozilla DeepSpeech
    - Other speech recognition libraries
    
    Args:
        audio_file (str): Path to audio file
        use_whisper (bool): Whether to use Whisper model (placeholder)
        
    Returns:
        dict: Dictionary with text features
    """
    # Placeholder for speech recognition
    # In a real implementation, you would add code to transcribe the audio
    # and extract linguistic features from the transcription
    
    # Example features that could be extracted from transcription:
    # - Word count
    # - Sentence count
    # - Average words per sentence
    # - Grammatical complexity measures (parsing)
    # - Language model perplexity 
    # - Language errors detected
    # - Vocabulary richness
    
    # For now, return placeholder features
    return {
        'placeholder_speech_feature': 0.0
    }

def combine_feature_sets(audio_features_df, text_features_df=None):
    """
    Combine audio and text features.
    
    Args:
        audio_features_df (pd.DataFrame): DataFrame with audio features
        text_features_df (pd.DataFrame): DataFrame with text features
        
    Returns:
        pd.DataFrame: Combined features
    """
    if text_features_df is not None:
        # Merge audio and text features
        combined_df = pd.merge(audio_features_df, text_features_df, on='filename', how='left')
        return combined_df
    else:
        # Return just audio features
        return audio_features_df
