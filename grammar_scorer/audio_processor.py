import os
import whisper
import torch
import librosa
import numpy as np
import speech_recognition as sr
from typing import Dict, Any, Optional, Tuple

class AudioProcessor:
    """
    Class for processing audio files and converting speech to text.
    Supports multiple ASR engines including Whisper and Google Speech Recognition.
    """
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize the AudioProcessor with the specified model.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run the model on ('cpu' or 'cuda'). If None, will use CUDA if available.
        """
        self.model_size = model_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize Whisper model
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        # Initialize speech recognizer for Google Speech Recognition
        self.recognizer = sr.Recognizer()
        
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file for ASR.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of audio array and sample rate
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        return y, sr
    
    def transcribe_with_whisper(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper ASR.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription and metadata
        """
        # Preprocess audio
        audio, _ = self.preprocess_audio(audio_path)
        
        # Transcribe using Whisper
        result = self.whisper_model.transcribe(audio)
        
        return result
    
    def transcribe_with_google(self, audio_path: str) -> str:
        """
        Transcribe audio using Google Speech Recognition.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        with sr.AudioFile(audio_path) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Google Speech Recognition could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from Google Speech Recognition service; {e}"
    
    def transcribe(self, audio_path: str, engine: str = "whisper") -> Dict[str, Any]:
        """
        Transcribe audio using the specified engine.
        
        Args:
            audio_path: Path to the audio file
            engine: ASR engine to use ('whisper' or 'google')
            
        Returns:
            Dictionary containing transcription and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if engine.lower() == "whisper":
            return self.transcribe_with_whisper(audio_path)
        elif engine.lower() == "google":
            text = self.transcribe_with_google(audio_path)
            return {"text": text}
        else:
            raise ValueError(f"Unsupported engine: {engine}. Use 'whisper' or 'google'")
