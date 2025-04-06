#!/usr/bin/env python3
"""
Simple test script for the Audio Processor component.
This script processes an audio file and prints the transcription.
"""

import os
import sys
from grammar_scorer.audio_processor import AudioProcessor

def main():
    # Define path to sample audio file
    samples_dir = './samples'
    
    # Find audio files in samples directory
    audio_files = []
    for filename in os.listdir(samples_dir):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            audio_files.append(os.path.join(samples_dir, filename))
    
    if not audio_files:
        print(f"No audio files found in {samples_dir}")
        return
    
    print(f"Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Process the first audio file
    audio_file = audio_files[0]
    print(f"\nProcessing: {os.path.basename(audio_file)}")
    
    try:
        # Initialize AudioProcessor with tiny model to speed up loading
        print("Initializing AudioProcessor with tiny model...")
        processor = AudioProcessor(model_size="tiny")
        
        # Transcribe the audio
        print(f"Transcribing audio file: {audio_file}")
        result = processor.transcribe(audio_file)
        
        print("\nTranscription Result:")
        print("-" * 50)
        print(result.get('text', 'No transcription generated'))
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
