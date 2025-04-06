#!/usr/bin/env python3
"""
Test script for the Grammar Scoring Engine.
This script processes audio files in the samples folder, analyzes grammar,
and generates scores and visualizations.
"""

import os
import argparse
from grammar_scorer.grammar_scorer import GrammarScorer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Grammar Scoring Engine on audio samples')
    parser.add_argument('--model', type=str, default='base', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size to use')
    parser.add_argument('--engine', type=str, default='whisper',
                        choices=['whisper', 'google'],
                        help='ASR engine to use')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--samples_dir', type=str, default='./samples',
                        help='Directory containing audio samples')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cpu/cuda)')
    args = parser.parse_args()
    
    # Initialize the grammar scorer
    print(f"Initializing Grammar Scorer with {args.model} model...")
    scorer = GrammarScorer(whisper_model_size=args.model, device=args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each audio file in the samples directory
    print(f"Processing audio files in {args.samples_dir}...")
    audio_files = []
    for filename in os.listdir(args.samples_dir):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            audio_files.append(os.path.join(args.samples_dir, filename))
    
    if not audio_files:
        print(f"No audio files found in {args.samples_dir}")
        return
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Process each audio file
    for audio_file in audio_files:
        print(f"\nProcessing: {os.path.basename(audio_file)}")
        try:
            # Score the audio file
            result = scorer.score_audio(
                audio_path=audio_file,
                engine=args.engine,
                save_results=True,
                output_dir=args.output_dir
            )
            
            # Generate and save a report with visualizations
            report = scorer.generate_report(
                audio_path=audio_file,
                output_dir=args.output_dir,
                include_visualizations=True
            )
            
            # Print a summary of the results
            print("\nTranscription:")
            print("-" * 50)
            print(result['transcription'][:200] + "..." if len(result['transcription']) > 200 else result['transcription'])
            print("-" * 50)
            
            print("\nScores:")
            print(f"Overall Score: {result['scoring']['scores']['overall_score']}/100")
            print("\nSubscores:")
            for category, score in result['scoring']['scores']['subscores'].items():
                print(f"  - {category.replace('_', ' ').title()}: {score}/100")
            
            print("\nFeedback Highlights:")
            for feedback in result['scoring']['feedback']:
                print(f"  - {feedback['category']}: {feedback['message']}")
            
            print(f"\nDetailed report and visualizations saved to: {args.output_dir}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()

