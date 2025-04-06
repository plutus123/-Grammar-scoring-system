import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import json

from .audio_processor import AudioProcessor
from .grammar_analyzer import GrammarAnalyzer
from .scoring import GrammarScorer as Scorer

class GrammarScorer:
    """
    Main class for the Grammar Scoring Engine.
    Integrates audio processing, grammar analysis, and scoring.
    """
    
    def __init__(self, 
                 whisper_model_size: str = "base", 
                 device: str = None,
                 custom_weights: Dict[str, float] = None):
        """
        Initialize the Grammar Scoring Engine.
        
        Args:
            whisper_model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run models on ('cpu' or 'cuda')
            custom_weights: Custom weights for scoring components
        """
        # Initialize components
        self.audio_processor = AudioProcessor(model_size=whisper_model_size, device=device)
        self.grammar_analyzer = GrammarAnalyzer()
        self.scorer = Scorer()
        
        # Set custom weights if provided
        if custom_weights:
            self.scorer.set_weights(custom_weights)
            
        # Store results
        self.results = {}
        
    def process_audio(self, audio_path: str, engine: str = "whisper") -> Dict[str, Any]:
        """
        Process audio file and transcribe to text.
        
        Args:
            audio_path: Path to the audio file
            engine: ASR engine to use ('whisper' or 'google')
            
        Returns:
            Dictionary containing transcription results
        """
        result = self.audio_processor.transcribe(audio_path, engine=engine)
        
        return result
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze grammar in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing grammar analysis
        """
        analysis = self.grammar_analyzer.analyze_grammar(text)
        
        return analysis
    
    def score_grammar(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score grammar based on analysis.
        
        Args:
            analysis: Grammar analysis results
            
        Returns:
            Dictionary containing scores and feedback
        """
        scores = self.scorer.calculate_combined_score(analysis)
        feedback = self.scorer.generate_feedback(analysis, scores)
        
        return {
            'scores': scores,
            'feedback': feedback,
        }
    
    def score_audio(self, 
                    audio_path: str, 
                    engine: str = "whisper",
                    save_results: bool = True, 
                    output_dir: str = None) -> Dict[str, Any]:
        """
        Complete pipeline: process audio, analyze grammar, and generate scores.
        
        Args:
            audio_path: Path to the audio file
            engine: ASR engine to use ('whisper' or 'google')
            save_results: Whether to save results to file
            output_dir: Directory to save results to
            
        Returns:
            Dictionary containing complete results
        """
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Process audio
        print(f"Transcribing audio file: {audio_path}")
        audio_result = self.process_audio(audio_path, engine=engine)
        transcription = audio_result.get('text', '')
        
        # If no transcription, return error
        if not transcription:
            return {
                'error': 'Failed to transcribe audio',
                'audio_path': audio_path,
            }
        
        # Analyze grammar
        print(f"Analyzing grammar in transcription")
        analysis = self.analyze_text(transcription)
        
        # Score grammar
        print(f"Scoring grammar")
        scoring_result = self.score_grammar(analysis)
        
        # Combine results
        result = {
            'audio_path': audio_path,
            'transcription': transcription,
            'analysis': analysis,
            'scoring': scoring_result,
        }
        
        # Save to instance
        self.results[audio_path] = result
        
        # Save to file if requested
        if save_results and output_dir:
            self.save_results(audio_path, output_dir)
        
        return result
    
    def save_results(self, audio_path: str, output_dir: str) -> str:
        """
        Save results to file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save results to
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get result for audio
        result = self.results.get(audio_path)
        if not result:
            raise ValueError(f"No results found for audio: {audio_path}")
        
        # Create output filename
        audio_basename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_basename)[0]
        output_file = os.path.join(output_dir, f"{audio_name}_grammar_score.json")
        
        # Save to file
        with open(output_file, 'w') as f:
            # Create a simplified version for saving
            save_data = {
                'audio_path': audio_path,
                'transcription': result['transcription'],
                'overall_score': result['scoring']['scores']['overall_score'],
                'subscores': result['scoring']['scores']['subscores'],
                'feedback': result['scoring']['feedback'],
                'error_counts': result['analysis']['grammar_check']['error_counts'],
                'total_errors': result['analysis']['grammar_check']['total_errors'],
                'sentence_stats': {
                    'num_sentences': result['analysis']['sentence_analysis']['num_sentences'],
                    'avg_sentence_length': result['analysis']['sentence_analysis']['avg_sentence_length'],
                    'sentence_types': result['analysis']['sentence_analysis']['sentence_type_counts'],
                },
                'vocabulary_stats': {
                    'total_words': result['analysis']['pos_analysis']['total_words'],
                    'unique_words': result['analysis']['pos_analysis']['unique_words'],
                    'vocabulary_diversity': result['analysis']['pos_analysis']['vocabulary_diversity'],
                },
            }
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return output_file
    
    def generate_report(self, 
                        audio_path: str, 
                        output_dir: str = None,
                        include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive report for the audio.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save report to
            include_visualizations: Whether to include visualizations
            
        Returns:
            Dictionary containing report data and paths to saved files
        """
        # Get result for audio
        result = self.results.get(audio_path)
        if not result:
            raise ValueError(f"No results found for audio: {audio_path}")
        
        # Create report data
        report = {
            'audio_path': audio_path,
            'transcription': result['transcription'],
            'overall_score': result['scoring']['scores']['overall_score'],
            'subscores': result['scoring']['scores']['subscores'],
            'feedback': result['scoring']['feedback'],
        }
        
        # Save visualizations if requested
        if include_visualizations and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create basename for files
            audio_basename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_basename)[0]
            
            # Generate and save visualizations
            viz_paths = self._generate_visualizations(result, audio_name, output_dir)
            report['visualizations'] = viz_paths
        
        return report
    
    def _generate_visualizations(self, 
                                result: Dict[str, Any], 
                                name_prefix: str, 
                                output_dir: str) -> Dict[str, str]:
        """
        Generate visualizations for the result.
        
        Args:
            result: Result data
            name_prefix: Prefix for filenames
            output_dir: Directory to save visualizations to
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        viz_paths = {}
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # 1. Subscores bar chart
        plt.figure(figsize=(10, 6))
        subscores = result['scoring']['scores']['subscores']
        score_df = pd.DataFrame({
            'Category': list(subscores.keys()),
            'Score': list(subscores.values())
        })
        
        ax = sns.barplot(x='Category', y='Score', data=score_df)
        ax.set_title('Grammar Subscores', fontsize=16)
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for i, v in enumerate(score_df['Score']):
            ax.text(i, v + 2, str(v), ha='center')
            
        plt.tight_layout()
        subscores_file = os.path.join(output_dir, f"{name_prefix}_subscores.png")
        plt.savefig(subscores_file)
        plt.close()
        viz_paths['subscores'] = subscores_file
        
        # 2. Error types pie chart
        if result['analysis']['grammar_check']['total_errors'] > 0:
            plt.figure(figsize=(10, 6))
            error_counts = result['analysis']['grammar_check']['error_counts']
            
            # Sort errors by count
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in sorted_errors]
            sizes = [x[1] for x in sorted_errors]
            
            # If too many categories, group small ones
            if len(labels) > 5:
                top_n = 4
                other_sum = sum(sizes[top_n:])
                labels = labels[:top_n] + ['Other Errors']
                sizes = sizes[:top_n] + [other_sum]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Grammar Error Types', fontsize=16)
            
            plt.tight_layout()
            errors_file = os.path.join(output_dir, f"{name_prefix}_error_types.png")
            plt.savefig(errors_file)
            plt.close()
            viz_paths['error_types'] = errors_file
        
        # 3. Sentence types bar chart
        plt.figure(figsize=(10, 6))
        sentence_types = result['analysis']['sentence_analysis']['sentence_type_counts']
        sentence_df = pd.DataFrame({
            'Type': list(sentence_types.keys()),
            'Count': list(sentence_types.values())
        })
        
        ax = sns.barplot(x='Type', y='Count', data=sentence_df)
        ax.set_title('Sentence Types', fontsize=16)
        
        for i, v in enumerate(sentence_df['Count']):
            ax.text(i, v + 0.1, str(v), ha='center')
            
        plt.tight_layout()
        sentence_file = os.path.join(output_dir, f"{name_prefix}_sentence_types.png")
        plt.savefig(sentence_file)
        plt.close()
        viz_paths['sentence_types'] = sentence_file
        
        # 4. POS distribution
        plt.figure(figsize=(12, 6))
        pos_categories = result['analysis']['pos_analysis']['pos_categories']
        pos_df = pd.DataFrame({
            'Category': list(pos_categories.keys()),
            'Count': list(pos_categories.values())
        })
        
        ax = sns.barplot(x='Category', y='Count', data=pos_df)
        ax.set_title('Parts of Speech Distribution', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for i, v in enumerate(pos_df['Count']):
            ax.text(i, v + 0.1, str(v), ha='center')
            
        plt.tight_layout()
        pos_file = os.path.join(output_dir, f"{name_prefix}_pos_distribution.png")
        plt.savefig(pos_file)
        plt.close()
        viz_paths['pos_distribution'] = pos_file
        
        return viz_paths
    
    def batch_process(self, 
                      audio_paths: List[str], 
                      engine: str = "whisper",
                      output_dir: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            engine: ASR engine to use ('whisper' or 'google')
            output_dir: Directory to save results to
            
        Returns:
            Dictionary mapping audio paths to results
        """
        batch_results = {}
        
        for audio_path in audio_paths:
            try:
                print(f"Processing {audio_path}")
                result = self.score_audio(audio_path, engine=engine, save_results=(output_dir is not None), output_dir=output_dir)
                batch_results[audio_path] = result
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                batch_results[audio_path] = {'error': str(e)}
        
        return batch_results
    
    def compare_audios(self, 
                       audio_paths: List[str], 
                       output_dir: str = None) -> Dict[str, Any]:
        """
        Compare multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            output_dir: Directory to save comparison to
            
        Returns:
            Dictionary containing comparison data
        """
        # Ensure all audios have been processed
        for audio_path in audio_paths:
            if audio_path not in self.results:
                self.score_audio(audio_path)
        
        # Extract scores for comparison
        comparison_data = {
            'overall_scores': {},
            'subscores': {},
            'error_counts': {},
            'sentence_stats': {},
            'vocabulary_stats': {},
        }
        
        for audio_path in audio_paths:
            result = self.results[audio_path]
            audio_name = os.path.basename(audio_path)
            
            comparison_data['overall_scores'][audio_name] = result['scoring']['scores']['overall_score']
            comparison_data['subscores'][audio_name] = result['scoring']['scores']['subscores']
            comparison_data['error_counts'][audio_name] = result['analysis']['grammar_check']['total_errors']
            comparison_data['sentence_stats'][audio_name] = {
                'num_sentences': result['analysis']['sentence_analysis']['num_sentences'],
                'avg_sentence_length': result['analysis']['sentence_analysis']['avg_sentence_length'],
            }
            comparison_data['vocabulary_stats'][audio_name] = {
                'vocabulary_diversity': result['analysis']['pos_analysis']['vocabulary_diversity'],
                'total_words': result['analysis']['pos_analysis']['total_words'],
            }
        
        # Generate comparison visualization if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            viz_paths = self._generate_comparison_visualizations(comparison_data, output_dir)
            comparison_data['visualizations'] = viz_paths
        
        return comparison_data
    
    def _generate_comparison_visualizations(self, 
                                           comparison_data: Dict[str, Any], 
                                           output_dir: str) -> Dict[str, str]:
        """
        Generate visualizations for audio comparison.
        
        Args:
            comparison_data: Comparison data
            output_dir: Directory to save visualizations to
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        viz_paths = {}
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # 1. Overall scores comparison
        plt.figure(figsize=(12, 6))
        overall_df = pd.DataFrame({
            'Audio': list(comparison_data['overall_scores'].keys()),
            'Score': list(comparison_data['overall_scores'].values())
        })
        
        ax = sns.barplot(x='Audio', y='Score', data=overall_df)
        ax.set_title('Overall Grammar Score Comparison', fontsize=16)
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for i, v in enumerate(overall_df['Score']):
            ax.text(i, v + 2, str(v), ha='center')
            
        plt.tight_layout()
        overall_file = os.path.join(output_dir, "comparison_overall_scores.png")
        plt.savefig(overall_file)
        plt.close()
        viz_paths['overall_scores'] = overall_file
        
        # 2. Subscores comparison
        # Reshape data for visualization
        subscores_data = []
        for audio, subscores in comparison_data['subscores'].items():
            for category, score in subscores.items():
                subscores_data.append({
                    'Audio': audio,
                    'Category': category,
                    'Score': score
                })
        
        subscores_df = pd.DataFrame(subscores_data)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Category', y='Score', hue='Audio', data=subscores_df)
        ax.set_title('Subscores Comparison', fontsize=16)
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        subscores_file = os.path.join(output_dir, "comparison_subscores.png")
        plt.savefig(subscores_file)
        plt.close()
        viz_paths['subscores'] = subscores_file
        
        # 3. Error counts comparison
        plt.figure(figsize=(12, 6))
        error_df = pd.DataFrame({
            'Audio': list(comparison_data['error_counts'].keys()),
            'Errors': list(comparison_data['error_counts'].values())
        })
        
        ax = sns.barplot(x='Audio', y='Errors', data=error_df)
        ax.set_title('Grammar Error Counts Comparison', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for i, v in enumerate(error_df['Errors']):
            ax.text(i, v + 0.1, str(v), ha='center')
            
        plt.tight_layout()
        errors_file = os.path.join(output_dir, "comparison_error_counts.png")
        plt.savefig(errors_file)
        plt.close()
        viz_paths['error_counts'] = errors_file
        
        # 4. Vocabulary diversity comparison
        plt.figure(figsize=(12, 6))
        vocab_data = []
        for audio, stats in comparison_data['vocabulary_stats'].items():
            vocab_data.append({
                'Audio': audio,
                'Diversity': stats['vocabulary_diversity']
            })
        
        vocab_df = pd.DataFrame(vocab_data)
        
        ax = sns.barplot(x='Audio', y='Diversity', data=vocab_df)
        ax.set_title('Vocabulary Diversity Comparison', fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for i, v in enumerate(vocab_df['Diversity']):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        vocab_file = os.path.join(output_dir, "comparison_vocabulary_diversity.png")
        plt.savefig(vocab_file)
        plt.close()
        viz_paths['vocabulary_diversity'] = vocab_file
        
        return viz_paths
