from typing import Dict, Any, List
import numpy as np
import pandas as pd

class GrammarScorer:
    """
    Class for scoring grammar based on analysis results.
    Converts grammar analysis into numerical scores and feedback.
    """
    
    def __init__(self):
        """
        Initialize the GrammarScorer with default scoring weights.
        """
        # Default weights for different aspects of grammar scoring
        self.weights = {
            'error_density': 0.35,           # Weight for error density (errors per character)
            'sentence_complexity': 0.20,      # Weight for sentence complexity
            'vocabulary_diversity': 0.20,     # Weight for vocabulary diversity
            'pos_distribution': 0.15,         # Weight for part of speech distribution
            'error_variety': 0.10,            # Weight for variety of errors
        }
        
        # Benchmark values for scoring calibration
        self.benchmarks = {
            'error_density': {
                'excellent': 0.005,  # Less than 0.5% error density
                'good': 0.015,       # Less than 1.5% error density
                'average': 0.03,     # Less than 3% error density
            },
            'sentence_complexity': {
                'simple_ratio': 0.5,     # Ideal ratio of simple sentences
                'compound_ratio': 0.3,   # Ideal ratio of compound sentences
                'complex_ratio': 0.2,    # Ideal ratio of complex sentences
            },
            'vocabulary_diversity': {
                'excellent': 0.7,    # 70% unique words
                'good': 0.5,         # 50% unique words
                'average': 0.3,      # 30% unique words
            },
            'pos_distribution': {
                'noun_verb_ratio': 1.5,  # Ideal ratio of nouns to verbs
                'adj_adv_ratio': 2.0,    # Ideal ratio of adjectives to adverbs
            },
        }
        
    def set_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update scoring weights.
        
        Args:
            new_weights: Dictionary containing new weights
        """
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value
                
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def score_error_density(self, grammar_check: Dict[str, Any]) -> float:
        """
        Score based on error density.
        
        Args:
            grammar_check: Grammar check results
            
        Returns:
            Score from 0 to 100
        """
        error_density = grammar_check['error_density']
        
        # Calculate score (lower error density is better)
        if error_density <= self.benchmarks['error_density']['excellent']:
            score = 100
        elif error_density <= self.benchmarks['error_density']['good']:
            # Linear interpolation between excellent (100) and good (80)
            range_size = self.benchmarks['error_density']['good'] - self.benchmarks['error_density']['excellent']
            position = (error_density - self.benchmarks['error_density']['excellent']) / range_size
            score = 100 - (position * 20)
        elif error_density <= self.benchmarks['error_density']['average']:
            # Linear interpolation between good (80) and average (60)
            range_size = self.benchmarks['error_density']['average'] - self.benchmarks['error_density']['good']
            position = (error_density - self.benchmarks['error_density']['good']) / range_size
            score = 80 - (position * 20)
        else:
            # Linear decrease below average, capped at 0
            score = max(0, 60 - ((error_density - self.benchmarks['error_density']['average']) * 1000))
            
        return score
    
    def score_sentence_complexity(self, sentence_analysis: Dict[str, Any]) -> float:
        """
        Score based on sentence complexity and variety.
        
        Args:
            sentence_analysis: Sentence analysis results
            
        Returns:
            Score from 0 to 100
        """
        # Extract sentence type ratios
        type_ratio = sentence_analysis['sentence_type_ratio']
        simple_ratio = type_ratio['simple']
        compound_ratio = type_ratio['compound']
        complex_ratio = type_ratio['complex']
        
        # Calculate deviation from ideal ratios
        simple_dev = abs(simple_ratio - self.benchmarks['sentence_complexity']['simple_ratio'])
        compound_dev = abs(compound_ratio - self.benchmarks['sentence_complexity']['compound_ratio'])
        complex_dev = abs(complex_ratio - self.benchmarks['sentence_complexity']['complex_ratio'])
        
        # Calculate average deviation (lower is better)
        avg_deviation = (simple_dev + compound_dev + complex_dev) / 3
        
        # Convert deviation to score (0 deviation = 100 score, 1 deviation = 0 score)
        score = max(0, 100 - (avg_deviation * 100))
        
        # Bonus for having all types of sentences
        if simple_ratio > 0 and compound_ratio > 0 and complex_ratio > 0:
            score = min(100, score + 10)
            
        # Penalty for very short or very long average sentence length
        avg_length = sentence_analysis['avg_sentence_length']
        if avg_length < 5 or avg_length > 30:
            score = max(0, score - 10)
            
        return score
    
    def score_vocabulary_diversity(self, pos_analysis: Dict[str, Any]) -> float:
        """
        Score based on vocabulary diversity.
        
        Args:
            pos_analysis: Part of speech analysis results
            
        Returns:
            Score from 0 to 100
        """
        diversity = pos_analysis['vocabulary_diversity']
        
        # Calculate score based on diversity
        if diversity >= self.benchmarks['vocabulary_diversity']['excellent']:
            score = 100
        elif diversity >= self.benchmarks['vocabulary_diversity']['good']:
            # Linear interpolation between excellent (100) and good (80)
            range_size = self.benchmarks['vocabulary_diversity']['excellent'] - self.benchmarks['vocabulary_diversity']['good']
            position = (self.benchmarks['vocabulary_diversity']['excellent'] - diversity) / range_size
            score = 100 - (position * 20)
        elif diversity >= self.benchmarks['vocabulary_diversity']['average']:
            # Linear interpolation between good (80) and average (60)
            range_size = self.benchmarks['vocabulary_diversity']['good'] - self.benchmarks['vocabulary_diversity']['average']
            position = (self.benchmarks['vocabulary_diversity']['good'] - diversity) / range_size
            score = 80 - (position * 20)
        else:
            # Linear decrease below average, capped at 0
            score = max(0, 60 - ((self.benchmarks['vocabulary_diversity']['average'] - diversity) * 200))
            
        return score
    
    def score_pos_distribution(self, pos_analysis: Dict[str, Any]) -> float:
        """
        Score based on part of speech distribution.
        
        Args:
            pos_analysis: Part of speech analysis results
            
        Returns:
            Score from 0 to 100
        """
        pos_categories = pos_analysis['pos_categories']
        
        # Calculate noun-verb and adjective-adverb ratios
        noun_verb_ratio = pos_categories['nouns'] / pos_categories['verbs'] if pos_categories['verbs'] > 0 else float('inf')
        adj_adv_ratio = pos_categories['adjectives'] / pos_categories['adverbs'] if pos_categories['adverbs'] > 0 else float('inf')
        
        # Calculate deviation from ideal ratios
        noun_verb_dev = abs(noun_verb_ratio - self.benchmarks['pos_distribution']['noun_verb_ratio']) / self.benchmarks['pos_distribution']['noun_verb_ratio']
        adj_adv_dev = abs(adj_adv_ratio - self.benchmarks['pos_distribution']['adj_adv_ratio']) / self.benchmarks['pos_distribution']['adj_adv_ratio']
        
        # Cap very large deviations
        noun_verb_dev = min(noun_verb_dev, 1.0)
        adj_adv_dev = min(adj_adv_dev, 1.0)
        
        # Calculate average deviation (lower is better)
        avg_deviation = (noun_verb_dev + adj_adv_dev) / 2
        
        # Convert deviation to score (0 deviation = 100 score, 1 deviation = 0 score)
        score = max(0, 100 - (avg_deviation * 100))
        
        return score
    
    def score_error_variety(self, grammar_check: Dict[str, Any]) -> float:
        """
        Score based on variety of errors (fewer types of errors is better).
        
        Args:
            grammar_check: Grammar check results
            
        Returns:
            Score from 0 to 100
        """
        error_counts = grammar_check['error_counts']
        num_error_types = len(error_counts)
        total_errors = grammar_check['total_errors']
        
        # Calculate score (fewer error types is better)
        if num_error_types == 0:
            score = 100
        elif num_error_types <= 2:
            score = 90
        elif num_error_types <= 5:
            score = 80 - ((num_error_types - 2) * 5)
        else:
            score = max(0, 65 - ((num_error_types - 5) * 5))
            
        # Apply penalty for repeated errors of the same type
        if total_errors > 0:
            max_single_error = max(error_counts.values()) if error_counts else 0
            repeat_ratio = max_single_error / total_errors
            
            if repeat_ratio > 0.5:  # More than half of errors are of the same type
                score = max(0, score - 10)
                
        return score
    
    def calculate_combined_score(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate combined grammar score based on analysis results.
        
        Args:
            analysis_results: Complete grammar analysis results
            
        Returns:
            Dictionary containing overall score and subscores
        """
        grammar_check = analysis_results['grammar_check']
        sentence_analysis = analysis_results['sentence_analysis']
        pos_analysis = analysis_results['pos_analysis']
        
        # Calculate subscores
        error_density_score = self.score_error_density(grammar_check)
        sentence_complexity_score = self.score_sentence_complexity(sentence_analysis)
        vocabulary_diversity_score = self.score_vocabulary_diversity(pos_analysis)
        pos_distribution_score = self.score_pos_distribution(pos_analysis)
        error_variety_score = self.score_error_variety(grammar_check)
        
        # Combine subscores using weights
        combined_score = (
            self.weights['error_density'] * error_density_score +
            self.weights['sentence_complexity'] * sentence_complexity_score +
            self.weights['vocabulary_diversity'] * vocabulary_diversity_score +
            self.weights['pos_distribution'] * pos_distribution_score +
            self.weights['error_variety'] * error_variety_score
        )
        
        # Round to nearest integer
        combined_score = round(combined_score)
        
        return {
            'overall_score': combined_score,
            'subscores': {
                'error_density': round(error_density_score),
                'sentence_complexity': round(sentence_complexity_score),
                'vocabulary_diversity': round(vocabulary_diversity_score),
                'pos_distribution': round(pos_distribution_score),
                'error_variety': round(error_variety_score),
            },
            'weights': self.weights.copy(),
        }
    
    def generate_feedback(self, analysis_results: Dict[str, Any], scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate detailed feedback based on analysis results and scores.
        
        Args:
            analysis_results: Complete grammar analysis results
            scores: Score results
            
        Returns:
            List of feedback items with categories and suggestions
        """
        feedback = []
        
        # Get components
        grammar_check = analysis_results['grammar_check']
        sentence_analysis = analysis_results['sentence_analysis']
        pos_analysis = analysis_results['pos_analysis']
        subscores = scores['subscores']
        
        # Error feedback
        if grammar_check['total_errors'] > 0:
            feedback.append({
                'category': 'Grammar Errors',
                'score': subscores['error_density'],
                'message': f"Found {grammar_check['total_errors']} grammar errors in the text.",
                'suggestions': [
                    f"Fix {error['category']}: '{error['context']}'" 
                    for error in grammar_check['errors'][:3]  # Show first 3 errors
                ],
            })
            
        # Sentence structure feedback
        if subscores['sentence_complexity'] < 70:
            sentence_types = sentence_analysis['sentence_type_counts']
            type_ratio = sentence_analysis['sentence_type_ratio']
            
            feedback.append({
                'category': 'Sentence Structure',
                'score': subscores['sentence_complexity'],
                'message': "Your sentence structure could be more varied.",
                'suggestions': [],
            })
            
            if type_ratio['simple'] > 0.7:
                feedback[-1]['suggestions'].append(
                    "Try using more compound and complex sentences to add variety."
                )
            elif type_ratio['complex'] > 0.7:
                feedback[-1]['suggestions'].append(
                    "Try using more simple sentences to balance complexity."
                )
                
            if sentence_analysis['avg_sentence_length'] < 5:
                feedback[-1]['suggestions'].append(
                    "Your sentences are very short. Try combining some related ideas."
                )
            elif sentence_analysis['avg_sentence_length'] > 25:
                feedback[-1]['suggestions'].append(
                    "Your sentences are quite long. Try breaking some into shorter sentences."
                )
                
        # Vocabulary feedback
        if subscores['vocabulary_diversity'] < 70:
            feedback.append({
                'category': 'Vocabulary',
                'score': subscores['vocabulary_diversity'],
                'message': "Your vocabulary diversity could be improved.",
                'suggestions': [
                    "Try using a wider range of words instead of repeating the same ones.",
                    "Consider using more specific words rather than general ones.",
                ],
            })
            
        # Part of speech feedback
        if subscores['pos_distribution'] < 70:
            pos_categories = pos_analysis['pos_categories']
            
            feedback.append({
                'category': 'Word Usage',
                'score': subscores['pos_distribution'],
                'message': "The distribution of parts of speech could be improved.",
                'suggestions': [],
            })
            
            noun_verb_ratio = pos_categories['nouns'] / pos_categories['verbs'] if pos_categories['verbs'] > 0 else float('inf')
            if noun_verb_ratio > 2.5:
                feedback[-1]['suggestions'].append(
                    "Try using more action verbs rather than relying heavily on nouns."
                )
            elif noun_verb_ratio < 1.0:
                feedback[-1]['suggestions'].append(
                    "Try using more descriptive nouns to balance your frequent use of verbs."
                )
                
            if pos_categories['adjectives'] < 0.05 * pos_analysis['total_words']:
                feedback[-1]['suggestions'].append(
                    "Consider using more adjectives to add descriptive detail."
                )
                
        # Overall feedback
        overall_score = scores['overall_score']
        if overall_score >= 90:
            feedback.append({
                'category': 'Overall Assessment',
                'score': overall_score,
                'message': "Excellent grammar! Your writing shows strong command of language rules.",
                'suggestions': ["Keep up the good work!"],
            })
        elif overall_score >= 80:
            feedback.append({
                'category': 'Overall Assessment',
                'score': overall_score,
                'message': "Good grammar overall with minor issues.",
                'suggestions': ["Focus on the specific areas mentioned above to improve further."],
            })
        elif overall_score >= 70:
            feedback.append({
                'category': 'Overall Assessment',
                'score': overall_score,
                'message': "Satisfactory grammar with some notable issues.",
                'suggestions': ["Regular practice and focusing on problem areas will help improve your grammar."],
            })
        elif overall_score >= 60:
            feedback.append({
                'category': 'Overall Assessment',
                'score': overall_score,
                'message': "Basic grammar with several areas needing improvement.",
                'suggestions': ["Consider reviewing basic grammar rules and practicing regularly."],
            })
        else:
            feedback.append({
                'category': 'Overall Assessment',
                'score': overall_score,
                'message': "Significant grammar issues affecting comprehension.",
                'suggestions': [
                    "Focus on fundamentals of grammar and sentence structure.",
                    "Consider using grammar learning resources or working with a tutor."
                ],
            })
            
        return feedback
