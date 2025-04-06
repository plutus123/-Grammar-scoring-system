import nltk
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tag import pos_tag as nltk_pos_tag
from nltk.corpus import wordnet
import string

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')

class GrammarAnalyzer:
    """
    Class for analyzing grammar in text using various NLP techniques.
    """
    
    def __init__(self):
        """
        Initialize the GrammarAnalyzer with required tools.
        """
        # Download additional required NLTK resources
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        # Common grammar error types to track
        self.grammar_error_types = {
            'SPELLING_ERROR': 'Spelling Error',
            'ARTICLE_ERROR': 'Article Error (a/an)',
            'SUBJECT_VERB_AGREEMENT': 'Subject-Verb Agreement Error',
            'DOUBLE_NEGATIVE': 'Double Negative Error',
            'MISSING_COMMA': 'Missing Comma Error',
            'TO_TOO_TWO_CONFUSION': 'To/Too/Two Confusion',
            'DOUBLE_PUNCTUATION': 'Double Punctuation',
            'VERB_FORM_ERROR': 'Incorrect Verb Form',
            'RUN_ON_SENTENCE': 'Run-on Sentence',
            'PASSIVE_VOICE': 'Passive Voice Use',
            'WORDINESS': 'Wordy Phrases',
            'THERE_THEIR_CONFUSION': "There/Their/They're Confusion",
            'ITS_ITS_CONFUSION': "Its/It's Confusion"
        }
        
        # Initialize common word lists
        from nltk.corpus import words as nltk_words
        self.english_words = set(w.lower() for w in nltk_words.words())
        
        # Common contractions and their expanded forms
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "it's": "it is",
            "that's": "that is",
            "they're": "they are",
            "there's": "there is",
            "i'm": "i am",
            "we're": "we are",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "i've": "i have",
            "we've": "we have",
            "you've": "you have",
            "they've": "they have"
        }
    
    def check_spelling(self, word: str) -> bool:
        """
        Check if a word is spelled correctly.
        
        Args:
            word: The word to check
            
        Returns:
            True if the word is spelled correctly, False otherwise
        """
        # Remove punctuation from the word
        clean_word = word.strip(string.punctuation).lower()
        
        # Skip short words, numbers, and contractions
        if len(clean_word) <= 1 or clean_word.isdigit() or "'" in clean_word:
            return True
            
        # Check if the word is in our dictionary
        return clean_word in self.english_words
        
    def check_grammar(self, text: str) -> Dict[str, Any]:
        """
        Check grammar using custom rules and NLTK.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing grammar errors and statistics
        """
        # Tokenize the text
        # Use a simple approach to avoid dependency on punkt_tab
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to a simple regex tokenizer if sent_tokenize fails
            sentence_tokenizer = RegexpTokenizer(r'[.!?]+', gaps=True)
            sentences = [s.strip() for s in sentence_tokenizer.tokenize(text) if s.strip()]
        
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to a simple regex tokenizer if word_tokenize fails
            word_tokenizer = RegexpTokenizer(r'\w+')
            tokens = word_tokenizer.tokenize(text)
            
        words = [word for word in tokens if word.isalpha()]
        # Fix for NLTK tagger issue - using default 'en' tagger
        try:
            pos_tagged = nltk_pos_tag(tokens)
        except LookupError:
            # Fall back to manually running the perceptron tagger with the installed model
            from nltk.tag.perceptron import PerceptronTagger
            tagger = PerceptronTagger()
            pos_tagged = tagger.tag(tokens)
        
        # Process and categorize errors
        errors = []
        
        # Check spelling errors
        for i, word in enumerate(words):
            if not self.check_spelling(word):
                # Create context by getting surrounding words
                start_idx = max(0, i - 3)
                end_idx = min(len(words), i + 4)
                context = " ".join(words[start_idx:end_idx])
                
                error = {
                    'message': f"Possible spelling error: '{word}'",
                    'type': 'SPELLING_ERROR',
                    'category': self.grammar_error_types['SPELLING_ERROR'],
                    'context': context,
                    'offset': text.find(word),
                    'length': len(word),
                    'suggested_replacements': [],
                }
                errors.append(error)
        
        # Check article errors (a vs. an)
        for i in range(len(pos_tagged) - 1):
            if pos_tagged[i][0].lower() == 'a' and pos_tagged[i+1][1].startswith('NN'):
                next_word = pos_tagged[i+1][0].lower()
                if next_word[0] in 'aeiou':
                    start_idx = max(0, i - 2)
                    end_idx = min(len(pos_tagged), i + 3)
                    context = " ".join([w for w, _ in pos_tagged[start_idx:end_idx]])
                    
                    error = {
                        'message': f"Use 'an' before vowel sounds: 'a {next_word}'",
                        'type': 'ARTICLE_ERROR',
                        'category': self.grammar_error_types['ARTICLE_ERROR'],
                        'context': context,
                        'offset': 0,  # Placeholder
                        'length': 0,  # Placeholder
                        'suggested_replacements': [f"an {next_word}"],
                    }
                    errors.append(error)
                    
            elif pos_tagged[i][0].lower() == 'an' and pos_tagged[i+1][1].startswith('NN'):
                next_word = pos_tagged[i+1][0].lower()
                if next_word[0] not in 'aeiou':
                    start_idx = max(0, i - 2)
                    end_idx = min(len(pos_tagged), i + 3)
                    context = " ".join([w for w, _ in pos_tagged[start_idx:end_idx]])
                    
                    error = {
                        'message': f"Use 'a' before consonant sounds: 'an {next_word}'",
                        'type': 'ARTICLE_ERROR',
                        'category': self.grammar_error_types['ARTICLE_ERROR'],
                        'context': context,
                        'offset': 0,  # Placeholder
                        'length': 0,  # Placeholder
                        'suggested_replacements': [f"a {next_word}"],
                    }
                    errors.append(error)
        
        # Check for double negatives
        negative_words = ['no', 'not', 'none', 'nobody', 'nothing', 'nowhere', 'never']
        for sentence in sentences:
            s_lower = sentence.lower()
            neg_count = sum(1 for word in negative_words if f" {word} " in f" {s_lower} ")
            
            if neg_count >= 2:
                error = {
                    'message': "Possible double negative detected",
                    'type': 'DOUBLE_NEGATIVE',
                    'category': self.grammar_error_types['DOUBLE_NEGATIVE'],
                    'context': sentence,
                    'offset': 0,  # Placeholder
                    'length': 0,  # Placeholder
                    'suggested_replacements': [],
                }
                errors.append(error)
                
        # Check for common confusion errors (there/their/they're, to/too/two, its/it's)
        confusion_patterns = [
            (r'\b(their)\b\s+(is|are|were|was)\b', 'THERE_THEIR_CONFUSION', "'their' might be confused with 'there'"),
            (r'\b(there)\b\s+(car|house|book|dog|cat|thing|stuff)\b', 'THERE_THEIR_CONFUSION', "'there' might be confused with 'their'"),
            (r'\b(to)\b\s+(much|many|late|early|good|bad|high|low)\b', 'TO_TOO_CONFUSION', "'to' might be confused with 'too'"),
            (r'\b(its)\b\s+(a|the|going|not|very|really|quite)\b', 'ITS_ITS_CONFUSION', "'its' might be confused with 'it's'"),
        ]
        
        for sentence in sentences:
            for pattern, error_type, message in confusion_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    error = {
                        'message': message,
                        'type': error_type,
                        'category': self.grammar_error_types.get(error_type, 'Other Error'),
                        'context': sentence,
                        'offset': 0,  # Placeholder
                        'length': 0,  # Placeholder
                        'suggested_replacements': [],
                    }
                    errors.append(error)
        
        # Check for run-on sentences (very basic check - sentences longer than 40 words)
        for sentence in sentences:
            if len(word_tokenize(sentence)) > 40:
                error = {
                    'message': "Possible run-on sentence",
                    'type': 'RUN_ON_SENTENCE',
                    'category': self.grammar_error_types['RUN_ON_SENTENCE'],
                    'context': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    'offset': 0,  # Placeholder
                    'length': 0,  # Placeholder
                    'suggested_replacements': [],
                }
                errors.append(error)
        
        # Check for passive voice
        passive_patterns = [
            r'\b(am|is|are|was|were)\s+(being\s+)?([a-z]+ed|done|made|created|written|said)\b',
            r'\b(has|have|had)\s+been\s+([a-z]+ed|done|made|created|written|said)\b',
        ]
        
        for sentence in sentences:
            for pattern in passive_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    error = {
                        'message': "Passive voice detected",
                        'type': 'PASSIVE_VOICE',
                        'category': self.grammar_error_types['PASSIVE_VOICE'],
                        'context': sentence,
                        'offset': 0,  # Placeholder
                        'length': 0,  # Placeholder
                        'suggested_replacements': [],
                    }
                    errors.append(error)
        
        # Count error types
        error_counts = {}
        for error in errors:
            category = error['category']
            error_counts[category] = error_counts.get(category, 0) + 1
            
        # Count total errors
        total_errors = len(errors)
        
        return {
            'errors': errors,
            'error_counts': error_counts,
            'total_errors': total_errors,
            'text_length': len(text),
            'error_density': total_errors / len(text) if len(text) > 0 else 0,
        }
    
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentence structure, complexity, and variety.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentence structure statistics
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        
        # Calculate sentence lengths
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0
        
        # Identify sentence types (simple, compound, complex)
        simple_sentences = 0
        compound_sentences = 0
        complex_sentences = 0
        
        for sentence in sentences:
            # Count coordinating conjunctions (and, but, or, so, for, yet, nor)
            has_coord_conj = bool(re.search(r'\s(and|but|or|so|for|yet|nor)\s', sentence.lower()))
            
            # Count subordinating conjunctions and relative pronouns
            has_subord_conj = bool(re.search(
                r'\s(although|because|since|unless|while|after|before|if|though|whether|which|who|that|whom)\s', 
                sentence.lower()
            ))
            
            if has_coord_conj and not has_subord_conj:
                compound_sentences += 1
            elif has_subord_conj:
                complex_sentences += 1
            else:
                simple_sentences += 1
        
        # Calculate sentence variety
        sentence_type_ratio = {
            'simple': simple_sentences / num_sentences if num_sentences > 0 else 0,
            'compound': compound_sentences / num_sentences if num_sentences > 0 else 0,
            'complex': complex_sentences / num_sentences if num_sentences > 0 else 0,
        }
        
        return {
            'num_sentences': num_sentences,
            'avg_sentence_length': avg_sentence_length,
            'sentence_lengths': sentence_lengths,
            'sentence_type_counts': {
                'simple': simple_sentences,
                'compound': compound_sentences,
                'complex': complex_sentences,
            },
            'sentence_type_ratio': sentence_type_ratio,
        }
    
    def analyze_part_of_speech(self, text: str) -> Dict[str, Any]:
        """
        Analyze part of speech distribution and patterns.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing POS statistics
        """
        # Tokenize and get POS tags
        tokens = word_tokenize(text)
        pos_tags = nltk_pos_tag(tokens)
        
        # Count POS occurrences
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Group POS tags into categories
        pos_categories = {
            'nouns': sum(pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS']),
            'verbs': sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
            'adjectives': sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS']),
            'adverbs': sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS']),
            'pronouns': sum(pos_counts.get(tag, 0) for tag in ['PRP', 'PRP$', 'WP', 'WP$']),
            'determiners': sum(pos_counts.get(tag, 0) for tag in ['DT', 'PDT', 'WDT']),
            'conjunctions': sum(pos_counts.get(tag, 0) for tag in ['CC', 'IN']),
        }
        
        # Calculate diversity of vocabulary (unique words / total words)
        unique_words = len(set(word.lower() for word, _ in pos_tags))
        total_words = len(pos_tags)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        return {
            'pos_counts': dict(pos_counts),
            'pos_categories': pos_categories,
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_diversity': vocabulary_diversity,
        }
    
    def analyze_grammar(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive grammar analysis combining all methods.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing comprehensive grammar analysis
        """
        # Run all analyses
        grammar_check = self.check_grammar(text)
        sentence_analysis = self.analyze_sentence_structure(text)
        pos_analysis = self.analyze_part_of_speech(text)
        
        # Combine results
        return {
            'grammar_check': grammar_check,
            'sentence_analysis': sentence_analysis,
            'pos_analysis': pos_analysis,
        }
