# Grammar Scoring Engine for Voice Samples

This project provides a comprehensive system for evaluating and scoring grammar in spoken English. It processes audio samples, transcribes them to text, analyzes grammar patterns, and provides detailed scoring and feedback.

## Features

- Speech-to-text conversion using Whisper ASR
- Comprehensive grammar analysis using various NLP techniques
- Multiple scoring metrics for different aspects of grammar
- Visualization of results
- Sample Jupyter notebook for demonstration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from grammar_scorer import GrammarScorer

# Initialize the scorer
scorer = GrammarScorer()

# Score an audio file
result = scorer.score_audio("path/to/audio.wav")
print(result)
```

### Running the Demo

```bash
python main.py --audio samples/audio1.wav
```

### Using the Jupyter Notebook

Open `demo_notebook.ipynb` to see a comprehensive demonstration.

## Project Structure

- `grammar_scorer/`: Main package
  - `audio_processor.py`: Speech-to-text conversion
  - `grammar_analyzer.py`: Grammar analysis tools
  - `scoring.py`: Scoring algorithms
  - `utils.py`: Utility functions
- `samples/`: Sample audio files
- `models/`: Pre-trained models
- `notebooks/`: Jupyter notebooks for demonstration
- `tests/`: Unit tests

## Kaggle Integration

This project is designed to work seamlessly on Kaggle. The included notebook can be uploaded directly to Kaggle.

## License

MIT
