#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit Application for Grammar Scoring
=========================================

This Streamlit app allows users to upload audio files
and get grammar scoring results with visualizations.
"""

import os
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import sys
import subprocess

# Set page configuration
st.set_page_config(
    page_title="Grammar Scoring App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #212121;
        margin-bottom: 0.5rem;
    }
    .score-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .score-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .score-label {
        font-size: 1rem;
        color: #616161;
    }
    .info-box {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .feedback-item {
        background-color: #f5f5f5;
        border-left: 3px solid #1E88E5;
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return the path"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path

def get_score_color(score):
    """Return color based on score range"""
    if score >= 90:
        return "#4CAF50"  # Green
    elif score >= 70:
        return "#2196F3"  # Blue
    elif score >= 50:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def display_subscore(title, score, col):
    """Display a subscore in a column with styling"""
    color = get_score_color(score)
    col.markdown(f"""
    <div class="score-box">
        <div class="score-label">{title}</div>
        <div class="score-value" style="color: {color};">{score}</div>
        <div class="score-label">/100</div>
    </div>
    """, unsafe_allow_html=True)

def process_audio_file(input_file, model_size="base"):
    """Process an audio file directly using the GrammarScorer class"""
    try:
        # Create a temporary directory to store results
        temp_output_dir = tempfile.mkdtemp()
        
        # Get the filename from the input path
        filename = os.path.basename(input_file)
        
        # Create a temporary directory in samples_dir to place the file
        temp_samples_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_samples_dir, filename)
        
        # Copy the file to the temporary samples directory
        import shutil
        shutil.copy2(input_file, temp_audio_path)
        
        # Run the main.py script with the correct arguments
        result = subprocess.run(
            [sys.executable, "main.py", 
             "--model", model_size,
             "--samples_dir", temp_samples_dir,
             "--output_dir", temp_output_dir],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Find the results file
        result_filename = os.path.splitext(filename)[0] + "_grammar_score.json"
        result_path = os.path.join(temp_output_dir, result_filename)
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                return json.load(f)
        else:
            # Check if there's any other json file in the output directory
            json_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
            if json_files:
                with open(os.path.join(temp_output_dir, json_files[0]), 'r') as f:
                    return json.load(f)
            
        return {"error": "Could not find results file", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error running command: {e.stderr}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def main():
    """Main Streamlit application function"""
    # Header
    st.markdown('<div class="main-header">Grammar Scoring for Speech</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower to process"
        )
        
        st.markdown("## About")
        st.markdown("""
        This app analyzes speech audio files and provides grammar scoring and feedback.
        
        **Features:**
        - Transcribe speech audio using Whisper
        - Analyze grammar and provide detailed feedback
        - Generate visualizations of speech patterns
        """)

    # File uploader
    st.markdown('<div class="sub-header">Upload Audio File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select an audio file (.mp3, .wav, .ogg, .flac)", 
                                    type=["mp3", "wav", "ogg", "flac", "m4a"])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Process button
        if st.button("Analyze Grammar", type="primary", use_container_width=True):
            with st.spinner("Processing audio..."):
                # Save uploaded file
                temp_path = save_uploaded_file(uploaded_file)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Step 1: Transcription
                st.text("Step 1/3: Transcribing audio...")
                progress_bar.progress(20)
                
                # Step 2: Grammar analysis
                st.text("Step 2/3: Analyzing grammar...")
                progress_bar.progress(60)
                
                # Run analysis with the selected model size
                result = process_audio_file(temp_path, model_size=model_size)
                time.sleep(1)
                
                # Step 3: Generating results
                st.text("Step 3/3: Generating results...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                st.empty()
                progress_bar.empty()
                
                # Results section
                st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
                
                if 'error' in result:
                    st.error(f"Error processing audio: {result['error']}")
                else:
                    # Display transcription
                    st.markdown('<div class="sub-header">Transcription</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="info-box">
                        {result.get('transcription', 'No transcription available')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display overall score
                    st.markdown('<div class="sub-header">Overall Grammar Score</div>', unsafe_allow_html=True)
                    overall_score = result.get('overall_score', 0)
                    score_color = get_score_color(overall_score)
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center; margin: 20px 0;">
                        <div style="background-color: {score_color}; width: 150px; height: 150px; 
                                border-radius: 50%; display: flex; align-items: center; 
                                justify-content: center; flex-direction: column; color: white;">
                            <div style="font-size: 3rem; font-weight: bold;">{overall_score}</div>
                            <div style="font-size: 1rem;">/100</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display subscores
                    st.markdown('<div class="sub-header">Subscores</div>', unsafe_allow_html=True)
                    subscores = result.get('subscores', {})
                    
                    cols = st.columns(len(subscores))
                    for i, (title, score) in enumerate(subscores.items()):
                        display_subscore(title.replace('_', ' ').title(), score, cols[i])
                    
                    # Feedback section
                    st.markdown('<div class="sub-header">Feedback</div>', unsafe_allow_html=True)
                    
                    # Display feedback from the feedback array
                    if 'feedback' in result and isinstance(result['feedback'], list):
                        for feedback_item in result['feedback']:
                            category = feedback_item.get('category', '')
                            message = feedback_item.get('message', '')
                            score = feedback_item.get('score', '')
                            
                            st.markdown(f"**{category} ({score}/100):** {message}")
                            
                            if 'suggestions' in feedback_item and feedback_item['suggestions']:
                                st.markdown("**Suggestions:**")
                                for suggestion in feedback_item['suggestions']:
                                    st.markdown(f"- {suggestion}")
                                st.markdown("")
                    
                    # Error counts
                    if 'error_counts' in result and result['error_counts']:
                        st.markdown("**Error Types Breakdown:**")
                        error_data = result['error_counts']
                        
                        # Create a simple bar chart of error counts
                        if sum(error_data.values()) > 0:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            bars = ax.bar(error_data.keys(), error_data.values(), color=sns.color_palette("Reds_d", len(error_data)))
                            ax.set_ylabel('Count')
                            ax.set_title('Grammar Error Types')
                            plt.xticks(rotation=45, ha='right')
                            st.pyplot(fig)
                    
                    # Export options
                    st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
                    
                    # Create download button for JSON results
                    st.download_button(
                        label="Download Full Results (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name=f"grammar_score_{uploaded_file.name}.json",
                        mime="application/json",
                    )

if __name__ == "__main__":
    main()
