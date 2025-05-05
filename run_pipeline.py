#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the LDA NHANES pipeline')
    parser.add_argument('--steps', type=str, default='all',
                        help='Comma-separated list of steps to run (download,narratives,concepts,lda,mapping,visualize)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip steps where output files already exist')
    parser.add_argument('--num_topics', type=int, default=10,
                        help='Number of topics for the LDA model (if not running model selection)')
    return parser.parse_args()

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return False

def check_output_exists(file_path):
    return os.path.exists(file_path)

def main():
    args = parse_arguments()
    
    # Set up base directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Add the project root to the Python path
    sys.path.append(BASE_DIR)
    
    # Import config after setting up path
    from config import PROCESSED_DATA_DIR, CORPUS_DIR, MODEL_DIR, TOPIC_NAMES_DIR, VISUALIZATIONS_DIR
    
    # Determine which steps to run
    if args.steps.lower() == 'all':
        steps = ['download', 'narratives', 'concepts', 'lda', 'mapping', 'visualize']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]
    
    print(f"Running pipeline with steps: {', '.join(steps)}")
    
    # Define paths to check for output files
    output_checks = {
        'download': os.path.join(PROCESSED_DATA_DIR, "DEMO_J.csv"),
        'narratives': os.path.join(CORPUS_DIR, "all_documents.pkl"),
        'concepts': os.path.join(PROCESSED_DATA_DIR, "preprocessed", "corpus_bow.pkl"),
        'lda': os.path.join(MODEL_DIR, f"lda_best_overall_score_{args.num_topics}_topics.gensim"),
        'mapping': os.path.join(TOPIC_NAMES_DIR, f"topic_to_cpt_mappings_lda_best_overall_score_{args.num_topics}_topics.csv"),
        'visualize': os.path.join(VISUALIZATIONS_DIR, f"topic_wordclouds_lda_best_overall_score_{args.num_topics}_topics.png")
    }
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Download NHANES data
    if 'download' in steps:
        if args.skip_existing and check_output_exists(output_checks['download']):
            print("Skipping data download as files already exist")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'data', 'download.py')}", "Data Download")
            if not success:
                print("Exiting pipeline due to download failure")
                return
    
    # Step 2: Generate synthetic narratives
    if 'narratives' in steps:
        if args.skip_existing and check_output_exists(output_checks['narratives']):
            print("Skipping narrative generation as files already exist")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'data', 'narrative_gen.py')}", "Synthetic Narrative Generation")
            if not success:
                print("Exiting pipeline due to narrative generation failure")
                return
    
    # Step 3: Extract concepts and preprocess text
    if 'concepts' in steps:
        if args.skip_existing and check_output_exists(output_checks['concepts']):
            print("Skipping concept extraction as files already exist")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'features', 'concept_extraction.py')}", "Concept Extraction")
            if not success:
                print("Exiting pipeline due to concept extraction failure")
                return
    
    # Step 4: Train and evaluate LDA models
    if 'lda' in steps:
        if args.skip_existing and check_output_exists(output_checks['lda']):
            print(f"Skipping LDA model training as model with {args.num_topics} topics already exists")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'models', 'lda_model.py')}", "LDA Model Training")
            if not success:
                print("Exiting pipeline due to LDA model training failure")
                return
    
    # Step 5: Map topics to CPT codes
    if 'mapping' in steps:
        if args.skip_existing and check_output_exists(output_checks['mapping']):
            print("Skipping topic-to-CPT mapping as files already exist")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'models', 'topic_cpt_mappings.py')}", "Topic-to-CPT Mapping")
            if not success:
                print("Exiting pipeline due to topic mapping failure")
                return
    
    # Step 6: Visualize topics
    if 'visualize' in steps:
        if args.skip_existing and check_output_exists(output_checks['visualize']):
            print("Skipping visualization as files already exist")
        else:
            success = run_command(f"python {os.path.join(BASE_DIR, 'src', 'visualization', 'visualize_lda.py')}", "Topic Visualization")
            if not success:
                print("Exiting pipeline due to visualization failure")
                return
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("="*80)

if __name__ == "__main__":
    main()
