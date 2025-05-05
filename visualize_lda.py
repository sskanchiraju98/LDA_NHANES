#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import json
import numpy as np
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from pathlib import Path
import sys
import argparse

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize LDA topic models')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--viz_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--custom_stopwords', type=str, default=None)
    return parser.parse_args()

def remove_complex(obj):
    if isinstance(obj, complex):
        return obj.real
    elif isinstance(obj, (np.complex64, np.complex128)):
        return float(obj.real)
    elif isinstance(obj, np.number):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def normalize_topic_terms(model):
    topic_terms = model.get_topics()
    normalized_terms = topic_terms / topic_terms.sum(axis=1, keepdims=True)
    model.state.sstats[:] = normalized_terms * model.state.sstats.sum()
    return model

def create_pyldavis_visualization(lda_model, corpus, id2word, output_prefix):
    try:
        lda_model = normalize_topic_terms(lda_model)
        print("Preparing pyLDAvis visualization (this may take a while for large models)...")
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
        html_path = f"{output_prefix}.html"
        pyLDAvis.save_html(vis_data, html_path)
        json_path = f"{output_prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(vis_data.to_dict(), f, default=remove_complex)
        print(f"✅ Saved pyLDAvis visualization to {html_path} and {json_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating pyLDAvis visualization: {e}")
        return False

def create_wordcloud_visualization(lda_model, custom_stopwords, output_path):
    try:
        num_topics = lda_model.num_topics
        cols = min(5, num_topics)
        rows = int((num_topics + cols - 1) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=True, sharey=True)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        
        for i in range(num_topics):
            if rows > 1 and cols > 1:
                ax = axes[i // cols, i % cols]
            else:
                ax = axes[i] if i < len(axes) else None
            
            if ax is None:
                continue
            
            raw_terms = lda_model.show_topic(i, topn=100)
            topic_terms = {word: weight for word, weight in raw_terms if word not in custom_stopwords}
            
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=30,
                colormap='viridis',
                contour_width=1, 
                contour_color='steelblue',
                stopwords=custom_stopwords
            ).generate_from_frequencies(topic_terms)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Topic {i}', fontsize=16)
        
        if rows > 1 or cols > 1:
            for i in range(num_topics, rows * cols):
                if rows > 1 and cols > 1:
                    axes[i // cols, i % cols].axis('off')
                elif i < len(axes):
                    axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Word clouds saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error creating word cloud visualization: {e}")
        return False

def load_custom_stopwords(stopwords_path=None):
    default_stopwords = {
        'activity', 'history', 'report', 'object', 'level', 'difficulty',
        'indicate', 'light', 'physical', 'perform', 'recreational', 'walk',
        'push', 'pull', 'grasp', 'climb', 'carry', 'sit', 'stand', 'period',
        'month', 'quarter', 'year', 'follow', 'elevate', 'admit', 'status',
        'normal', 'within', 'range', 'per', 'time'
    }
    
    if stopwords_path and os.path.exists(stopwords_path):
        try:
            with open(stopwords_path, 'r') as f:
                custom_stopwords = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(custom_stopwords)} custom stopwords from {stopwords_path}")
            return custom_stopwords
        except Exception as e:
            print(f"Error loading custom stopwords: {e}")
    
    print(f"Using {len(default_stopwords)} default stopwords")
    return default_stopwords

def main():
    args = parse_arguments()
    
    BASE_DIR = os.path.abspath("nhanes_lda_project")
    MODEL_DIR = args.model_dir or os.path.join(BASE_DIR, "models")
    PREPROC_DIR = os.path.join(BASE_DIR, "data", "processed", "preprocessed")
    VIZ_DIR = args.viz_dir or os.path.join(BASE_DIR, "results", "visualizations")
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    custom_stopwords = load_custom_stopwords(args.custom_stopwords)
    
    print("Loading corpus and dictionary...")
    try:
        with open(os.path.join(PREPROC_DIR, "id2word.pkl"), "rb") as f:
            id2word = pickle.load(f)
        with open(os.path.join(PREPROC_DIR, "corpus_bow.pkl"), "rb") as f:
            corpus = pickle.load(f)
        print(f"Loaded corpus with {len(corpus)} documents and dictionary with {len(id2word)} words")
    except FileNotFoundError as e:
        print(f"Error: Required corpus files not found - {e}")
        print(f"Make sure preprocessed data exists in {PREPROC_DIR}")
        return
    
    if args.model_name:
        model_path = os.path.join(MODEL_DIR, f"{args.model_name}.gensim")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        models_to_visualize = [(args.model_name, model_path)]
    else:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.gensim')]
        if not model_files:
            print(f"Error: No LDA model files found in {MODEL_DIR}")
            return
        
        models_to_visualize = [(f.split('.')[0], os.path.join(MODEL_DIR, f)) for f in model_files]
    
    print(f"Found {len(models_to_visualize)} model(s) to visualize")
    
    for model_name, model_path in models_to_visualize:
        try:
            print(f"\nProcessing model: {model_name}")
            lda_model = LdaModel.load(model_path)
            num_topics = lda_model.num_topics
            print(f"Model has {num_topics} topics")
            
            pyldavis_prefix = os.path.join(VIZ_DIR, f"topic_model_viz_{model_name}_{num_topics}_topics")
            wordcloud_path = os.path.join(VIZ_DIR, f"topic_wordclouds_{model_name}_{num_topics}_topics.png")
            
            create_pyldavis_visualization(lda_model, corpus, id2word, pyldavis_prefix)
            create_wordcloud_visualization(lda_model, custom_stopwords, wordcloud_path)
            
            print(f"Visualizations for {model_name} complete")
            
        except Exception as e:
            print(f"❌ Error processing model {model_name}: {e}")
    
    print("\n🎯 Visualization process complete!")

if __name__ == "__main__":
    main()
