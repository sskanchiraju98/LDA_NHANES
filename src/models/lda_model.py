#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import LdaModel, CoherenceModel, LdaMulticore
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath("nhanes_lda_project")
PREPROC_DIR = os.path.join(BASE_DIR, "data", "processed", "preprocessed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "model_selection")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_preprocessed_data():
    print("Loading preprocessed data...")
    
    with open(os.path.join(PREPROC_DIR, "processed_docs.pkl"), "rb") as f:
        processed_docs = pickle.load(f)
    
    with open(os.path.join(PREPROC_DIR, "id2word.pkl"), "rb") as f:
        id2word = pickle.load(f)
    
    with open(os.path.join(PREPROC_DIR, "corpus_bow.pkl"), "rb") as f:
        corpus = pickle.load(f)
    
    print(f"Loaded {len(processed_docs)} documents with {len(id2word)} unique terms")
    return processed_docs, id2word, corpus

def calculate_coherence(model, texts, dictionary, coherence_measure='c_v'):
    coherence_model = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence_measure
    )
    return coherence_model.get_coherence()

def calculate_topic_diversity(model, num_words=20):
    topics = [dict(model.show_topic(topicid, num_words)) for topicid in range(model.num_topics)]
    
    unique_words = set()
    for topic in topics:
        unique_words.update(topic.keys())
    
    total_words = model.num_topics * num_words
    
    diversity = len(unique_words) / total_words
    return diversity

def calculate_topic_distance(model):
    topic_vectors = model.get_topics()
    
    similarities = cosine_similarity(topic_vectors)
    
    distances = 1 - similarities
    
    n_topics = model.num_topics
    total_distance = np.sum(distances) - np.trace(distances)
    n_pairs = n_topics * (n_topics - 1)
    
    if n_pairs > 0:
        avg_distance = total_distance / n_pairs
    else:
        avg_distance = 0
    
    return avg_distance

def train_and_evaluate_models(processed_docs, id2word, corpus, topic_range=range(10, 51, 5)):
    results = defaultdict(list)
    models = {}

    print(f"Training and evaluating models with {len(topic_range)} different topic counts...")

    for k in tqdm(topic_range):
        model = LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=k,
            random_state=42,
            chunksize=100,
            passes=5,
            alpha='symmetric',
            per_word_topics=True,
            eval_every=0,
            workers=4
        )

        models[k] = model

        c_v = calculate_coherence(model, processed_docs, id2word, 'c_v')
        diversity = calculate_topic_diversity(model)
        distance = calculate_topic_distance(model)

        results['num_topics'].append(k)
        results['coherence_c_v'].append(c_v)
        results['topic_diversity'].append(diversity)
        results['topic_distance'].append(distance)

        print(f"Topics: {k}, C_v: {c_v:.4f}, Diversity: {diversity:.4f}, Distance: {distance:.4f}")

    results_df = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved evaluation results to {results_file}")

    return results_df, models

def normalize_score(score, is_higher_better=True):
    if len(score) <= 1:
        return np.array([1.0])
    
    min_val = np.min(score)
    max_val = np.max(score)
    
    if max_val == min_val:
        return np.ones_like(score)
    
    normalized = (score - min_val) / (max_val - min_val)
    
    if not is_higher_better:
        normalized = 1 - normalized
    
    return normalized

def find_best_models(results_df):
    results_df['norm_c_v'] = normalize_score(results_df['coherence_c_v'].values)
    results_df['norm_diversity'] = normalize_score(results_df['topic_diversity'].values)
    results_df['norm_distance'] = normalize_score(results_df['topic_distance'].values)
    
    results_df['coherence_composite'] = results_df['norm_c_v']
    
    results_df['overall_score'] = (
        results_df['coherence_composite'] * 0.6 +
        results_df['norm_diversity'] * 0.2 +
        results_df['norm_distance'] * 0.2
    )
    
    best_models = {
        'c_v': results_df.loc[results_df['coherence_c_v'].idxmax()],
        'topic_diversity': results_df.loc[results_df['topic_diversity'].idxmax()],
        'topic_distance': results_df.loc[results_df['topic_distance'].idxmax()],
        'coherence_composite': results_df.loc[results_df['coherence_composite'].idxmax()],
        'overall_score': results_df.loc[results_df['overall_score'].idxmax()]
    }
    
    return best_models

def plot_evaluation_results(results_df, best_models, save_dir=RESULTS_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    axes[0].plot(results_df['num_topics'], results_df['coherence_c_v'], 'o-', label='C_v')
    axes[0].set_title('Coherence Scores by Number of Topics')
    axes[0].set_xlabel('Number of Topics')
    axes[0].set_ylabel('Coherence Score')
    axes[0].axvline(x=best_models['coherence_composite']['num_topics'], color='r', linestyle='--', 
                   label=f"Best Composite ({int(best_models['coherence_composite']['num_topics'])})")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].plot(results_df['num_topics'], results_df['topic_diversity'], 'o-', label='Topic Diversity')
    axes[1].plot(results_df['num_topics'], results_df['topic_distance'], 's-', label='Topic Distance')
    axes[1].set_title('Topic Diversity Metrics by Number of Topics')
    axes[1].set_xlabel('Number of Topics')
    axes[1].set_ylabel('Score')
    axes[1].axvline(x=best_models['topic_diversity']['num_topics'], color='g', linestyle='--', 
                   label=f"Best Diversity ({int(best_models['topic_diversity']['num_topics'])})")
    axes[1].grid(True)
    axes[1].legend()
    
    axes[2].plot(results_df['num_topics'], results_df['overall_score'], 'o-', color='purple')
    axes[2].set_title('Overall Model Score (Combined Metrics)')
    axes[2].set_xlabel('Number of Topics')
    axes[2].set_ylabel('Overall Score')
    axes[2].axvline(x=best_models['overall_score']['num_topics'], color='purple', linestyle='--', 
                   label=f"Best Overall ({int(best_models['overall_score']['num_topics'])})")
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    
    plot_file = os.path.join(save_dir, "model_evaluation_plots.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Saved evaluation plots to {plot_file}")
    
    plt.show()

def save_best_models(models, best_models):
    print("\nSaving best models...")
    
    for metric, result in best_models.items():
        num_topics = int(result['num_topics'])
        
        if num_topics in models:
            model_file = os.path.join(MODEL_DIR, f"lda_best_{metric}_{num_topics}_topics.gensim")
            models[num_topics].save(model_file)
            print(f"Saved best model by {metric} ({num_topics} topics) to {model_file}")

def main():
    processed_docs, id2word, corpus = load_preprocessed_data()
    
    results_df, models = train_and_evaluate_models(processed_docs, id2word, corpus)
    
    best_models = find_best_models(results_df)
    
    plot_evaluation_results(results_df, best_models)
    
    save_best_models(models, best_models)
    
    print("\n===== Best Models Summary =====")
    print(f"Best C_v Coherence: {int(best_models['c_v']['num_topics'])} topics (score: {best_models['c_v']['coherence_c_v']:.4f})")
    print(f"Best Topic Diversity: {int(best_models['topic_diversity']['num_topics'])} topics (score: {best_models['topic_diversity']['topic_diversity']:.4f})")
    print(f"Best Coherence Composite: {int(best_models['coherence_composite']['num_topics'])} topics (score: {best_models['coherence_composite']['coherence_composite']:.4f})")
    print(f"Best Overall Model: {int(best_models['overall_score']['num_topics'])} topics (score: {best_models['overall_score']['overall_score']:.4f})")
    
    overall_best = int(best_models['overall_score']['num_topics'])
    print(f"\n✅ Recommended model: {overall_best} topics")
    print(f"This model balances coherence, diversity, and topic separation.")
    
    return results_df, models, best_models

if __name__ == "__main__":
    main()
