#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# Base project directories
BASE_DIR = os.path.abspath("nhanes_lda_project")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PREPROC_DIR = os.path.join(PROCESSED_DATA_DIR, "preprocessed")
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")
TOPIC_NAMES_DIR = os.path.join(RESULTS_DIR, "topic_names")
MEDCAT_DIR = os.path.join(MODEL_DIR, "medcat")

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CORPUS_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    PREPROC_DIR,
    VISUALIZATIONS_DIR,
    TOPIC_NAMES_DIR,
    MEDCAT_DIR
]:
    os.makedirs(directory, exist_ok=True)

# NHANES data configuration
NHANES_CYCLE = "2017-2018"
NHANES_BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
NHANES_DOWNLOAD_URL = f"https://wwwn.cdc.gov/Nchs/Nhanes/{NHANES_CYCLE.replace('-', '')}/"

REQUIRED_FILES = [
    ("Demographics", "DEMO_J"),
    ("Medical Conditions", "MCQ_J"),
    ("Physical Functioning", "PFQ_J"),
    ("Diet Behavior", "DBQ_J"),
    ("Physical Activity", "PAQ_J"),
    ("Smoking", "SMQ_J"),
    ("Alcohol Use", "ALQ_J"),
    ("Examination", "BPX_J")
]

# Narrative generation parameters
NARRATIVE_TEMPLATES = [
    "Patient is a {AGE}-year-old {GENDER}, identified as {RACE}. Patient has a history of {CONDITIONS}. They report {FUNCTIONAL_LIMITATIONS}.",
    "{GENDER} patient, {AGE} years old, {RACE} ethnicity. Medical history includes {CONDITIONS}. Patient describes {FUNCTIONAL_LIMITATIONS}.",
    "{AGE}-year-old {RACE} {GENDER} presenting with history of {CONDITIONS}. Patient reports {FUNCTIONAL_LIMITATIONS}."
]

# Concept extraction parameters
CONCEPT_EXTRACTION_PARAMS = {
    "min_tokens_per_document": 5,
    "custom_stopwords_file": None,
    "spacy_model": "en_core_web_md",
    "medical_terminology_file": None
}

# LDA model parameters
LDA_PARAMS = {
    "num_topics_range": range(10, 51, 5),
    "passes": 5,
    "iterations": 400,
    "chunksize": 100,
    "random_state": 42,
    "workers": 4,
    "alpha": "symmetric",
    "eval_every": 0,
    "per_word_topics": True
}

# Dictionary filtering parameters
DICTIONARY_FILTER_PARAMS = {
    "no_below": 1,
    "no_above": 0.8
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    "pyldavis_mds": "mmds",
    "wordcloud_width": 800,
    "wordcloud_height": 400,
    "wordcloud_max_words": 30,
    "wordcloud_colormap": "viridis",
    "wordcloud_background_color": "white"
}

# CPT mapping parameters
CPT_MAPPING_PARAMS = {
    "fuzzy_threshold": 90,
    "top_k": 5,
    "min_matched_keywords": 1
}
