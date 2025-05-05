# LDA with NHANES

A project applying Latent Dirichlet Allocation (LDA) topic modeling to synthetic narratives derived from NHANES data to uncover health-related topics and map them to medical procedure codes.

## Overview

Clinical notes contain valuable information about patient encounters, but analyzing them at scale is challenging due to their unstructured nature. This project bridges that gap by:

1. Downloading NHANES data (2017-2018 cycle)
2. Generating synthetic clinical narratives from structured data
3. Extracting medical concepts for topic modeling
4. Training and evaluating LDA models 
5. Mapping topics to CPT (Current Procedural Terminology) codes
6. Visualizing topic models and relationships

## Setup

```bash
# Clone the repository
git clone https://github.com/sskanchiraju98/LDA_NHANES.git
cd LDA_NHANES

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required resources
python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger
python -m spacy download en_core_web_md
```

## Pipeline Steps

### 1. Data Download
```bash
python src/data/download.py
```
Downloads NHANES dataset files from CDC and converts them to CSV format.

### 2. Narrative Generation
```bash
python src/data/narrative_gen.py
```
Transforms structured data into synthetic clinical narratives.

### 3. Concept Extraction
```bash
python src/features/concept_extraction.py
```
Extracts medical concepts and prepares text for topic modeling.

### 4. LDA Model Training
```bash
python src/models/lda_model.py
```
Trains multiple LDA models and evaluates them to find the optimal number of topics.

### 5. Topic-to-CPT Mapping
```bash
python src/models/topic_cpt_mappings.py
```
Names topics based on medical domains and maps them to CPT procedure codes.

### 6. Visualization
```bash
python src/visualization/visualize_lda.py
```
Creates interactive visualizations of the LDA topic models.

### Running the Full Pipeline

```bash
python run_pipeline.py
```

Options:
- `--steps`: Specify which steps to run (comma-separated)
- `--skip_existing`: Skip steps where output files already exist
- `--num_topics`: Set number of topics for the LDA model

Example:
```bash
python run_pipeline.py --steps download,narratives --skip_existing
```

## Project Structure

```
LDA_NHANES/
│
├── config.py                  # Configuration settings
├── requirements.txt           # Project dependencies
├── run_pipeline.py            # End-to-end pipeline script
│
├── data/                      # Data directory
│   ├── raw/                   # Raw NHANES XPT files
│   ├── processed/             # Processed data files
│   │   └── preprocessed/      # Preprocessed data for LDA
│   └── corpus/                # Synthetic narrative corpus
│
├── models/                    # Trained models
│   └── medcat/                # MedCAT models
│
├── results/                   # Results and outputs
│   ├── visualizations/        # Topic visualizations
│   └── topic_names/           # Named topics and CPT mappings
│
└── src/                       # Source code
    ├── data/                  # Data processing
    │   ├── download.py        # Download NHANES data
    │   └── narrative_gen.py   # Generate narratives
    │
    ├── features/              # Feature engineering
    │   └── concept_extraction.py  # Extract medical concepts
    │
    ├── models/                # Model training
    │   ├── lda_model.py       # Train LDA models
    │   └── topic_cpt_mappings.py  # Map topics to CPT codes
    │
    └── visualization/         # Visualization code
        └── visualize_lda.py   # Visualize topics
```

## Data Source

This project uses data from the 2017-2018 cycle of the National Health and Nutrition Examination Survey (NHANES), conducted by the Centers for Disease Control and Prevention (CDC).

## Author

Sandeep Kanchiraju
