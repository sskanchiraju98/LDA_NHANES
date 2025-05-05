#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import json
from collections import Counter, defaultdict
import re
import string
import time
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

BASE_DIR = os.path.abspath("nhanes_lda_project")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MEDCAT_DIR = os.path.join(MODEL_DIR, "medcat")

for directory in [PROCESSED_DIR, MODEL_DIR, MEDCAT_DIR]:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NHANESPreprocessor")

class NHANESMedCATPreprocessor:
    def __init__(self, corpus_path=None):
        self.corpus_path = corpus_path or os.path.join(CORPUS_DIR, "all_documents.pkl")
        
        self.documents = self._load_corpus()
        
        self.vocab = Vocab()
        self.cdb = CDB()
        self.cat = CAT(cdb=self.cdb, vocab=self.vocab)
        
        self.custom_stopwords = self._get_custom_stopwords()
        
        self.medical_ngrams = self._get_medical_ngrams()
        
        try:
            self.nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found, attempting to download...")
            os.system("python -m spacy download en_core_web_md")
            self.nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
            logger.info("SpaCy model downloaded and loaded")
        
        self.processed_docs = None
        self.id2word = None
        self.corpus_bow = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self.document_concepts = {}
        
        self.medical_terminology = self._create_medical_terminology()
    
    def _load_corpus(self):
        logger.info(f"Loading corpus from {self.corpus_path}...")
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, 'rb') as f:
            documents = pickle.load(f)
        
        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents
    
    def _create_medical_terminology(self):
      terminology = {
          "htn": "hypertension",
          "high blood pressure": "hypertension",
          "elevated blood pressure": "hypertension",
          "hypertensive": "hypertension",
          "hypertensive crisis": "hypertension",
          "dm": "diabetes mellitus",
          "t2dm": "type 2 diabetes mellitus",
          "t1dm": "type 1 diabetes mellitus",
          "type ii diabetes": "type 2 diabetes mellitus",
          "type i diabetes": "type 1 diabetes mellitus",
          "diabetes type 2": "type 2 diabetes mellitus",
          "diabetes type 1": "type 1 diabetes mellitus",
          "gestational diabetes": "gestational diabetes",
          "niddm": "type 2 diabetes mellitus",
          "iddm": "type 1 diabetes mellitus",
          "diabetic": "diabetes mellitus",
          "chd": "coronary heart disease",
          "cad": "coronary artery disease",
          "cvd": "cardiovascular disease",
          "heart condition": "heart disease",
          "cardiac disease": "heart disease",
          "coronary disease": "coronary heart disease",
          "ami": "acute myocardial infarction",
          "mi": "myocardial infarction",
          "stemi": "st elevation myocardial infarction",
          "nstemi": "non st elevation myocardial infarction",
          "afib": "atrial fibrillation",
          "arrhythmia": "cardiac arrhythmia",
          "chf": "congestive heart failure",
          "hf": "heart failure",
          "cardiac failure": "heart failure",
          "copd": "chronic obstructive pulmonary disease",
          "chronic bronchitis": "chronic obstructive pulmonary disease",
          "emphysema": "emphysema",
          "asthma": "asthma",
          "asthmatic": "asthma",
          "reactive airway disease": "asthma",
          "respiratory failure": "respiratory failure",
          "cva": "stroke",
          "stroke": "stroke",
          "tpa": "stroke treatment",
          "tia": "transient ischemic attack",
          "ms": "multiple sclerosis",
          "parkinson disease": "parkinson's disease",
          "parkinsons": "parkinson's disease",
          "alzheimers": "alzheimer's disease",
          "dementia": "dementia",
          "migraine": "migraine headache",
          "ckd": "chronic kidney disease",
          "esrd": "end stage renal disease",
          "aki": "acute kidney injury",
          "renal failure": "renal failure",
          "kidney failure": "renal failure",
          "proteinuria": "proteinuria",
          "hep c": "hepatitis c",
          "hep b": "hepatitis b",
          "liver cirrhosis": "cirrhosis",
          "fatty liver": "fatty liver disease",
          "gallstones": "gallbladder disease",
          "gerd": "gastroesophageal reflux disease",
          "ulcer": "peptic ulcer disease",
          "thyroid disease": "thyroid dysfunction",
          "hypothyroid": "hypothyroidism",
          "hyperthyroid": "hyperthyroidism",
          "metabolic syndrome": "metabolic syndrome",
          "breast ca": "breast cancer",
          "lung ca": "lung cancer",
          "colon ca": "colorectal cancer",
          "prostate ca": "prostate cancer",
          "melanoma": "malignant melanoma",
          "hiv": "human immunodeficiency virus",
          "aids": "acquired immunodeficiency syndrome",
          "tb": "tuberculosis",
          "flu": "influenza virus",
          "sle": "systemic lupus erythematosus",
          "lupus": "systemic lupus erythematosus",
          "psa": "psoriatic arthritis",
          "low back pain": "low back pain",
          "knee osteoarthritis": "osteoarthritis knee",
          "hip fracture": "hip fracture",
          "skin cancer": "basal cell carcinoma",
          "psoriasis": "psoriasis vulgaris",
          "diabetic eye disease": "diabetic retinopathy",
          "amd": "age related macular degeneration",
          "cataract": "cataract surgery",
          "glaucoma": "glaucoma diagnosis",
          "smoking": "tobacco use disorder",
          "tobacco use": "tobacco use disorder",
          "alcohol use": "alcohol use disorder",
          "opioid addiction": "opioid use disorder",
          "obese": "obesity",
          "morbid obesity": "morbid obesity",
          "high bmi": "obesity",
          "sedentary lifestyle": "physical inactivity",
          "physical inactivity": "physical inactivity",
          "ldl": "ldl cholesterol",
          "hdl": "hdl cholesterol",
          "hba1c": "hemoglobin a1c",
          "a1c": "hemoglobin a1c",
          "triglycerides": "triglycerides",
          "total cholesterol": "total cholesterol",
          "fasting glucose": "fasting blood sugar"
      }
      
      return terminology

    def _get_custom_stopwords(self):
      stop_words = set(stopwords.words('english'))
      
      domain_specific = {
          'survey', 'completed', 'data', 'unavailable', 'refused', 'don\'t', 'know',
          'missing', 'none', 'not', 'applicable', 'available', 'remember',
          'unknown', 'unspecified', 'participant', 'examination', 'questionnaire',
          'assessment', 'test', 'result', 'response',
          'is', 'has', 'had', 'does', 'did', 'was', 'were', 'be', 'been',
          'being', 'am', 'are', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because',
          'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
          'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
          'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'
      }
      
      stop_words.update(domain_specific)
      
      stop_words.update([chr(i) for i in range(97, 123)])
      stop_words.update([str(i) for i in range(10)])
      
      medical_keywords = {
          "fever", "pain", "infection", "cancer", "asthma", "stroke", "arthritis",
          "migraine", "diabetes", "hepatitis", "kidney", "liver", "heart", "thyroid",
          "hypertension", "anemia", "fracture", "injury", "dementia", "parkinson",
          "covid", "hiv", "aids", "flu", "malaria", "cholesterol", "insulin"
      }
      stop_words -= medical_keywords
      
      return stop_words

    def _get_medical_ngrams(self):
      medical_phrases = {
          "type 1 diabetes", "type 2 diabetes", "diabetes mellitus", "gestational diabetes", "insulin resistance", "blood sugar level",
          "blood pressure", "high blood pressure", "hypertensive crisis", "coronary artery disease", "coronary heart disease",
          "heart disease", "heart attack", "myocardial infarction", "congestive heart failure", "cardiac arrhythmia",
          "atrial fibrillation", "ventricular tachycardia", "sudden cardiac arrest",
          "chronic obstructive pulmonary disease", "chronic bronchitis", "emphysema", "pulmonary fibrosis",
          "reactive airway disease", "asthma exacerbation", "respiratory failure", "hay fever", "seasonal allergies",
          "stroke prevention", "ischemic stroke", "hemorrhagic stroke", "transient ischemic attack",
          "multiple sclerosis", "parkinson's disease", "alzheimer's disease", "migraine headache", "seizure disorder",
          "chronic kidney disease", "end stage renal disease", "acute kidney injury", "renal failure", "proteinuria",
          "liver cirrhosis", "fatty liver disease", "gallbladder disease", "hepatitis c", "hepatitis b", "gastroesophageal reflux",
          "peptic ulcer disease", "gastrointestinal bleeding",
          "thyroid dysfunction", "hypothyroidism", "hyperthyroidism", "metabolic syndrome",
          "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer", "pancreatic cancer", "malignant melanoma",
          "human immunodeficiency virus", "hiv infection", "tuberculosis infection", "influenza virus", "hepatitis virus",
          "covid 19", "covid-19",
          "rheumatoid arthritis", "systemic lupus erythematosus", "psoriatic arthritis",
          "low back pain", "osteoarthritis knee", "hip fracture", "joint replacement surgery",
          "basal cell carcinoma", "melanoma skin cancer", "psoriasis vulgaris",
          "diabetic retinopathy", "age related macular degeneration", "cataract surgery", "glaucoma diagnosis",
          "alcohol use disorder", "substance use disorder", "tobacco use disorder", "opioid overdose",
          "body mass index", "waist circumference", "central obesity",
          "physical inactivity", "sedentary behavior", "smoking cessation", "alcohol cessation",
          "colorectal cancer screening", "lung cancer screening", "breast cancer screening",
          "immunization schedule", "vaccine administration",
          "chronic pain syndrome", "neuropathic pain", "lower back pain"
      }
      
      return set(medical_phrases)

    def standardize_medical_terminology(self, text):
        text = text.lower()
        
        standardized_text = text
        
        sorted_terms = sorted(self.medical_terminology.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            standardized_text = re.sub(pattern, self.medical_terminology[term], standardized_text)
        
        return standardized_text
    
    def extract_medical_concepts(self, text, doc_id=None):
      standardized_text = self.standardize_medical_terminology(text)

      entities = self.cat.get_entities(standardized_text)

      concepts = []
      if not entities.get('entities', []):
          for term, std_form in self.medical_terminology.items():
              pattern = r'\b' + re.escape(term) + r'\b'
              for match in re.finditer(pattern, standardized_text):
                  concept = {
                      'text': match.group(0),
                      'name': std_form,
                      'start': match.start(),
                      'end': match.end(),
                      'type': 'medical_term'
                  }
                  concepts.append(concept)
      else:
          for entity in entities['entities']:
              concept = {
                  'text': entity['source_value'],
                  'name': entity['name'],
                  'start': entity['start'],
                  'end': entity['end'],
                  'type': entity.get('type_ids', ['unknown'])[0] if isinstance(entity.get('type_ids'), list) else entity.get('type_ids')
              }
              concepts.append(concept)

      if doc_id is not None and concepts:
          self.document_concepts[doc_id] = concepts

      return concepts
    
    def detect_ngrams(self, tokens):
        text = ' '.join(tokens)
        
        for phrase in self.medical_ngrams:
            if phrase in text:
                text = text.replace(phrase, phrase.replace(' ', '_'))
        
        preserved_tokens = text.split()
        
        return preserved_tokens
    
    def preprocess_document(self, doc, doc_id=None):
      self.extract_medical_concepts(doc, doc_id)
      
      standardized_doc = self.standardize_medical_terminology(doc)
      
      spacy_doc = self.nlp(standardized_doc.lower())
      
      tokens = []
      for token in spacy_doc:
          if not token.is_punct and not token.is_space:
              lemma = token.lemma_.strip()
              if lemma and lemma not in self.custom_stopwords and len(lemma) > 2:
                  tokens.append(lemma)
      
      tokens = self.detect_ngrams(tokens)
      
      return tokens
    
    def preprocess_corpus(self):
        logger.info("Preprocessing corpus...")
        
        processed_docs = []
        
        for participant_id, doc in tqdm(self.documents.items(), desc="Processing documents"):
            preprocessed = self.preprocess_document(doc, participant_id)
            
            if len(preprocessed) >= 5:
                processed_docs.append(preprocessed)
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        
        id2word = corpora.Dictionary(processed_docs)
        print(f"Vocabulary size BEFORE filtering: {len(id2word)} terms")
        
        id2word.filter_extremes(no_below=1, no_above=0.8)
        print(f"Vocabulary size after filtering: {len(id2word)} terms")
        
        corpus_bow = [id2word.doc2bow(doc) for doc in processed_docs]
        
        logger.info(f"Dictionary contains {len(id2word)} unique tokens")
        
        self.processed_docs = processed_docs
        self.id2word = id2word
        self.corpus_bow = corpus_bow
        
        return processed_docs, id2word, corpus_bow
    
    def build_tfidf_matrix(self):
        logger.info("Building TF-IDF matrix...")
        
        if not self.processed_docs:
            raise ValueError("No processed documents available. Run preprocess_corpus() first")
        
        processed_texts = [' '.join(doc) for doc in self.processed_docs]
        
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=5,
            stop_words='english',
            use_idf=True,
            ngram_range=(1, 3)
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        
        return tfidf_vectorizer, tfidf_matrix
    
    def save_preprocessed_data(self):
        logger.info("Saving preprocessed data...")
        
        preproc_dir = os.path.join(PROCESSED_DIR, "preprocessed")
        os.makedirs(preproc_dir, exist_ok=True)
        
        processed_docs_file = os.path.join(preproc_dir, "processed_docs.pkl")
        with open(processed_docs_file, 'wb') as f:
            pickle.dump(self.processed_docs, f)
        
        dict_file = os.path.join(preproc_dir, "id2word.pkl")
        with open(dict_file, 'wb') as f:
            pickle.dump(self.id2word, f)
        
        corpus_bow_file = os.path.join(preproc_dir, "corpus_bow.pkl")
        with open(corpus_bow_file, 'wb') as f:
            pickle.dump(self.corpus_bow, f)
        
        if self.document_concepts:
            concepts_file = os.path.join(preproc_dir, "document_concepts.pkl")
            with open(concepts_file, 'wb') as f:
                pickle.dump(self.document_concepts, f)
        
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            tfidf_vec_file = os.path.join(preproc_dir, "tfidf_vectorizer.pkl")
            with open(tfidf_vec_file, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            tfidf_matrix_file = os.path.join(preproc_dir, "tfidf_matrix.pkl")
            with open(tfidf_matrix_file, 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
        
        logger.info(f"Preprocessed data saved to {preproc_dir}")
        return preproc_dir
    
    def generate_preproc_stats(self, output_file=None):
        if not self.processed_docs or not self.id2word or not self.corpus_bow:
            raise ValueError("No processed data available. Run preprocess_corpus() first")
        
        stats = {}
        
        stats['original_doc_count'] = len(self.documents)
        stats['processed_doc_count'] = len(self.processed_docs)
        stats['docs_removed_count'] = len(self.documents) - len(self.processed_docs)
        stats['docs_removed_percent'] = (stats['docs_removed_count'] / len(self.documents)) * 100 if len(self.documents) > 0 else 0
        
        original_token_counts = [len(doc.split()) for doc in self.documents.values()]
        processed_token_counts = [len(doc) for doc in self.processed_docs]
        
        stats['avg_original_tokens'] = np.mean(original_token_counts)
        stats['avg_processed_tokens'] = np.mean(processed_token_counts)
        stats['token_reduction_percent'] = ((stats['avg_original_tokens'] - stats['avg_processed_tokens']) / stats['avg_original_tokens']) * 100
        
        stats['vocabulary_size'] = len(self.id2word)
        
        term_freqs = defaultdict(int)
        for doc in self.processed_docs:
            for token in doc:
                term_freqs[token] += 1
        
        stats['most_common_terms'] = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)[:50]
        
        if self.document_concepts:
            total_concepts = sum(len(concepts) for concepts in self.document_concepts.values())
            stats['total_concepts_extracted'] = total_concepts
            stats['avg_concepts_per_doc'] = total_concepts / len(self.document_concepts) if len(self.document_concepts) > 0 else 0
            
            concept_types = defaultdict(int)
            for doc_concepts in self.document_concepts.values():
                for concept in doc_concepts:
                    if concept.get('type'):
                        concept_types[concept['type']] += 1
            
            stats['concept_types'] = dict(concept_types)
            
            concept_names = defaultdict(int)
            for doc_concepts in self.document_concepts.values():
                for concept in doc_concepts:
                    concept_names[concept['name']] += 1
            
            stats['top_concept_names'] = sorted(concept_names.items(), key=lambda x: x[1], reverse=True)[:20]
        
        avg_topics_per_doc = np.mean([len(doc) for doc in self.corpus_bow])
        stats['avg_topics_per_doc'] = avg_topics_per_doc
        
        if self.tfidf_matrix is not None:
            stats['tfidf_shape'] = self.tfidf_matrix.shape
            stats['tfidf_sparsity'] = (1.0 - (self.tfidf_matrix.nnz / float(self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))) * 100
        
        if output_file:
            serializable_stats = {}
            for key, value in stats.items():
                if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    serializable_stats[key] = float(value) if 'float' in str(type(value)) else int(value)
                elif key == 'tfidf_shape' and value is not None:
                    serializable_stats[key] = [int(x) for x in value]
                elif key == 'most_common_terms':
                    serializable_stats[key] = [[term, int(freq)] for term, freq in value]
                elif key == 'top_concept_names':
                    serializable_stats[key] = [[name, int(count)] for name, count in value]
                elif key == 'concept_types':
                    serializable_stats[key] = {k: int(v) for k, v in value.items()}
                else:
                    serializable_stats[key] = value
            
            with open(output_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            
            logger.info(f"Preprocessing statistics saved to {output_file}")
        
        return stats
    
    def get_sample_preprocessed_docs(self, n=5, output_file=None):
        if not self.processed_docs:
            raise ValueError("No processed documents available. Run preprocess_corpus() first")
        
        if len(self.processed_docs) <= n:
            sample_indices = range(len(self.processed_docs))
        else:
            sample_indices = np.random.choice(len(self.processed_docs), n, replace=False)
        
        samples = []
        participant_ids = list(self.documents.keys())
        
        for i in sample_indices:
            if i < len(participant_ids):
                participant_id = participant_ids[i]
                original_doc = self.documents.get(participant_id, "")
                processed_doc = self.processed_docs[i]
                
                sample = {
                    'participant_id': participant_id,
                    'original': original_doc,
                    'processed': processed_doc
                }
                
                if self.document_concepts and participant_id in self.document_concepts:
                    sample['concepts'] = self.document_concepts[participant_id]
                
                samples.append(sample)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("SAMPLE PREPROCESSED DOCUMENTS\n")
                f.write("=============================\n\n")
                
                for i, sample in enumerate(samples, 1):
                    f.write(f"Sample {i} (Participant ID: {sample['participant_id']}):\n")
                    f.write("-" * 80 + "\n")
                    f.write("Original:\n")
                    f.write(sample['original'] + "\n\n")
                    f.write("Processed:\n")
                    f.write(', '.join(sample['processed']) + "\n\n")
                    
                    if 'concepts' in sample:
                        f.write("Extracted Medical Concepts:\n")
                        for concept in sample['concepts']:
                            f.write(f"- '{concept['text']}' -> '{concept['name']}'\n")
                        f.write("\n")
            
            logger.info(f"Sample preprocessed documents saved to {output_file}")
        
        return samples

def download_nltk_resources():
    import nltk
      
    resources = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
      
    for resource in resources:
         try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
            print(f"NLTK resource '{resource}' already downloaded")
         except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)
            print(f"NLTK resource '{resource}' downloaded successfully")

def main():
    print("Starting NHANES text preprocessing pipeline with MedCAT integration...")

    download_nltk_resources()
    
    np.random.seed(42)
    
    try:
        preprocessor = NHANESMedCATPreprocessor()
        
        print("Preprocessing corpus...")
        preprocessor.preprocess_corpus()
        
        print("Building TF-IDF matrix...")
        preprocessor.build_tfidf_matrix()
        
        print("Saving preprocessed data...")
        preproc_dir = preprocessor.save_preprocessed_data()
        
        print("Generating preprocessing statistics...")
        stats_file = os.path.join(preproc_dir, "preproc_stats.json")
        stats = preprocessor.generate_preproc_stats(stats_file)
        
        print("Generating sample preprocessed documents...")
        samples_file = os.path.join(preproc_dir, "sample_preprocessed_docs.txt")
        preprocessor.get_sample_preprocessed_docs(n=10, output_file=samples_file)
        
        print("\nPreprocessing pipeline with MedCAT integration completed successfully!")
        print(f"Processed {stats['processed_doc_count']} documents")
        print(f"Vocabulary size: {stats['vocabulary_size']} unique terms")
        print(f"Token reduction: {stats['token_reduction_percent']:.1f}%")
        
        if 'total_concepts_extracted' in stats:
            print(f"Total medical concepts extracted: {stats['total_concepts_extracted']}")
            print(f"Average concepts per document: {stats['avg_concepts_per_doc']:.2f}")
        
        print(f"Preprocessed data saved to {preproc_dir}")
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
   start_time = time.time()
   main()
   execution_time = time.time() - start_time
   print(f"Total execution time: {execution_time:.2f} seconds")
