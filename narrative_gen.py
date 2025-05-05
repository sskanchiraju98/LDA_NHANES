#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict, Counter
import re
from tqdm import tqdm
import time

BASE_DIR = os.path.abspath("nhanes_lda_project")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")

os.makedirs(CORPUS_DIR, exist_ok=True)

class NHANESCorpusCreator:
    def __init__(self, datasets=None):
        self.datasets = datasets if datasets is not None else self._load_datasets()
        self.participant_documents = {}
        self.participant_data_coverage = defaultdict(set)
        self._map_participant_coverage()
        self.codebook_mappings = self._load_codebook_mappings()
        self.medical_conditions_dict = self._load_medical_conditions_dict()

    def _load_datasets(self):
        datasets = {}
        print("Loading NHANES datasets from CSV files...")
        csv_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]

        for csv_file in csv_files:
            dataset_name = os.path.splitext(csv_file)[0]
            file_path = os.path.join(PROCESSED_DATA_DIR, csv_file)
            datasets[dataset_name] = pd.read_csv(file_path)
            print(f"Loaded {dataset_name} with {len(datasets[dataset_name])} rows")

        print(f"Loaded {len(datasets)} NHANES datasets")
        return datasets

    def _map_participant_coverage(self):
        for dataset_name, df in self.datasets.items():
            if 'SEQN' in df.columns:
                participant_ids = set(df['SEQN'])
                for participant_id in participant_ids:
                    self.participant_data_coverage[participant_id].add(dataset_name)

        total_participants = len(self.participant_data_coverage)
        print(f"Found {total_participants} total participants across all datasets")

        all_datasets = set(self.datasets.keys())
        all_coverage_count = sum(1 for datasets in self.participant_data_coverage.values()
                             if datasets == all_datasets)
        print(f"Participants with data in all datasets: {all_coverage_count}")

    def _load_codebook_mappings(self):
        mappings_file = os.path.join(PROCESSED_DATA_DIR, "codebook_mappings.json")

        if os.path.exists(mappings_file):
            with open(mappings_file, 'r') as f:
                return json.load(f)

        mappings = self._create_default_mappings()

        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2)

        return mappings

    def _load_medical_conditions_dict(self):
        conditions_file = os.path.join(PROCESSED_DATA_DIR, "medical_conditions.json")

        if os.path.exists(conditions_file):
            with open(conditions_file, 'r') as f:
                return json.load(f)

        conditions = {
          "MCQ010": "asthma",
          "MCQ035": "age asthma first diagnosed",
          "MCQ040": "current asthma status",
          "MCQ050": "hay fever",
          "MCQ060": "eczema",
          "MCQ070": "food allergies",
          "MCQ080": "seasonal allergies",
          "MCQ090": "drug allergies",
          "MCQ160A": "hay fever",
          "MCQ160B": "congestive heart failure",
          "MCQ160C": "coronary heart disease",
          "MCQ160D": "angina pectoris",
          "MCQ160E": "heart attack (myocardial infarction)",
          "MCQ160F": "stroke",
          "MCQ160G": "emphysema",
          "MCQ160H": "chronic bronchitis",
          "MCQ160I": "liver disease",
          "MCQ160J": "ulcer",
          "MCQ160K": "diabetes",
          "MCQ160L": "thyroid disease",
          "MCQ160M": "kidney disease",
          "MCQ160N": "arthritis",
          "MCQ160O": "chronic bronchitis",
          "MCQ160P": "hypertension",
          "MCQ160Q": "high cholesterol",
          "MCQ160R": "sleep disorder",
          "MCQ160S": "anemia",
          "MCQ160T": "psoriasis",
          "MCQ160U": "lupus",
          "DIQ010": "doctor told you have diabetes",
          "DIQ050": "age when diabetes diagnosed",
          "DIQ070": "now taking insulin",
          "DIQ170": "gestational diabetes",
          "BPQ020": "ever told you have high blood pressure",
          "BPQ030": "high blood pressure on 2+ visits",
          "BPQ040A": "currently taking medication for high blood pressure",
          "MCQ220": "cancer or malignancy",
          "MCQ230A": "first cancer type",
          "MCQ230B": "second cancer type",
          "MCQ230C": "third cancer type",
          "MCQ240A": "age first diagnosed with cancer",
          "MCQ240B": "age second diagnosed with cancer",
          "MCQ240C": "age third diagnosed with cancer",
          "ARQ050": "ever told you had arthritis",
          "ARQ060": "ever told you had rheumatoid arthritis",
          "ARQ070": "ever told you had osteoarthritis",
          "ARQ080": "ever told you had gout",
          "RDQ070": "wheezing or whistling in chest",
          "RDQ080": "asthma attack",
          "RDQ090": "asthma emergency room visit",
          "SLQ050": "sleep apnea",
          "SLQ060": "doctor diagnosed sleep disorder",
          "KIQ022": "weak/failing kidneys diagnosis",
          "KIQ026": "currently on dialysis",
          "MCQ300A": "hepatitis A",
          "MCQ300B": "hepatitis B",
          "MCQ300C": "hepatitis C",
          "IMQ020": "received hepatitis A vaccine",
          "IMQ030": "received hepatitis B vaccine",
          "IMQ040": "received hepatitis C vaccine",
          "IMQ050": "received influenza vaccine",
          "DPQ020": "feeling down or depressed",
          "DPQ030": "feeling little interest or pleasure",
          "DPQ040": "trouble sleeping",
          "DPQ050": "feeling tired or little energy",
          "DPQ060": "poor appetite or overeating",
          "DPQ070": "feeling bad about yourself",
          "DPQ080": "trouble concentrating",
          "DPQ090": "moving/speaking slowly or fidgety",
          "DPQ100": "thoughts of self-harm",
          "SMQ020": "smoking status",
          "SMQ040": "cigarettes per day",
          "ALQ130": "number of drinks per day",
          "ALQ120Q": "alcohol use frequency",
          "RHQ030": "pregnant now",
          "RHQ540": "age at first live birth",
          "BMXBMI": "body mass index",
          "BMXWT": "body weight",
          "BMXHT": "body height",
          "VIQ020": "trouble seeing even with glasses",
          "VIQ050": "eye exam in past 2 years",
          "AUQ110": "hearing trouble",
          "AUQ100": "ear infection history",
          "HUQ010": "general health condition",
          "HUQ030": "physical health bad days",
          "HUQ040": "mental health bad days",
          "IMQ040": "psoriasis",
          "MCQ365A": "eczema",
          "MCQ366A": "psoriasis",
          "MCQ367A": "rosacea",
          "CFQ010": "chronic fatigue syndrome diagnosis",
        }

        with open(conditions_file, 'w') as f:
            json.dump(conditions, f, indent=2)

        return conditions

    def _get_numeric_category_statement(self, row, column, term, thresholds, descriptions):
      if column not in row or pd.isna(row[column]):
          return ""

      value = row[column]
      for threshold in sorted(thresholds.keys()):
          if value <= threshold:
              desc = thresholds[threshold]
              return descriptions.get(desc, "")

      max_desc = list(descriptions.values())[-1]
      return max_desc

    def _create_default_mappings(self):
        mappings = {
            "RIAGENDR": {
                "1": "male",
                "2": "female"
            },
            "RIDRETH3": {
                "1": "Mexican American",
                "2": "Other Hispanic",
                "3": "Non-Hispanic White",
                "4": "Non-Hispanic Black",
                "6": "Non-Hispanic Asian",
                "7": "Other Race/Multi-Racial"
            },
            "HUQ010": {
                "1": "excellent",
                "2": "very good",
                "3": "good",
                "4": "fair",
                "5": "poor"
            },
            "YES_NO": {
                "1": "yes",
                "2": "no"
            },
            "FREQUENCY": {
                "0": "never",
                "1": "rarely",
                "2": "sometimes",
                "3": "often",
                "4": "always"
            },
            "DIFFICULTY": {
                "1": "no difficulty",
                "2": "some difficulty",
                "3": "much difficulty",
                "4": "unable to do"
            },
            "SMOKING": {
                "1": "current smoker",
                "2": "former smoker",
                "3": "never smoked"
            },
            "ALCOHOL": {
                "0": "never drinks alcohol",
                "1": "drinks alcohol less than once per month",
                "2": "drinks alcohol 1-3 times per month",
                "3": "drinks alcohol 1-2 times per week",
                "4": "drinks alcohol 3-4 times per week",
                "5": "drinks alcohol almost daily"
            },
            "ACTIVITY": {
                "1": "sedentary",
                "2": "light activity",
                "3": "moderate activity",
                "4": "vigorous activity"
            }
        }

        return mappings

    def _get_condition_statement(self, row, condition_col, dataset=None):
      condition_name = self.medical_conditions_dict.get(condition_col, condition_col)

      if dataset and dataset in self.datasets:
          df = self.datasets[dataset]
          if 'SEQN' in row and row['SEQN'].values:
              participant_id = row['SEQN']
              matching_rows = df[df['SEQN'] == participant_id]
              if not matching_rows.empty and condition_col in matching_rows.columns:
                  value = matching_rows[condition_col].iloc[0]
              else:
                  return ""
          else:
              return ""
      elif condition_col in row:
          value = row[condition_col]
      else:
          return ""

      if pd.isna(value):
          return ""

      major_diseases = {
          "cancer or malignancy",
          "stroke",
          "heart attack (myocardial infarction)",
          "hepatitis A",
          "hepatitis B",
          "hepatitis C",
          "chronic kidney disease",
          "diabetes",
          "coronary heart disease",
          "congestive heart failure",
      }

      if value == 1:
          if condition_name.lower() in major_diseases:
              return f"Patient has been diagnosed with {condition_name.lower()}. "
          else:
              return f"Patient has {condition_name.lower()}. "
      elif value == 2:
          if condition_name.lower() in major_diseases:
              return f"Patient has no history of {condition_name.lower()}. "
          else:
              return ""
      else:
          return ""

    def _get_categorical_statement(self, row, column, mapping_key, prefix="", suffix="", dataset=None):
        if mapping_key in self.codebook_mappings:
            mapping = self.codebook_mappings[mapping_key]
        else:
            return ""

        if dataset and dataset in self.datasets:
            df = self.datasets[dataset]
            if 'SEQN' in row and row['SEQN'] in df['SEQN'].values:
                participant_id = row['SEQN']
                matching_rows = df[df['SEQN'] == participant_id]
                if not matching_rows.empty and column in matching_rows.columns:
                    value = matching_rows[column].iloc[0]
                else:
                    return ""
            else:
                return ""
        elif column in row:
            value = row[column]
        else:
            return ""

        if pd.isna(value):
            return ""

        value_key = str(int(value)) if pd.notna(value) and not isinstance(value, str) else str(value)

        if value_key in mapping:
            return f"{prefix}{mapping[value_key]}{suffix}. "
        else:
            return ""

    def _get_numeric_statement(self, row, column, term, thresholds=None, units="", dataset=None):
        if dataset and dataset in self.datasets:
            df = self.datasets[dataset]
            if 'SEQN' in row and row['SEQN'] in df['SEQN'].values:
                participant_id = row['SEQN']
                matching_rows = df[df['SEQN'] == participant_id]
                if not matching_rows.empty and column in matching_rows.columns:
                    value = matching_rows[column].iloc[0]
                else:
                    return ""
            else:
                return ""
        elif column in row:
            value = row[column]
        else:
            return ""

        if pd.isna(value):
            return ""

        try:
            value = float(value)
        except (ValueError, TypeError):
            return ""

        if thresholds:
            numeric_thresholds = {float(k): v for k, v in thresholds.items()}

            for threshold in sorted(numeric_thresholds.keys()):
                if value <= threshold:
                    return f"{term} is {numeric_thresholds[threshold]} ({value:.1f}{units}). "

            return f"{term} is very high ({value:.1f}{units}). "
        else:
            return f"{term} is {value:.1f}{units}. "

    def create_synthetic_documents(self):
      print("Creating synthetic medical documents...")

      if 'DEMO_J' not in self.datasets or 'SEQN' not in self.datasets['DEMO_J'].columns:
          print("Error: Demographics data missing or no SEQN column")
          return {}

      participant_ids = set(self.datasets['DEMO_J']['SEQN'])
      print(f"Found {len(participant_ids)} participants in demographics data")

      bp_systolic_thresholds = {
          120: "normal",
          130: "elevated",
          140: "stage 1 hypertension",
          180: "stage 2 hypertension"
      }

      bp_diastolic_thresholds = {
          80: "normal",
          90: "stage 1 hypertension",
          120: "stage 2 hypertension"
      }

      bmi_thresholds = {
          18.5: "underweight",
          25: "normal weight",
          30: "overweight",
          35: "obese (class 1)",
          40: "obese (class 2)"
      }

      documents_created = 0

      for participant_id in tqdm(participant_ids, desc="Creating documents"):
          document_parts = []

          demo_df = self.datasets['DEMO_J']
          if participant_id not in demo_df['SEQN'].values:
              continue

          demo_row = demo_df[demo_df['SEQN'] == participant_id].iloc[0]

          if 'DEMO_J' in self.participant_data_coverage[participant_id]:
              if 'RIDAGEYR' in demo_row and not pd.isna(demo_row['RIDAGEYR']):
                  age = int(demo_row['RIDAGEYR'])
                  if age < 18:
                      document_parts.append(f"Patient is a {age}-year-old child.")
                  elif age < 65:
                      document_parts.append(f"Patient is a {age}-year-old adult.")
                  else:
                      document_parts.append(f"Patient is a {age}-year-old older adult.")

              gender = self._get_categorical_statement(demo_row, 'RIAGENDR', 'RIAGENDR', "Patient is a ")
              if gender:
                  document_parts.append(gender)

              ethnicity = self._get_categorical_statement(demo_row, 'RIDRETH3', 'RIDRETH3', "Patient is ")
              if ethnicity:
                  document_parts.append(ethnicity)

              bmi_statement = self._get_numeric_category_statement(
                  demo_row, 'BMXBMI', "BMI", bmi_thresholds,
                  {
                      "underweight": "Patient is underweight.",
                      "normal weight": "Patient has normal weight.",
                      "overweight": "Patient is overweight.",
                      "obese (class 1)": "Patient is classified as obese class 1.",
                      "obese (class 2)": "Patient is classified as obese class 2."
                  }
              )
              if bmi_statement:
                  document_parts.append(bmi_statement)

          if 'MCQ_J' in self.participant_data_coverage[participant_id]:
              mc_df = self.datasets['MCQ_J']
              if participant_id in mc_df['SEQN'].values:
                  mc_row = mc_df[mc_df['SEQN'] == participant_id].iloc[0]
                  for col in mc_df.columns:
                      if col in self.medical_conditions_dict:
                          statement = self._get_condition_statement(mc_row, col)
                          if statement:
                              document_parts.append(statement)

          if 'PFQ_J' in self.participant_data_coverage[participant_id]:
              pf_df = self.datasets['PFQ_J']
              if participant_id in pf_df['SEQN'].values:
                  pf_row = pf_df[pf_df['SEQN'] == participant_id].iloc[0]
                  pf_variables = {
                      'PFQ061B': 'walking a quarter mile',
                      'PFQ061C': 'standing for long periods',
                      'PFQ061D': 'sitting for long periods',
                      'PFQ061E': 'climbing stairs',
                      'PFQ061F': 'stooping, crouching, or kneeling',
                      'PFQ061G': 'reaching overhead',
                      'PFQ061H': 'grasping small objects',
                      'PFQ061I': 'carrying heavy objects',
                      'PFQ061J': 'pushing or pulling large objects'
                  }
                  for col, desc in pf_variables.items():
                      difficulty = self._get_categorical_statement(pf_row, col, 'DIFFICULTY')
                      if difficulty:
                          document_parts.append(f"Patient reports {difficulty.lower()} when {desc}.")

          if 'DBQ_J' in self.participant_data_coverage[participant_id]:
              diet_df = self.datasets['DBQ_J']
              if participant_id in diet_df['SEQN'].values:
                  diet_row = diet_df[diet_df['SEQN'] == participant_id].iloc[0]
                  if 'DBD895' in diet_row and not pd.isna(diet_row['DBD895']):
                      if diet_row['DBD895'] == 1:
                          document_parts.append("Patient adheres to a special diet.")
                      else:
                          document_parts.append("Patient does not follow a special diet.")

          if 'SMQ_J' in self.participant_data_coverage[participant_id]:
              smoke_df = self.datasets['SMQ_J']
              if participant_id in smoke_df['SEQN'].values:
                  smoke_row = smoke_df[smoke_df['SEQN'] == participant_id].iloc[0]
                  smoking_status = self._get_categorical_statement(smoke_row, 'SMQ020', 'SMOKING')
                  if smoking_status:
                      document_parts.append(f"Patient is a {smoking_status.lower()}.")

          if 'ALQ_J' in self.participant_data_coverage[participant_id]:
              alc_df = self.datasets['ALQ_J']
              if participant_id in alc_df['SEQN'].values:
                  alc_row = alc_df[alc_df['SEQN'] == participant_id].iloc[0]
                  alcohol_frequency_cols = ['ALQ120Q', 'ALQ120U', 'ALQ151']
                  alcohol_freq_col = next((col for col in alcohol_frequency_cols if col in alc_row.index), None)
                  if alcohol_freq_col:
                      drinking_status = self._get_categorical_statement(alc_row, alcohol_freq_col, 'ALCOHOL')
                      if drinking_status:
                          document_parts.append(f"{drinking_status.lower()}.")

          if 'PAQ_J' in self.participant_data_coverage[participant_id]:
              pa_df = self.datasets['PAQ_J']
              if participant_id in pa_df['SEQN'].values:
                  pa_row = pa_df[pa_df['SEQN'] == participant_id].iloc[0]
                  work_activity = self._get_categorical_statement(pa_row, 'PAQ605', 'ACTIVITY')
                  recreation_activity = self._get_categorical_statement(pa_row, 'PAQ650', 'ACTIVITY')
                  if work_activity:
                    document_parts.append(f"Patient performs {work_activity.lower()} at work.")
                  if recreation_activity:
                    document_parts.append(f"Patient has {recreation_activity.lower()} during recreational activities.")

                  if recreation_activity:
                      document_parts.append(f"Physical activity level during recreation: {recreation_activity.lower()}.")

          if 'BPX_J' in self.participant_data_coverage[participant_id]:
              exam_df = self.datasets['BPX_J']
              if participant_id in exam_df['SEQN'].values:
                  exam_row = exam_df[exam_df['SEQN'] == participant_id].iloc[0]
                  systolic_statement = self._get_numeric_category_statement(
                      exam_row, 'BPXSY1', "Systolic blood pressure", bp_systolic_thresholds,
                      {
                          "normal": "Systolic blood pressure is within normal range.",
                          "elevated": "Systolic blood pressure is elevated.",
                          "stage 1 hypertension": "Systolic blood pressure indicates stage 1 hypertension.",
                          "stage 2 hypertension": "Systolic blood pressure indicates stage 2 hypertension."
                      }
                  )
                  diastolic_statement = self._get_numeric_category_statement(
                      exam_row, 'BPXDI1', "Diastolic blood pressure", bp_diastolic_thresholds,
                      {
                          "normal": "Diastolic blood pressure is within normal range.",
                          "stage 1 hypertension": "Diastolic blood pressure indicates stage 1 hypertension.",
                          "stage 2 hypertension": "Diastolic blood pressure indicates stage 2 hypertension."
                      }
                  )
                  if systolic_statement:
                      document_parts.append(systolic_statement)
                  if diastolic_statement:
                      document_parts.append(diastolic_statement)

          full_document = " ".join(document_parts).strip()

          if full_document:
              self.participant_documents[participant_id] = full_document
              documents_created += 1

      print(f"Created synthetic documents for {documents_created} participants")
      return self.participant_documents

    def save_documents(self, output_dir=None):
        if output_dir is None:
            output_dir = CORPUS_DIR

        os.makedirs(output_dir, exist_ok=True)

        for participant_id, document in tqdm(self.participant_documents.items(), desc="Saving documents"):
            file_path = os.path.join(output_dir, f"participant_{participant_id}.txt")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(document)

        all_docs_file = os.path.join(output_dir, "all_documents.pkl")

        with open(all_docs_file, 'wb') as f:
            pickle.dump(self.participant_documents, f)

        docs_df = pd.DataFrame({
            'participant_id': list(self.participant_documents.keys()),
            'document': list(self.participant_documents.values())
        })

        docs_csv_file = os.path.join(output_dir, "all_documents.csv")
        docs_df.to_csv(docs_csv_file, index=False)

        print(f"Saved {len(self.participant_documents)} documents to {output_dir}")
        print(f"All documents saved to {all_docs_file} and {docs_csv_file}")

        return output_dir

    def create_corpus_stats(self, output_dir=None):
      if output_dir is None:
          output_dir = CORPUS_DIR

      os.makedirs(output_dir, exist_ok=True)

      stats = {}

      stats['document_count'] = len(self.participant_documents)

      doc_lengths = [len(doc.split()) for doc in self.participant_documents.values()]
      stats['avg_document_length'] = float(np.mean(doc_lengths)) if doc_lengths else 0
      stats['min_document_length'] = int(np.min(doc_lengths)) if doc_lengths else 0
      stats['max_document_length'] = int(np.max(doc_lengths)) if doc_lengths else 0
      stats['median_document_length'] = float(np.median(doc_lengths)) if doc_lengths else 0

      all_words = []
      for doc in self.participant_documents.values():
          all_words.extend(doc.lower().split())

      unique_words = set(all_words)
      stats['vocabulary_size'] = len(unique_words)
      stats['total_word_count'] = len(all_words)

      word_freq = Counter(all_words)
      stats['most_common_words'] = [(word, int(count)) for word, count in word_freq.most_common(50)]

      stats_file = os.path.join(output_dir, "corpus_stats.json")

      serializable_stats = {}
      for key, value in stats.items():
          if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
              serializable_stats[key] = float(value) if 'float' in str(type(value)) else int(value)
          elif key == 'most_common_words':
              serializable_stats[key] = [[word, int(count)] for word, count in value]
          else:
              serializable_stats[key] = value

      with open(stats_file, 'w') as f:
          json.dump(serializable_stats, f, indent=2)

      print(f"Corpus statistics saved to {stats_file}")

      return stats

    def generate_sample_documents(self, n=5, output_file=None):
        if not self.participant_documents:
            print("No documents available. Run create_synthetic_documents() first")
            return []

        if len(self.participant_documents) <= n:
            sample_ids = list(self.participant_documents.keys())
        else:
            sample_ids = np.random.choice(list(self.participant_documents.keys()), n, replace=False)

        samples = [(pid, self.participant_documents[pid]) for pid in sample_ids]

        if output_file:
            with open(output_file, 'w') as f:
                f.write("SAMPLE SYNTHETIC MEDICAL NARRATIVES\n")
                f.write("=================================\n\n")

                for i, (pid, doc) in enumerate(samples, 1):
                    f.write(f"Sample {i} (Participant ID: {pid}):\n")
                    f.write("-" * 80 + "\n")
                    f.write(doc + "\n\n")

        return samples

def main():
    print("Starting NHANES text corpus creation process...")

    try:
        print("Initializing corpus creator...")
        corpus_creator = NHANESCorpusCreator()

        print("Creating synthetic medical narratives...")
        corpus_creator.create_synthetic_documents()

        print("Saving documents to disk...")
        output_dir = corpus_creator.save_documents()

        print("Generating corpus statistics...")
        stats = corpus_creator.create_corpus_stats()

        print("Generating sample documents...")
        samples_file = os.path.join(CORPUS_DIR, "sample_documents.txt")
        corpus_creator.generate_sample_documents(n=10, output_file=samples_file)

        print(f"Corpus creation complete! Documents saved to {output_dir}")
        print(f"Created {stats['document_count']} documents with average length of {stats['avg_document_length']:.1f} words")
        print(f"Vocabulary size: {stats['vocabulary_size']} unique words")
        print(f"Sample documents saved to {samples_file}")

    except Exception as e:
        print(f"Error in corpus creation process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(42)
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
