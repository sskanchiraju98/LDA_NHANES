#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rapidfuzz import fuzz
from gensim.models import LdaModel

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Name LDA topics and map to CPT codes')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--cpt_path', type=str, default=None)
    parser.add_argument('--fuzzy_threshold', type=int, default=90)
    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()

MEDICAL_DOMAINS = {
    "Diabetes / Endocrinology": ['diabetes', 'glucose', 'insulin', 'blood sugar', 'hba1c', 'hyperglycemia', 'hypoglycemia', 'thyroid', 'hypothyroidism', 'hyperthyroidism', 'endocrine'],
    "Cardiology / Cardiovascular Health": ['hypertension', 'blood pressure', 'cholesterol', 'ldl', 'hdl', 'triglycerides', 'statin', 'heart attack', 'stroke', 'angina', 'arrhythmia', 'heart failure', 'cardiovascular', 'ischemia', 'coronary', 'atherosclerosis'],
    "Respiratory Health": ['asthma', 'copd', 'emphysema', 'bronchitis', 'pneumonia', 'pulmonary', 'respiratory', 'lung', 'oxygen', 'airway', 'breathing', 'dyspnea'],
    "Kidney / Renal Disease": ['kidney', 'renal', 'chronic kidney disease', 'ckd', 'dialysis', 'glomerular', 'nephritis', 'proteinuria', 'creatinine'],
    "Liver / Gastrointestinal Health": ['liver', 'hepatic', 'hepatitis', 'cirrhosis', 'fatty liver', 'gallbladder', 'pancreas', 'digestive', 'gastrointestinal', 'ulcer', 'constipation', 'diarrhea', 'reflux', 'nausea'],
    "Obesity / Metabolism": ['obesity', 'overweight', 'bmi', 'body mass index', 'weight', 'diet', 'nutrition', 'calories', 'metabolism', 'metabolic syndrome'],
    "Musculoskeletal / Orthopedic": ['arthritis', 'osteoarthritis', 'rheumatoid arthritis', 'joint pain', 'bone', 'fracture', 'osteoporosis', 'mobility', 'gait', 'back pain', 'scoliosis', 'injury', 'sprain'],
    "Mental Health / Behavioral Health": ['depression', 'anxiety', 'mental', 'stress', 'psychiatric', 'bipolar', 'ptsd', 'psychological', 'insomnia', 'sleep', 'fatigue'],
    "Cancer / Oncology": ['cancer', 'malignancy', 'tumor', 'chemotherapy', 'radiation', 'metastasis', 'oncology', 'leukemia', 'lymphoma', 'carcinoma'],
    "Substance Use / Addiction": ['smoking', 'tobacco', 'alcohol', 'drinking', 'substance use', 'addiction', 'opioids', 'cocaine', 'marijuana', 'nicotine', 'overdose'],
    "Infectious Disease": ['infection', 'viral', 'bacterial', 'hiv', 'aids', 'influenza', 'tuberculosis', 'hepatitis', 'sepsis', 'vaccination', 'vaccine'],
    "Women's Health": ['pregnancy', 'prenatal', 'postpartum', 'breast', 'mammogram', 'cervical', 'pap smear', 'menopause', 'gynecology', 'contraception'],
    "Men's Health": ['prostate', 'testosterone', 'erectile dysfunction', 'bph', 'prostate cancer'],
    "Pediatrics": ['child', 'pediatric', 'infant', 'baby', 'adolescent', 'newborn', 'immunization', 'growth', 'development'],
    "Geriatrics / Aging": ['aging', 'elderly', 'geriatric', 'falls', 'dementia', 'alzheimers', 'memory loss', 'frailty'],
    "Neurology / Brain Health": ['stroke', 'seizure', 'epilepsy', 'migraine', 'neuropathy', 'multiple sclerosis', 'headache', 'parkinson', 'alzheimer', 'dizziness'],
    "Dermatology / Skin": ['rash', 'eczema', 'psoriasis', 'melanoma', 'acne', 'skin cancer', 'dermatology', 'itching'],
    "Ophthalmology / Vision": ['vision', 'glaucoma', 'cataract', 'macular', 'diabetic retinopathy', 'eye', 'blindness'],
    "Ear, Nose, and Throat (ENT)": ['hearing loss', 'ear infection', 'sinusitis', 'tonsillitis', 'vertigo', 'tinnitus'],
    "Pain Management": ['pain', 'chronic pain', 'opioids', 'analgesics', 'back pain', 'neck pain', 'joint pain', 'migraine'],
    "Immunology / Autoimmune": ['autoimmune', 'lupus', 'rheumatoid', 'multiple sclerosis', 'immune system', 'allergy', 'asthma'],
    "Reproductive Health / Fertility": ['fertility', 'infertility', 'ivf', 'pregnancy', 'miscarriage', 'birth control'],
    "Preventive Medicine / Wellness": ['screening', 'vaccination', 'checkup', 'prevention', 'wellness', 'health maintenance'],
    "General Symptoms / Constitutional": ['fatigue', 'weight loss', 'fever', 'night sweats', 'weakness', 'loss of appetite', 'general health'],
    "Environmental / Occupational Health": ['occupational', 'exposure', 'lead poisoning', 'environmental', 'asbestos', 'toxic'],
    "Lifestyle Factors": ['exercise', 'physical activity', 'sedentary', 'diet', 'nutrition', 'alcohol', 'smoking'],
    "Demographics / Social Determinants": ['age', 'gender', 'race', 'ethnicity', 'education', 'income', 'socioeconomic', 'insurance']
}

GENERIC_WORDS = {
    'male', 'female', 'current', 'former', 'history', 'pattern', 'report', 'light', 'object', 'difficulty', 'unable', 'walk',
    'mile', 'distance', 'recreation', 'recreational', 'blood_pressure', 'diastolic', 'systolic', 'value', 'range', 'normal',
    'stage', 'level', 'high', 'low', 'within', 'test', 'exam', 'survey', 'questionnaire', 'missing', 'unknown', 'response',
    'available', 'applicable', 'remember', 'occasion', 'daily', 'time', 'times', 'month', 'week', 'day', '999', 'multi',
    'single', 'moderate', 'severe', 'physical', 'exercise', 'activity'
}

def load_model_and_extract_paths(args):
    BASE_DIR = os.path.abspath("nhanes_lda_project")
    MODEL_DIR = args.model_dir or os.path.join(BASE_DIR, "models")
    RESULTS_DIR = args.results_dir or os.path.join(BASE_DIR, "results")
    TOPIC_NAMES_DIR = os.path.join(RESULTS_DIR, "topic_names")
    os.makedirs(TOPIC_NAMES_DIR, exist_ok=True)
    
    try:
        best_models
        model_name = 'overall_score'
        num_topics = int(best_models[model_name]['num_topics'])
        model_path = os.path.join(MODEL_DIR, f"lda_best_{model_name}_{num_topics}_topics.gensim")
        print(f"Using best model by {model_name} with {num_topics} topics")
    except NameError:
        if args.model_name:
            model_path = os.path.join(MODEL_DIR, f"{args.model_name}.gensim")
            print(f"Using specified model: {args.model_name}")
        else:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.gensim')]
            if not model_files:
                raise FileNotFoundError("No LDA model files found in the models directory!")
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            model_path = os.path.join(MODEL_DIR, model_files[0])
            print(f"Using most recent model: {model_files[0]}")
        
        num_topics_match = re.search(r'(\d+)_topics', os.path.basename(model_path))
        num_topics = int(num_topics_match.group(1)) if num_topics_match else None
    
    lda_model = LdaModel.load(model_path)
    if num_topics is None:
        num_topics = lda_model.num_topics
    print(f"Model has {num_topics} topics")
    
    model_name_short = os.path.basename(model_path).replace('.gensim', '')
    
    paths = {
        'topic_names_path': os.path.join(TOPIC_NAMES_DIR, f"topic_names_{model_name_short}.json"),
        'topic_keywords_path': os.path.join(TOPIC_NAMES_DIR, f"topic_keywords_{model_name_short}.json"),
        'confidence_scores_path': os.path.join(TOPIC_NAMES_DIR, f"confidence_scores_{model_name_short}.json"),
        'cpt_json_path': os.path.join(TOPIC_NAMES_DIR, f"topic_to_cpt_mappings_{model_name_short}.json"),
        'cpt_csv_path': os.path.join(TOPIC_NAMES_DIR, f"topic_to_cpt_mappings_{model_name_short}.csv"),
        'topic_names_dir': TOPIC_NAMES_DIR
    }
    
    return lda_model, num_topics, model_name_short, paths

def extract_topic_keywords(lda_model, num_topics, top_n=10):
    print("Extracting top keywords per topic...")
    topic_keywords = {}
    for topic_id in range(num_topics):
        terms = lda_model.show_topic(topic_id, topn=top_n)
        keywords = [term[0].lower() for term in terms]
        topic_keywords[topic_id] = keywords
    
    return topic_keywords

def generate_topic_names(topic_keywords):
    print("Generating intelligent topic names...")
    topic_names = {}
    confidence_scores = {}

    for topic_id, keywords in topic_keywords.items():
        filtered_keywords = [kw for kw in keywords if kw not in GENERIC_WORDS]
        keyword_weights = {kw: 1.0 - (i / len(filtered_keywords)) for i, kw in enumerate(filtered_keywords)}

        domain_scores = {}
        for domain, domain_keywords in MEDICAL_DOMAINS.items():
            domain_score = 0
            for kw in filtered_keywords:
                for domain_kw in domain_keywords:
                    if domain_kw in kw or kw in domain_kw:
                        domain_score += keyword_weights.get(kw, 0)
                        break
            domain_scores[domain] = domain_score

        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        best_domains = [d for d, score in sorted_domains if score >= 0.3 * sorted_domains[0][1] and score > 0]
        best_score = sorted_domains[0][1] if sorted_domains else 0

        if best_domains:
            domain_names = "; ".join(best_domains[:2])
            matching_kws = [kw for kw in filtered_keywords[:5] if any(domain_kw in kw or kw in domain_kw for domain in best_domains for domain_kw in MEDICAL_DOMAINS[domain])]
            if matching_kws:
                topic_names[topic_id] = f"{domain_names}: {', '.join(matching_kws[:2])}"
            else:
                topic_names[topic_id] = domain_names
        elif filtered_keywords:
            topic_names[topic_id] = f"Miscellaneous: {', '.join(filtered_keywords[:3])}"
        else:
            topic_names[topic_id] = f"Miscellaneous: {', '.join(keywords[:3])}"

        confidence_scores[topic_id] = round(best_score, 2)
    
    return topic_names, confidence_scores

def save_topic_names(topic_names, topic_keywords, confidence_scores, paths):
    with open(paths['topic_names_path'], 'w') as f:
        json.dump({str(k): v for k, v in topic_names.items()}, f, indent=2)
    
    with open(paths['topic_keywords_path'], 'w') as f:
        json.dump({str(k): v for k, v in topic_keywords.items()}, f, indent=2)
    
    with open(paths['confidence_scores_path'], 'w') as f:
        json.dump({str(k): v for k, v in confidence_scores.items()}, f, indent=2)
    
    print(f"✅ Topic names saved to {paths['topic_names_path']}")
    print(f"✅ Topic keywords saved to {paths['topic_keywords_path']}")
    print(f"✅ Confidence scores saved to {paths['confidence_scores_path']}")

def load_cpt_codes(cpt_path=None):
    if cpt_path is None:
        BASE_DIR = os.path.abspath("nhanes_lda_project")
        default_paths = [
            os.path.join(BASE_DIR, "data", "raw", "CptCoding_LDA.csv"),
            os.path.join(BASE_DIR, "data", "processed", "CptCoding_LDA.csv"),
            os.path.join(BASE_DIR, "CptCoding_LDA.csv")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                cpt_path = path
                break
    
    if cpt_path is None or not os.path.exists(cpt_path):
        raise FileNotFoundError(f"CPT code file not found. Please specify using --cpt_path")
    
    cpt_df = pd.read_csv(cpt_path)
    cpt_df['display_norm'] = cpt_df['display'].str.lower()
    
    print(f"✅ Loaded {len(cpt_df)} CPT codes from {cpt_path}")
    return cpt_df

def map_topics_to_cpt(topic_keywords, topic_names, cpt_df, fuzzy_threshold=90, top_k=5, min_matched=1):
    print(f"Mapping topics to CPT codes using fuzzy matching (threshold: {fuzzy_threshold})...")
    results = {}
    match_type_counter = {'mixed': 0}

    for topic_id, keywords in topic_keywords.items():
        matched_cpts_dict = {}

        for idx, row in cpt_df.iterrows():
            description = row['display_norm']
            matched_keywords = set()
            
            for keyword in keywords:
                keyword = keyword.lower()
                
                if keyword in description:
                    matched_keywords.add(keyword)
                else:
                    score = fuzz.partial_ratio(keyword, description)
                    if score >= fuzzy_threshold:
                        matched_keywords.add(keyword)

            if len(matched_keywords) >= min_matched:
                fuzzy_scores = [fuzz.partial_ratio(kw, description) for kw in matched_keywords]
                avg_fuzzy_score = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0

                matched_cpts_dict[row['code']] = {
                    'cpt_code': row['code'],
                    'cpt_display': row['display'],
                    'match_type': 'mixed',
                    'match_score': round(avg_fuzzy_score, 1),
                    'matched_keywords': list(matched_keywords),
                    'num_keywords_matched': len(matched_keywords)
                }

        matched_cpts = list(matched_cpts_dict.values())
        matched_cpts = sorted(matched_cpts, key=lambda x: (x['num_keywords_matched'], x['match_score']), reverse=True)
        top_matches = matched_cpts[:top_k]
        results[str(topic_id)] = top_matches

        match_type_counter['mixed'] += len(top_matches)
    
    return results, match_type_counter

def save_cpt_mappings(results, topic_names, paths):
    for topic_id, matches in results.items():
        topic_name = topic_names.get(int(topic_id), f"Topic {topic_id}")
        print(f"\n🏷️ {topic_name} ({len(matches)} matches)")
        for match in matches:
            print(f"  - {match['cpt_code']}: {match['cpt_display']} "
                  f"(Match Type: {match['match_type']}, "
                  f"Score: {match['match_score']}, "
                  f"Matched Keywords: {match['matched_keywords']})")

    with open(paths['cpt_json_path'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved JSON results to {paths['cpt_json_path']}")

    all_rows = []
    for topic_id, matches in results.items():
        for match in matches:
            all_rows.append({
                'topic_id': topic_id,
                'topic_name': topic_names.get(int(topic_id), f"Topic {topic_id}"),
                'cpt_code': match['cpt_code'],
                'cpt_display': match['cpt_display'],
                'match_type': match['match_type'],
                'match_score': match['match_score'],
                'num_keywords_matched': match['num_keywords_matched'],
                'matched_keywords': ', '.join(match['matched_keywords'])
            })

    matches_df = pd.DataFrame(all_rows)
    matches_df.to_csv(paths['cpt_csv_path'], index=False)
    print(f"✅ Saved CSV results to {paths['cpt_csv_path']}")

def plot_match_types(match_type_counter, output_dir):
    plt.figure(figsize=(6, 4))
    plt.bar(match_type_counter.keys(), match_type_counter.values(), color=['skyblue'])
    plt.title('Number of Matches by Match Type')
    plt.ylabel('Count')
    plt.xlabel('Match Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "match_type_counts.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Match type counts plot saved to {output_path}")

def main():
    args = parse_arguments()
    
    try:
        lda_model, num_topics, model_name_short, paths = load_model_and_extract_paths(args)
        
        topic_keywords = extract_topic_keywords(lda_model, num_topics)
        
        topic_names, confidence_scores = generate_topic_names(topic_keywords)
        
        print("\nTopic Names:")
        print("============")
        for topic_id in sorted(topic_names.keys()):
            print(f"🏷️ Topic {topic_id}: {topic_names[topic_id]} (Confidence: {confidence_scores[topic_id]})")
        
        save_topic_names(topic_names, topic_keywords, confidence_scores, paths)
        
        cpt_df = load_cpt_codes(args.cpt_path)
        
        results, match_counter = map_topics_to_cpt(
            topic_keywords, 
            topic_names, 
            cpt_df, 
            fuzzy_threshold=args.fuzzy_threshold if args.fuzzy_threshold else 90,
            top_k=args.top_k if args.top_k else 5
        )
        
        save_cpt_mappings(results, topic_names, paths)
        
        plot_match_types(match_counter, paths['topic_names_dir'])
        
        print("\n✅ Topic naming and CPT mapping process complete!")
        
    except Exception as e:
        print(f"❌ Error in topic naming and CPT mapping process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
