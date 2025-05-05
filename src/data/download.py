#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

BASE_DIR = os.path.abspath("nhanes_lda_project")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

for directory in [BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

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

BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
DATA_DOWNLOAD_URL = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/"

def search_nhanes_files(file_code):
    params = {
        'Component': '',
        'CycleBeginYear': '2017',
        'name': file_code
    }

    search_url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        response = requests.get(search_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'.*\.XPT$', re.IGNORECASE))

        if links:
            for link in links:
                href = link.get('href')
                if file_code.upper() in href.upper():
                    return f"https://wwwn.cdc.gov{href}" if href.startswith('/') else href

        direct_url = f"{DATA_DOWNLOAD_URL}{file_code}.XPT"
        return direct_url

    except Exception as e:
        print(f"Error searching for {file_code}: {e}")
        return None

def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(output_path, 'wb') as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=block_size),
                total=total_size // block_size,
                unit='KB',
                desc=os.path.basename(output_path)
            ):
                if chunk:
                    f.write(chunk)

        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def xpt_to_csv(xpt_file, csv_file):
    try:
        data = pd.read_sas(xpt_file, format='xport')
        data.to_csv(csv_file, index=False)
        print(f"Converted {os.path.basename(xpt_file)} to CSV")
        return True

    except Exception as e:
        print(f"Error converting {xpt_file} to CSV: {e}")
        return False

def load_nhanes_data():
    datasets = {}

    print(f"Downloading and processing {len(REQUIRED_FILES)} NHANES files...")

    for component, file_code in REQUIRED_FILES:
        print(f"\nProcessing {component} ({file_code})...")

        xpt_file = os.path.join(RAW_DATA_DIR, f"{file_code}.XPT")
        csv_file = os.path.join(PROCESSED_DATA_DIR, f"{file_code}.csv")

        if os.path.exists(csv_file):
            print(f"CSV file already exists: {csv_file}")
            datasets[file_code] = pd.read_csv(csv_file)
            continue

        if not os.path.exists(xpt_file):
            file_url = search_nhanes_files(file_code)

            if file_url:
                print(f"Downloading {file_code} from {file_url}")
                success = download_file(file_url, xpt_file)

                if not success:
                    print(f"Failed to download {file_code}")
                    continue
            else:
                print(f"Could not find download URL for {file_code}")
                continue

        if not os.path.exists(csv_file):
            success = xpt_to_csv(xpt_file, csv_file)

            if not success:
                continue

        datasets[file_code] = pd.read_csv(csv_file)
        print(f"Loaded {file_code} with {len(datasets[file_code])} rows and {len(datasets[file_code].columns)} columns")

    print("\nData loading complete!")
    return datasets

def check_datasets_integrity(datasets):
    participant_ids = None

    for file_code, df in datasets.items():
        if 'SEQN' not in df.columns:
            print(f"Warning: {file_code} does not have SEQN column")
            continue

        current_ids = set(df['SEQN'])

        if participant_ids is None:
            participant_ids = current_ids
        else:
            common_ids = participant_ids.intersection(current_ids)
            print(f"Common participants between previous datasets and {file_code}: {len(common_ids)}")
            participant_ids = common_ids

    if participant_ids:
        print(f"\nFound {len(participant_ids)} participants with data in all datasets")
    else:
        print("No common participants found across all datasets")

    return participant_ids

def save_datasets_summary(datasets):
    summary_file = os.path.join(BASE_DIR, "datasets_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("NHANES Datasets Summary\n")
        f.write("======================\n\n")

        for file_code, df in datasets.items():
            f.write(f"{file_code}:\n")
            f.write(f"  Rows: {len(df)}\n")
            f.write(f"  Columns: {len(df.columns)}\n")

            if 'SEQN' in df.columns:
                f.write(f"  Unique participants: {df['SEQN'].nunique()}\n")

            f.write("\n")

    print(f"Datasets summary saved to {summary_file}")

if __name__ == "__main__":
    datasets = load_nhanes_data()
    common_participants = check_datasets_integrity(datasets)
    save_datasets_summary(datasets)
    print("\nNHANES data ingestion completed successfully!")
