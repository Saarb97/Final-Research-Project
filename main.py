import create_summarized_table
import feature_extraction
import xgboost_clusters
from selection_and_clustering import *
from feature_extraction import *
from create_summarized_table import *
from calc_gen_ai_features import *
import time
import os
import subprocess
from xgboost_clusters import *

def _ensure_spacy_model():
    try:
        # Try importing the spacy model to see if it's already installed
        import spacy
        spacy.load("en_core_web_sm")
        print("The 'en_core_web_sm' model is already installed.")
    except (ImportError, OSError):
        # If not installed, run the command to download it
        print("The 'en_core_web_sm' model is not installed. Installing it now...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("The 'en_core_web_sm' model has been installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while installing the 'en_core_web_sm' model: {e}")

def _check_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"The folder '{folder_path}' already exists.")
    else:
        while True:
            response = input(f"The folder '{folder_path}' does not exist. Do you want to create it? (Y/N): ").strip().lower()
            if response == 'y':
                try:
                    os.makedirs(folder_path)
                    print(f"Folder '{folder_path}' has been created successfully.")
                    break
                except Exception as e:
                    print(f"An error occurred while creating the folder: {e}")
                    break
            elif response == 'n':
                print("Folder creation cancelled.")
                break
            else:
                print("Invalid response. Please reply with 'Y' or 'N'.")

def main():
    # Runs only on scipy==1.12 because of gensim requirement of deprecated function
    # if missing 'en_core_web_sm' -  python -m spacy download en_core_web_sm
    _ensure_spacy_model()

    destination = 'testfolder'
    ai_features_loc = 'clustered_ai_features'
    df = load_data('all_clustering_09_05.csv')
    start_whole = time.time()
    start = time.time()
    df_with_features = feature_extraction.generic_feature_extraction(df, 'text')
    df_with_features.to_csv(os.path.join(destination, 'test_feature_extraction_file.csv'), index=False)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for basic feature extraction: {elapsed} seconds')

    start = time.time()
    num_of_clusters = create_summarized_table.split_clusters_data(df_with_features, destination)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for cluster splitting clusters: {elapsed} seconds')

    start = time.time()
    deberta_for_llm_features(ai_features_loc, destination)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for running DeBERTa on LLM features: {elapsed} seconds')

    start = time.time()
    create_summarized_table.create_summarized_tables(df_with_features, 'text', 'performance', destination, num_of_clusters)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for cluster creating summarized tables for clusters: {elapsed} seconds')

    num_of_clusters = 20
    start = time.time()
    xgboost_clusters.main_kfold(destination, num_of_clusters, destination)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for running XGBoost models on each cluster: {elapsed} seconds')

if __name__ == '__main__':
    main()

