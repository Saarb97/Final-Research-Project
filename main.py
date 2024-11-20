import create_summarized_table
import feature_extraction
import result_analysis
import xgboost_clusters
from create_summarized_table import *
from calc_gen_ai_features import *
import time
import os
import sys
import subprocess
from xgboost_clusters import *

def _ensure_spacy_model():
    try:
        # Try importing the spacy model to see if it's already installed
        import spacy
        spacy.load("en_core_web_sm")
        print("Spacy's 'en_core_web_sm' model loaded successfully.")
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
        return True
    else:
        while True:
            response = input(f"The folder '{folder_path}' does not exist. Do you want to create it? (Y/N): ").strip().lower()
            if response == 'y':
                try:
                    os.makedirs(folder_path)
                    print(f"Folder '{folder_path}' has been created successfully.")
                    return True
                except Exception as e:
                    print(f"An error occurred while creating the folder: {e}")
                    return False
            elif response == 'n':
                print("Folder creation cancelled.")
                return False
            else:
                print("Invalid response. Please reply with 'Y' or 'N'.")

def main():
    # Runs only on scipy==1.12 because of gensim requirement of deprecated function
    # if missing 'en_core_web_sm' -  python -m spacy download en_core_web_sm
    _ensure_spacy_model()

    clusters_files_loc = 'testfolder2'
    xgboost_files_loc = os.path.join(clusters_files_loc, 'xgboost_files')
    results_files_loc = os.path.join(clusters_files_loc, 'results')


    # If destination folder for files doesn't exist / cannot be created / user chose to abort.
    if not _check_and_create_folder(clusters_files_loc):
        sys.exit()

    if not _check_and_create_folder(xgboost_files_loc):
        sys.exit()

    if not _check_and_create_folder(results_files_loc):
        sys.exit()

    ai_features_loc = 'clustered_ai_features.csv'
    df = load_data('all_clustering_09_05.csv')
    start_whole = time.time()
    start = time.time()
    df_with_features = feature_extraction.generic_feature_extraction(df, 'text')
    df_with_features.to_csv(os.path.join(clusters_files_loc, 'test_feature_extraction_file.csv'), index=False)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for basic feature extraction: {elapsed} seconds')

    start = time.time()
    num_of_clusters = create_summarized_table.split_clusters_data(df_with_features, clusters_files_loc)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for cluster splitting clusters: {elapsed} seconds')

    start = time.time()
    deberta_for_llm_features(ai_features_loc, clusters_files_loc)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for running DeBERTa on LLM features: {elapsed} seconds')

    start = time.time()
    create_summarized_table.create_summarized_tables(df_with_features, 'text', 'performance', clusters_files_loc, num_of_clusters)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for cluster creating summarized tables for clusters: {elapsed} seconds')

    start = time.time()
    xgboost_clusters.main_kfold(clusters_files_loc, num_of_clusters, xgboost_files_loc)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for running XGBoost models on each cluster: {elapsed} seconds')

    start = time.time()
    result_analysis.analyse_results(clusters_files_loc, num_of_clusters, results_files_loc)
    end = time.time()
    elapsed = round(end - start)
    print(f'time for running analysis on the results of each cluster: {elapsed} seconds')

    end = time.time()
    elapsed = round(end - start_whole)
    print(f'time for whole process: {elapsed} seconds')

if __name__ == '__main__':
    main()

