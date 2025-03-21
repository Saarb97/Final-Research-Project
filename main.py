import argparse
import re
import subprocess
import sys

import dspy
from dotenv import load_dotenv

import create_summarized_table
import feature_extraction
import llm_api_feature_extraction
import result_analysis
import xgboost_clusters
from calc_gen_ai_features import *
from create_summarized_table import *
from xgboost_clusters import *

load_dotenv()
def clean_text_column(df, text_col_name):
    """Clean the text column by removing rows with empty text while retaining specific symbols (!, ?, .)."""
    # Drop rows with missing or empty text
    start_len = len(df)
    df = df[df[text_col_name].notnull()]  # Remove NaN values
    df = df[df[text_col_name].str.strip().astype(bool)]  # Remove empty strings

    # Clean text by removing unusual symbols, non english text, except !, ?, .
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Remove non-alphanumeric characters except !, ?, .
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()  # Strip leading/trailing spaces

    df[text_col_name] = df[text_col_name].apply(clean_text)

    # Define a function to filter out rows with insufficient content
    def has_valid_content(text):
        # Remove rows that only contain symbols like !, ?, or .
        if re.fullmatch(r'[!?.,\s]*', text):
            return False
        # Removes rows with less than 4 chars
        if len(text.strip()) < 4:
            return False
        return True

    # Apply the filter
    df = df[df[text_col_name].apply(has_valid_content)]
    cur_len = len(df)
    print(f"Clean dataset's text. started with {start_len} rows, after cleaning: {cur_len}")
    return df


def _count_cluster_files(clusters_files_loc):
    # Match files with the pattern "<cluster>_data.csv"
    cluster_files = [
        f for f in os.listdir(clusters_files_loc)
        if os.path.isfile(os.path.join(clusters_files_loc, f)) and f.endswith("_data.csv")
    ]
    return len(cluster_files)


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
            response = input(
                f"The folder '{folder_path}' does not exist. Do you want to create it? (Y/N): ").strip().lower()
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


def main(openai_api_key, text_col_name='text'):
    # Runs only on scipy==1.12 because of gensim requirement of deprecated function
    VALID_STEPS = range(1, 7)  # Steps 1 to 6

    _ensure_spacy_model()

    clusters_files_loc = os.path.join('sarcasm_dataset', 'clusters')
    xgboost_files_loc = os.path.join(clusters_files_loc, 'xgboost_files')
    results_files_loc = os.path.join(clusters_files_loc, 'results')
    ai_features_loc = 'clustered_ai_features.csv'

    # If destination folder for files doesn't exist / cannot be created / user chose to abort.
    if not _check_and_create_folder(clusters_files_loc):
        sys.exit()

    if not _check_and_create_folder(xgboost_files_loc):
        sys.exit()

    if not _check_and_create_folder(results_files_loc):
        sys.exit()

    # Argument parsing for step control
    parser = argparse.ArgumentParser(description="Run pipeline from a specific step.")
    parser.add_argument(
        '--start', type=int, default=1,
        help=f"Step to start from ({VALID_STEPS.start}-{VALID_STEPS.stop - 1}). Default is 1."
    )
    args = parser.parse_args()

    # Validate the `--start` argument
    start_step = args.start
    if start_step not in VALID_STEPS:
        parser.error(f"Invalid value for --start. Must be between {VALID_STEPS.start} and {VALID_STEPS.stop - 1}.")

    # Timing the whole process
    start_whole = time.time()

    # Step 1: Feature extraction
    if start_step <= 1:
        print("Starting from Step 1: Feature Extraction")
        start = time.time()
        df = load_data('all_clustering_09_05.csv')
        # df = load_data(os.path.join('twitter sentiment', 'twitter_clustering24_11_24.csv'))

        # Clean text column
        df = clean_text_column(df, text_col_name)

        df_with_features = feature_extraction.generic_feature_extraction_parallel(df, text_col_name, 4)
        df_with_features.to_csv(os.path.join(clusters_files_loc, 'test_feature_extraction_file.csv'), index=False)
        elapsed = round(time.time() - start)
        print(f'Time for basic feature extraction: {elapsed} seconds')
    else:
        print("Skipping Step 1: Loading previously saved features")
        df_with_features = pd.read_csv(os.path.join(clusters_files_loc, 'test_feature_extraction_file.csv'))

    # Step 2: Cluster splitting
    if start_step <= 2:
        print("Step 2: Cluster Splitting")
        start = time.time()
        num_of_clusters = create_summarized_table.split_clusters_data(df_with_features, clusters_files_loc)
        elapsed = round(time.time() - start)
        print(f'Time for cluster splitting: {elapsed} seconds')
    else:
        print("Skipping Step 2: Using precomputed clusters")
        num_of_clusters = _count_cluster_files(clusters_files_loc)

    # Step 3: extracting LLM features from Open AI's GPT model
    if start_step <= 3:
        print("Step 3: extracting LLM features from Open AI's GPT model")
        start = time.time()
        lm = dspy.LM('openai/gpt-4o-mini',
                     api_key=openai_api_key,
                     cache=False)
        dspy.configure(lm=lm)
        llm_features_pd = (llm_api_feature_extraction.
                           llm_feature_extraction_for_clusters_folder_dspy(lm, clusters_files_loc, text_col_name,
                                                                           model="gpt-4o-mini"))

        ai_features_file_name = os.path.join(clusters_files_loc, 'llm_features_per_cluster.csv')
        llm_features_pd.to_csv(ai_features_file_name, index=False)
        elapsed = round(time.time() - start)
        print(f'Time for running DeBERTa on LLM features: {elapsed} seconds')
    else:
        print("Skipping Step 3: Assuming llm features file already exists.")
        ai_features_file_name = os.path.join(clusters_files_loc, 'ai_features_per_cluster.csv')

    # Step 4: DeBERTa for LLM features
    if start_step <= 4:
        print("Step 3: DeBERTa for LLM features")
        start = time.time()
        deberta_for_llm_features(ai_features_file_name, clusters_files_loc)
        elapsed = round(time.time() - start)
        print(f'Time for running DeBERTa on LLM features: {elapsed} seconds')

    # Step 5: Summarized tables creation
    if start_step <= 5:
        print("Step 4: Summarized Tables Creation")
        start = time.time()
        create_summarized_table.create_summarized_tables(df_with_features, 'text', 'performance', clusters_files_loc,
                                                         num_of_clusters)
        elapsed = round(time.time() - start)
        print(f'Time for summarized table creation: {elapsed} seconds')

    # Step 6: XGBoost models
    if start_step <= 6:
        print("Step 5: XGBoost Models")
        start = time.time()
        xgboost_clusters.main_kfold(clusters_files_loc, num_of_clusters, xgboost_files_loc)
        elapsed = round(time.time() - start)
        print(f'Time for running XGBoost models: {elapsed} seconds')

    # Step 7: Result analysis
    if start_step <= 7:
        print("Step 6: Result Analysis")
        start = time.time()
        result_analysis.analyse_results(clusters_files_loc, num_of_clusters, results_files_loc)
        elapsed = round(time.time() - start)
        print(f'Time for result analysis: {elapsed} seconds')

    elapsed = round(time.time() - start_whole)
    print(f'time for whole process: {elapsed} seconds')


if __name__ == '__main__':
    api_key = os.getenv('OPENAI_KEY')
    main(api_key, text_col_name='text')

    # df = pd.read_csv('twitter sentiment/raw/twitter_training.csv')
    # print(f'len before processing: {len(df)}')
    # df = clean_text_column(df, text_col_name='text')
    # print(f'len after processing: {len(df)}')
    # df['text'] = df['topic'].astype(str) + ' ' + df['text'].astype(str)
    # df.to_csv('twitter sentiment/processed/twitter_training.csv', index=False)
    #
    # df = pd.read_csv('twitter sentiment/raw/twitter_validation.csv')
    # print(f'len before processing: {len(df)}')
    # df = clean_text_column(df, text_col_name='text')
    # print(f'len after processing: {len(df)}')
    # df['text'] = df['topic'].astype(str) + ' ' + df['text'].astype(str)
    # df.to_csv('twitter sentiment/processed/twitter_validation.csv', index=False)
