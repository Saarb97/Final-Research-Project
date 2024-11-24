import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import time
import torch
import os
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import Pool, set_start_method
from itertools import chain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

#
nli_model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    # quantization_config=bnb_config,
    # torch_dtype=torch.float16,
)

nli_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

def collate_fn(batch):
    sentences, hypotheses = zip(*batch)
    return list(sentences), hypotheses

# Old experiment - Load SBERT model
#model = SentenceTransformer('all-mpnet-base-v2')

# Old experiment using sentence-bert.
def encode_text(text):
    """Encode text using the SBERT model."""
    return model.encode(text, convert_to_tensor=True)

# Old cosine similarity experiment - not used
def compute_cosine_similarity(text_embeddings, concept_embeddings):
    """Compute cosine similarity between text and concept embeddings."""
    return util.pytorch_cos_sim(text_embeddings, concept_embeddings)

# Old cosine similarity experiment - not used
def compute_best_scores(cosine_scores, ai_features):
    """Compute the best scores for each concept word for each text."""
    best_scores = []
    for scores in cosine_scores:
        best_score = {ai_features[i]: scores[i].item() for i in range(len(ai_features))}
        best_scores.append(best_score)
    return best_scores


# Old cosine similarity experiment - not used
def process_file(file_index, ai_features):
    """Process a single file, compute scores, and concatenate results."""
    file_name = f'clusters csv/{file_index}_data.csv'
    data = pd.read_csv(file_name)
    cluster_text = data['text']

    # Encode concept words and text data
    ai_features_embeddings = encode_text(ai_features)
    text_embeddings = encode_text(cluster_text.tolist())

    # Compute cosine similarity
    cosine_scores = compute_cosine_similarity(text_embeddings, ai_features_embeddings)
    # Compute the best scores
    scores_list = compute_best_scores(cosine_scores, ai_features)

    # Convert the list of scores to a DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Combine the scores with the original dataframe
    result_df = pd.concat([data, scores_df], axis=1)

    # Save the result back to the original CSV file
    # result_df.to_csv(file_name, index=False)


# Cosine similarity experiment - not used
def compute_dimension_wise_cosine_similarity(text_embedding, concept_embeddings):
    """Compute the best dimension-wise cosine similarity for a text embedding and each concept embedding."""
    best_similarities = []
    text_embedding = text_embedding.squeeze()
    concept_embeddings = concept_embeddings.squeeze()

    for concept_embedding in concept_embeddings:
        # Calculate cosine similarity for each dimension
        similarities = [
            (text_embedding[i] * concept_embedding[i]) /
            (np.linalg.norm(text_embedding[i]) * np.linalg.norm(concept_embedding[i]))
            for i in range(len(text_embedding))
        ]
        # Take the highest similarity score
        best_similarity = max(similarities)
        best_similarities.append(best_similarity)

    return best_similarities


# Cosine similarity experiment - not used
def compute_best_scores_dimension_wise(text_embeddings, concept_embeddings, ai_features):
    """Compute the best scores for each concept word for each text using dimension-wise cosine similarity."""
    best_scores = []
    for text_embedding in text_embeddings:
        best_similarities = compute_dimension_wise_cosine_similarity(text_embedding, concept_embeddings)
        best_score = {ai_features[i]: best_similarities[i] for i in range(len(ai_features))}
        best_scores.append(best_score)
    return best_scores


# Cosine similarity experiment - not used
def process_file_dimension_wise(file_index, ai_features):
    """Process a single file, compute scores using dimension-wise cosine similarity, and concatenate results."""
    file_name = f'clusters csv/{file_index}_data.csv'
    data = pd.read_csv(file_name)
    cluster_text = data['text']

    # Encode concept words and text data
    ai_features_embeddings = encode_text(ai_features)
    text_embeddings = encode_text(cluster_text.tolist())

    # Compute the best scores using the new method
    scores_list = compute_best_scores_dimension_wise(text_embeddings, ai_features_embeddings, ai_features)

    # Convert the list of scores to a DataFrame
    scores_df = pd.DataFrame(scores_list)
    print(scores_df)
    # Combine the scores with the original dataframe
    result_df = pd.concat([data, scores_df], axis=1)

    # Save the result back to the original CSV file
    result_df.to_csv('bert_test.csv', index=False)


# Legacy code
def classify_sentence(sentence, ai_features):
    """
    Classifies the sentence against a list of concepts using zero-shot classification.

    Args:
    sentence (str): The sentence to be classified.
    concepts (list): A list of concept words or phrases.

    Returns:
    dict: A dictionary with concept words as keys and their corresponding scores as values.
    """
    print(f"Classifying sentence {sentence}")
    results = classifier(sentence, ai_features, multi_label=True)
    print(f'Results: {results}')
    concept_scores = {feature: score for feature, score in zip(results['labels'], results['scores'])}
    return concept_scores


class TextDataset(Dataset):
    def __init__(self, sentences, hypotheses):
        self.sentences = sentences
        self.hypotheses = hypotheses

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.hypotheses


# Parallel File Processing
def process_single_file(args):
    return _process_file_with_classification(*args)


def parallel_process_files(file_indices, ai_features, data_files_location):
    """Parallelize processing of multiple files."""

    for file_index in file_indices:
        _process_file_with_classification(file_index, ai_features, data_files_location)

    # pool = Pool(processes=min(cpu_count(), 2))  # Use 2 CPU cores
    # tasks = [(index, ai_features, data_files_location) for index in file_indices]
    # pool.map(process_single_file, tasks)
    # pool.close()
    # pool.join()

def _process_file_with_classification(file_index, ai_features, data_files_location):
    """Process a single file, compute scores, and save results."""
    file_name = os.path.join(data_files_location, f'{file_index}_data.csv')
    data = pd.read_csv(file_name)
    cluster_text = data['text'].fillna('').astype(str).tolist() # TODO update 'text' to generic term

    start = time.time()
    # Create DataLoader for batching
    dataset = TextDataset(cluster_text, ai_features)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    scores_list = []
    for batch_sentences, batch_hypotheses in dataloader:
        # print(f'Processing batch of size {len(batch_sentences)} in cluster {file_index}')
        probabilities = _compute_probabilities(batch_sentences, batch_hypotheses)
        scores_list.extend(probabilities)

    # Convert the list of scores to a DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Combine the scores with the original dataframe
    result_df = pd.concat([data, scores_df], axis=1)
    # Save the result back to the original CSV file
    result_df.to_csv(file_name, index=False)
    end = time.time()
    print(f'Finished processing cluster {file_index} in {end - start:.2f}s')


def _compute_probabilities(sentences, hypotheses):
    """
    Computes probabilities for a batch of sentences and hypotheses.
    """
    # Ensure hypotheses is a flat list of strings
    if isinstance(hypotheses[0], list):
        from itertools import chain
        hypotheses = list(chain.from_iterable(hypotheses))

    if not all(isinstance(hypothesis, str) for hypothesis in hypotheses):
        raise ValueError(f"Invalid hypotheses structure: {hypotheses}")

    # Tokenize in batch
    try:
        inputs = tokenizer(
            [sentence for sentence in sentences for _ in hypotheses],
            hypotheses * len(sentences),
            return_tensors='pt',
            truncation=True,
            padding=True
        ).to(device)
    except Exception as e:
        print("Error during tokenization.")
        print("Sentences:", sentences)
        print("Hypotheses:", hypotheses)
        raise e

    # Perform inference
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # Extract probabilities
    entail_contradiction_logits = logits[:, [0, 1]]
    probs = entail_contradiction_logits.softmax(dim=1)
    probs_hypothesis_true = probs[:, 1].view(len(sentences), -1).tolist()

    probabilities = []
    for i, sentence in enumerate(sentences):
        probabilities.append(dict(zip(hypotheses, probs_hypothesis_true[i])))

    return probabilities



def _check_ai_features_file(ai_features_file_location: str):
    try:
        # clustered_ai_features = pd.read_csv(ai_features_file_location, encoding='ISO-8859-1')
        clustered_ai_features = pd.read_csv(ai_features_file_location)
        # Validate that all column headers are integers
        cols = clustered_ai_features.columns.tolist()
        for col in cols:
            int(col)  # Will raise ValueError if conversion fails

        return clustered_ai_features, cols

    except FileNotFoundError:
        raise FileNotFoundError("Error: File not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Error: The file is empty.")
    except pd.errors.ParserError:
        raise pd.errors.ParserError("Error: There was an issue parsing the file. Please check the file format.")
    except ValueError:
        raise ValueError("Error: One of the column names is not an integer representing cluster number.")
    except AttributeError:
        raise AttributeError("Error: Problem with the DataFrame structure.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def deberta_for_llm_features(ai_features_file_location: str, data_files_location: str) -> None:
    set_start_method("spawn", force=True)
    print('Torch info for running DeBERTa. Run on GPU')
    print(f'Is CUDA available? {torch.cuda.is_available()}')
    print(f'Torch version: {torch.__version__}')
    # print(f'Torch Device: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    try:
        clustered_ai_features, cols = _check_ai_features_file(ai_features_file_location)
        # Prepare tasks for parallel processing
        tasks = []
        for col in cols:
            try:
                start = time.time()
                ai_features = clustered_ai_features[col].dropna().tolist()
                tasks.append((col, ai_features, data_files_location))
            except ValueError:
                # Skip non-numeric columns if they exist
                print(f"non-numeric cluster number in ai features file: {col}, exiting")
                exit(1)

        # Parallelize file processing
        parallel_process_files([task[0] for task in tasks], ai_features, data_files_location)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    print('Torch info for running DeBERTa. Run on GPU')
    print(torch.cuda.is_available())
    print(torch.__version__)
#    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    clustered_ai_features = pd.read_csv('ai_features2.csv', encoding='ISO-8859-1')
    cols = clustered_ai_features.columns.tolist()

    print('start')
    start = time.time()
    ai_features_loc = 'clustered_ai_features.csv'
    destination = 'clusters csv'
    deberta_for_llm_features(ai_features_loc, destination)
    end = time.time()
    print(f'Finished processing all clusters, took {end - start:.2f}s')

    #
    # for col in cols:
    #     # Ensure the column name is numeric (if needed) by converting it to int
    #     try:
    #         cluster_index = int(col)
    #         ai_features = clustered_ai_features[col].dropna().tolist()
    #
    #         # process_file(cluster_index, ai_features)
    #         # process_file_dimension_wise(cluster_index, ai_features)
    #         _process_file_with_classification(cluster_index, ai_features, 'clusters csv')
    #
    #         print(f"Processing complete for file {cluster_index}_data.csv")
    #
    #     except ValueError:
    #         # Skip non-numeric columns if they exist
    #         print(f"Skipping non-numeric column: {col}")
