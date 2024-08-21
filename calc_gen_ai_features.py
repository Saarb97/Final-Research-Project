import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import time
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load SBERT model
#model = SentenceTransformer('all-mpnet-base-v2')
# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

# Local bert
#nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
#nli_model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

#torch.backends.cuda.matmul.allow_tf32 = True


nli_model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    #quantization_config=bnb_config,
    # torch_dtype=torch.float16,
)

nli_model.to(device)

#tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

def encode_text(text):
    """Encode text using the SBERT model."""
    return model.encode(text, convert_to_tensor=True)


def compute_cosine_similarity(text_embeddings, concept_embeddings):
    """Compute cosine similarity between text and concept embeddings."""
    return util.pytorch_cos_sim(text_embeddings, concept_embeddings)


def compute_best_scores(cosine_scores, ai_features):
    """Compute the best scores for each concept word for each text."""
    best_scores = []
    for scores in cosine_scores:
        # print("scores")
        # print(scores)
        best_score = {ai_features[i]: scores[i].item() for i in range(len(ai_features))}
        # print("best_score")
        # print(best_score)
        best_scores.append(best_score)
    return best_scores


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
    #result_df.to_csv(file_name, index=False)


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


def compute_best_scores_dimension_wise(text_embeddings, concept_embeddings, ai_features):
    """Compute the best scores for each concept word for each text using dimension-wise cosine similarity."""
    best_scores = []
    for text_embedding in text_embeddings:
        best_similarities = compute_dimension_wise_cosine_similarity(text_embedding, concept_embeddings)
        best_score = {ai_features[i]: best_similarities[i] for i in range(len(ai_features))}
        best_scores.append(best_score)
    return best_scores


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


def process_file_with_classification(file_index, ai_features):
    """Process a single file, compute scores, and concatenate results."""
    file_name = f'clusters csv/{file_index}_data.csv'
    data = pd.read_csv(file_name)
    cluster_text = data['text']

    # Compute the best scores for each text against the concept words
    # scores_list = [classify_sentence(text, ai_features) for text in cluster_text]

    scores_list = []
    for count, text in enumerate(cluster_text):
        print(f'Processing {count+1}/{len(cluster_text)}')
        start = time.time()
        probabilities = compute_probabilities(text, ai_features)
        # print(f'probabilities: {probabilities}')
        scores_list.append(probabilities)
        end = time.time()
        elapsed = end - start
        print(f'elapsed time: {elapsed}')

    #print(f'score 1: {scores_list[0]}')
    # Convert the list of scores to a DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Combine the scores with the original dataframe
    result_df = pd.concat([data, scores_df], axis=1)
    #print(f'results: {result_df.head}')
    # Save the result back to the original CSV file
    result_df.to_csv(file_name, index=False)


def compute_probabilities(sentence, hypotheses):
    """
    Computes probabilities for a sentence and a list of hypotheses.

    Args:
    sentence (str): The premise sentence.
    hypotheses (list): A list of hypothesis strings.

    Returns:
    dict: A dictionary with hypotheses as keys and their corresponding probabilities as values.
    """
    probabilities = {}
    for hypothesis in hypotheses:
        inputs = tokenizer.encode(sentence, hypothesis, return_tensors='pt', truncation_strategy='only_first').to(
            device)

        logits = nli_model(inputs)[0]

        #entail_contradiction_logits = logits[:, [0, 2]]
        entail_contradiction_logits = logits[:, [0, 1]]

        probs = entail_contradiction_logits.softmax(dim=1)

        prob_hypothesis_true = probs[:, 1].item()

        probabilities[hypothesis] = prob_hypothesis_true

    return probabilities

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    clustered_ai_features = pd.read_csv('ai_features2.csv', encoding='ISO-8859-1')
    cols = clustered_ai_features.columns.tolist()

    for i in range(20):
        cluster_index = str(i)
        if cluster_index in cols:
            ai_features = clustered_ai_features[f'{i}'].dropna().tolist()
            #process_file(i, ai_features)
            #process_file_dimension_wise(i, ai_features)
            process_file_with_classification(i, ai_features)
            print(f"Processing complete for file {i}_data.csv")