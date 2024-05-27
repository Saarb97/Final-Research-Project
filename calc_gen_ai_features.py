import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Load SBERT model
model = SentenceTransformer('all-mpnet-base-v2')

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
        best_score = {ai_features[i]: scores[i].item() for i in range(len(ai_features))}
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



if __name__ == '__main__':
    clustered_ai_features = pd.read_csv('clustered_ai_features.csv')
    cols = clustered_ai_features.columns.tolist()

    for i in range(1):
        cluster_index = str(i)
        if cluster_index in cols:
            ai_features = clustered_ai_features[f'{i}'].dropna().tolist()
            process_file(i, ai_features)
            print(f"Processing complete for file {i}_data.csv")
