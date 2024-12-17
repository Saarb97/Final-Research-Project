import pandas as pd
import tiktoken
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Function to get embeddings
def get_embedding(text, model):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... -> {e}")
        return None

def process_row(row_idx, text, embedding_model):
    embedding = get_embedding(text, model=embedding_model)
    return row_idx, embedding

def get_dataset_embeddings(file_path, client: OpenAI, model="text-embedding-3-small", embedding_encoding="cl100k_base"):
    # API settings
    max_tokens = 8000  # maximum tokens allowed per request

    df = pd.read_csv(file_path)
    encoding = tiktoken.get_encoding(embedding_encoding)

    # Filter the DataFrame
    df = df[['text']]
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]

    # Multithreading settings
    max_requests_per_minute = 3000
    max_threads = 50  # Number of concurrent threads
    time_window = 60  # Time window in seconds

    # Initialize a list for embeddings
    embeddings = [None] * len(df)

    # Process rows using ThreadPoolExecutor
    start_time = time.time()
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(process_row, idx, row['text'], model) for idx, row in df.iterrows()]

        completed_requests = 0
        for future in as_completed(futures):
            row_idx, embedding = future.result()
            embeddings[row_idx] = embedding
            completed_requests += 1
            print(f"Processed row {row_idx + 1}/{len(df)}")

            # Rate limiting to 3000 requests per minute
            elapsed_time = time.time() - start_time
            if completed_requests % max_requests_per_minute == 0 and elapsed_time < time_window:
                sleep_time = time_window - elapsed_time
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                start_time = time.time()

    # Add embeddings to the DataFrame
    df['embedding'] = embeddings
    # the comma seperated values causes bugs when saving as a CSV file
    df['embedding'] = df['embedding'].apply(lambda x: '|'.join(map(str, x)))

    # To revert:
    # df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.split('|'))))

    # Save the resulting DataFrame
    df.to_csv('deepkeep_openai_fairness_test2.csv', index=False)





if __name__ == '__main__':
    text_file_loc = 'clusters csv'
    text_col_name = 'text'

    api_key = ('API-KEY-HERE')
    client = OpenAI(api_key=api_key)
    file_path = 'all_clustering_09_05.csv'

