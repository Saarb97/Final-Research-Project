import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
import spacy
from gensim import corpora, models
from gensim.utils import simple_preprocess

'''
Function to load the prompt text from csv and extract only the prompts that had wrong response.
Returns dataframe
'''
def load_csv(path):
    try:
        df = pd.read_csv(path)
        print(df.head())
        df = df[['id', 'text', 'cluster']]
        return df
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(str(e))


def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def calculate_readability(text):
    return textstat.flesch_reading_ease(text)


def calculate_syntactic_complexity(text):
    doc = nlp(text)
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    return avg_sentence_length


def calculate_lexical_diversity(text):
    tokens = text.split()
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0


def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def preprocess_text_for_lda(text):
    return simple_preprocess(text, deacc=True)


def assign_topic_lda(text):
    bow = dictionary.doc2bow(text)
    topics = lda_model.get_document_topics(bow)
    # Sort topics by probability and return the topic with the highest probability
    return sorted(topics, key=lambda x: x[1], reverse=True)[0][0]


if __name__ == '__main__':
    FILE_PATH = "fairness_bbq_dataset_with_embeddings.csv"
    #df = load_csv(FILE_PATH)
    df = pd.read_csv(FILE_PATH)


    # Load spaCy's language model
    nlp = spacy.load("en_core_web_sm")
    print(f'started calculating features')
    df[['polarity', 'subjectivity']] = df['text'].apply(lambda x: calculate_sentiment(x)).apply(pd.Series)
    df['readability_score'] = df['text'].apply(lambda x: calculate_readability(x))
    df['syntactic_complexity'] = df['text'].apply(lambda x: calculate_syntactic_complexity(x))
    df['lexical_diversity'] = df['text'].apply(lambda x: calculate_lexical_diversity(x))
    df['text_length'] = df['text'].apply(len)
    df['named_entities'] = df['text'].apply(lambda x: extract_entities(x))
    print(f'preprocessing text for lda')
    # Apply preprocessing to the DataFrame
    df['processed_LDA_text'] = df['text'].apply(preprocess_text_for_lda)
    print(f'Create a dictionary and corpus for LDA')
    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(df['processed_LDA_text'])
    corpus = [dictionary.doc2bow(text) for text in df['processed_LDA_text']]
    print(f'Train the LDA model')
    # Train the LDA model
    lda_model = models.LdaMulticore(corpus, num_topics=100, id2word=dictionary, passes=50)
    print(f'classifying topics')
    df['topic'] = df['processed_LDA_text'].apply(assign_topic_lda)

    df.to_csv('feature_extraction_text2.csv')

