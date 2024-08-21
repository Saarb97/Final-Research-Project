import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
import spacy
from gensim import corpora, models
from gensim.utils import simple_preprocess
from spacy.lang.en.stop_words import STOP_WORDS
from scipy.stats import chi2_contingency
import readability
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')


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


def assign_topic_lda(text, dictionary, lda_model):
    bow = dictionary.doc2bow(text)
    topics = lda_model.get_document_topics(bow)
    # Sort topics by probability and return the topic with the highest probability
    return sorted(topics, key=lambda x: x[1], reverse=True)[0][0]


def count_question_marks(text):
    return text.count('?')


def count_exclamation_marks(text):
    return text.count('!')


def average_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return np.mean([len(word) for word in words])


def unique_word_count(text):
    return len(set(text.split()))


def sentiment_score_range(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    scores = [sentence.sentiment.polarity for sentence in sentences]
    return max(scores) - min(scores) if scores else 0


def noun_phrase_count(text):
    doc = nlp(text)
    return len(list(doc.noun_chunks))


def verb_phrase_count(text):
    doc = nlp(text)
    verbs = [token for token in doc if token.pos_ == "VERB"]
    return len(verbs)


def named_entity_count(text):
    doc = nlp(text)
    return len(doc.ents)


def passive_voice_count(text):
    doc = nlp(text)
    passive_sentences = 0
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'auxpass':
                passive_sentences += 1
                break
    return passive_sentences


def active_voice_count(text):
    doc = nlp(text)
    active_sentences = 0
    for sent in doc.sents:
        active = False
        for token in sent:
            if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                active = True
        if active:
            active_sentences += 1
    return active_sentences


def modal_verbs_count(text):
    doc = nlp(text)
    modals = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    count = sum(1 for token in doc if token.lemma_ in modals)
    return count


def conditional_sentences_count(text):
    doc = nlp(text)
    conditionals = 0
    for sent in doc.sents:
        if any(token.lemma_ == 'if' for token in sent):
            conditionals += 1
    return conditionals


def sentence_count(text):
    doc = nlp(text)
    return len(list(doc.sents))


def average_sentence_length(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return 0
    return np.mean([len(sentence) for sentence in sentences])


def stop_words_count(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.is_stop)


def punctuation_diversity(text):
    doc = nlp(text)
    punctuations = [token.text for token in doc if token.is_punct]
    return len(set(punctuations))


def analyze_text_readability(text):
    result = readability.getmeasures(text)
    flat_result = {}
    for main_key, sub_dict in result.items():
        for sub_key, value in sub_dict.items():
            flat_result[f"{main_key}__{sub_key}"] = value
    return flat_result


def apply_basic_text_features(df):
    df[['polarity', 'subjectivity']] = df['text'].apply(lambda x: calculate_sentiment(x)).apply(pd.Series)
    df['flesch_reading_ease'] = df['text'].apply(lambda x: calculate_readability(x))
    df['syntactic_complexity'] = df['text'].apply(lambda x: calculate_syntactic_complexity(x))
    df['lexical_diversity'] = df['text'].apply(lambda x: calculate_lexical_diversity(x))
    df['text_length'] = df['text'].apply(len)
    df['question_marks'] = df['text'].apply(count_question_marks)
    df['exclamation_marks'] = df['text'].apply(count_exclamation_marks)
    df['avg_word_length'] = df['text'].apply(average_word_length)
    df['unique_word_count'] = df['text'].apply(unique_word_count)
    df['sentiment_score_range'] = df['text'].apply(sentiment_score_range)
    df['noun_phrase_count'] = df['text'].apply(noun_phrase_count)
    df['verb_phrase_count'] = df['text'].apply(verb_phrase_count)
    df['named_entity_count'] = df['text'].apply(named_entity_count)
    df['passive_voice_count'] = df['text'].apply(passive_voice_count)
    df['active_voice_count'] = df['text'].apply(active_voice_count)
    df['modal_verbs_count'] = df['text'].apply(modal_verbs_count)
    df['conditional_sentences_count'] = df['text'].apply(conditional_sentences_count)
    df['sentence_count'] = df['text'].apply(sentence_count)
    df['average_sentence_length'] = df['text'].apply(average_sentence_length)
    df['stop_words_count'] = df['text'].apply(stop_words_count)
    df['punctuation_diversity'] = df['text'].apply(punctuation_diversity)
    # Adding 35 more features from readability package
    metrics_df = pd.DataFrame(df['text'].apply(analyze_text_readability).tolist())
    df = df.join(metrics_df)
    return df


def apply_LDA(df):
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
    df['topic'] = df['processed_LDA_text'].apply(lambda text: assign_topic_lda(text, dictionary, lda_model))
    df.drop(['processed_LDA_text'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    '''
    TODO: feature value standartization
          check https://doi.org/10.1145/3613904.3641960
                https://github.com/HLasse/TextDescriptives
    
    '''
    FILE_PATH = "all_clustering_09_05.csv"
    df = pd.read_csv(FILE_PATH)

    # Load spaCy's language model 
    nlp = spacy.load("en_core_web_sm")

    df = apply_basic_text_features(df)
    df = apply_LDA(df)
    # df.to_csv('full_dataset_feature_extraction_09-05.csv', Index=False)
    df.to_csv('full_dataset_feature_extraction_09-05.csv')

