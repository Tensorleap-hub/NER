from typing import List, Union
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

from NER.ner import tokenizer
from leap_binder import preprocess_func, decode_text


def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        key = f"Topic #{topic_idx}"
        val = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        top_words[key] = val
        print(key, " ", val)
    return top_words


def label_topic(text: Union[str, List[str]], lda, vectorizer):
    top_words = get_top_words(lda, vectorizer.get_feature_names_out(), 20)
    topic_names = [
        "Global Politics and Security",
        "Financial and Economic News",
        "Sports and Entertainment",
        "Sports Competitions and Events",
        "Sports Results and Competitions",
        "Public Safety and Urban Affairs",
        "International News and Events"
    ]
    topics = list(top_words.keys())
    data_vectorized = vectorizer.transform([text] if isinstance(text, str) else text)
    data_topics = np.argmax(lda.transform(data_vectorized), axis=1)

    res = {
        'topic_number': data_topics.tolist(),
        'topic_name': [topic_names[topic_num] for topic_num in data_topics],
        'top_words': [top_words[topics[topic_num]] for topic_num in data_topics],
    }
    if len(data_topics) == 1:
        for k, v in res.items():
            res[k] = v[0]
    return res

def build_corpus_from_tl_prep():
    train, val, test = preprocess_func()
    corpus = []
    idx = []
    for sub, val in zip([train, val, test], ['train', 'val', 'test']):
        for i in range(0, sub.length):
            text_data = decode_text(i, sub)
            corpus.append(text_data)
            idx.append((val, idx))
    return corpus, idx

def fit_LDA():
    corpus, idx = build_corpus_from_tl_prep()
    # Vectorize text
    stop_words = list(ENGLISH_STOP_WORDS) + ['said', 'told', 'did', 'saying']
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=10, max_df=0.5)
    data_vectorized = vectorizer.fit_transform(corpus)

    # Apply LDA
    n_components = 7
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0, doc_topic_prior=0.1, topic_word_prior=0.01)
    lda.fit(data_vectorized)

    top_words = get_top_words(lda, vectorizer.get_feature_names_out(), 20)
    topics = list(top_words.keys())
    data_topics = np.argmax(lda.transform(data_vectorized), axis=1)

    topic_names = [
        "Global Politics and Security",
        "Financial and Economic News",
        "Sports and Entertainment",
        "Sports Competitions and Events",
        "Sports Results and Competitions",
        "Public Safety and Urban Affairs",
        "International News and Events"
    ]
    # Create a DataFrame with the document, its topic, and the top words of that topic
    df = pd.DataFrame({
        'Document': corpus,
        'Topic Number': data_topics,
        'Topic Name': [topic_names[topic_num] for topic_num in data_topics],
        'Top Words': [top_words[topics[topic_num]] for topic_num in data_topics]
    })
    df['Topic Name'] = df['Topic Number'].map(dict(enumerate(topic_names)))

    return lda, vectorizer, df


if __name__ == "__main__":

    lda, vectorizer, df = fit_LDA()

    # Save the tokenizer and LDA model
    dump(vectorizer, 'assets/vectorizer.joblib')
    dump(lda, 'assets/lda_model.joblib')

    # Labeled based on topic representation


    # from bertopic import BERTopic
    # topic_model = BERTopic.load("davanstrien/chat_topics")
    # topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
    # topic_model = BERTopic.load("jaimevera1107/moderation-topics")
    # topics = topic_model.get_topic_info()
    #
    # data = corpus[0:20]
    # topic, prob = topic_model.transform(data)
    #
    # df = topics.iloc[topic+1][['Name', 'Representation']]
    # df["text"] = data
    # df["prob"] = prob

