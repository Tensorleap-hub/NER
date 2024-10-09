from typing import List, Union
# import spacy
from joblib import load
import os
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from NER.utils.ner import _is_entity, _tag_to_entity_type
from NER.utils.gcs_utils import _download

from tl.visualizers import *
from tl.tl_utils import decode_text
from NER.config import CONFIG
from NER.ner import tokenizer, map_idx_to_label


# nlp = spacy.load("en_core_web_sm")

def count_instances(int_tags):
    cats_cnt = {c: 0 for c in CONFIG["categories"][1:]}
    labels = [map_idx_to_label[i] for i in int_tags]
    for l in labels:
        if 'B' in l:
            cats_cnt[_tag_to_entity_type(l)] += 1
    return cats_cnt


def calc_instances_avg_len(int_tags):
    cats_cnt = count_instances(int_tags)
    cats_tokens_cnt = {c: 0 for c in CONFIG["categories"][1:]}
    labels = [map_idx_to_label[i] for i in int_tags]
    for l in labels:
        if l != CONFIG["categories"][0]:        # not 'O'
            cats_tokens_cnt[_tag_to_entity_type(l)] += 1     # count category tokens
    # divide eahch tokens count per instances count
    for k, v in cats_tokens_cnt.items():
        n = cats_cnt[k]
        cats_tokens_cnt[k] = v/(n if n > 0 else 1)
    return cats_tokens_cnt


def count_oov(tokens, int_tags):
    oov_tokens_cnt = {c: 0 for c in CONFIG["categories"][1:]}
    oov_tokens_cnt['total'] = 0
    labels = [map_idx_to_label[i] for i in int_tags]
    oov_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for i, token in enumerate(token_ids):
        if token == oov_id:
            oov_tokens_cnt['total'] += 1
            if labels[i] != CONFIG["categories"][0]:        # not 'O'
                oov_tokens_cnt[_tag_to_entity_type(labels[i])] += 1  # count entity OOV category tokens
    return oov_tokens_cnt


def string_formatting(tokens, int_tags):
    tokens_cnt = {f"{c}_{c_case}": 0 for c in CONFIG["categories"][1:]+["total"] for c_case in ["lower", "upper", "capitalize"]}
    tags = [map_idx_to_label[i] for i in int_tags]
    for i, tag in enumerate(tags):
        token = tokens[i]
        key = ""
        if token.istitle():
            key = "capitalize"
        elif token.islower():
            key = "lower"
        else: #elif token.isupper():
            key = "upper"

        if _is_entity(tags[i]):        # check if Entity label
            cat = _tag_to_entity_type(tags[i])
            tokens_cnt[cat + f"_{key}"] += 1

        tokens_cnt["total" + f"_{key}"] += 1        # update count of all tokens

    tokens_cnt_prec = {}
    # add relative counts as well
    length = max(len(tags), 1)
    for k, v in tokens_cnt.items():
        tokens_cnt_prec[k + "_percentage"] = v / length
    tokens_cnt.update(tokens_cnt_prec)
    return tokens_cnt


# Get the LDA and vectorizer
# dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NER', 'utils', 'assets'))
# Load the LDA model from disk
cloud_fpath = os.path.join(CONFIG['gcs_data_dir'], 'vectorizer.joblib')
fpath = _download(cloud_fpath)
vectorizer = load(fpath)

cloud_fpath = os.path.join(CONFIG['gcs_data_dir'], 'lda_model.joblib')
fpath = _download(cloud_fpath)
lda = load(fpath)

def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        key = f"Topic #{topic_idx}"
        val = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        top_words[key] = val
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
def get_sample_topic(i: int, subset: PreprocessResponse) -> dict:
    text = decode_text(i, subset)
    res = label_topic(text, lda, vectorizer)
    return res



# Metadata functions allow to add extra data for a later use in analysis.
@tensorleap_metadata(name="metadata_dic")
def metadata_dic(idx: int, preprocess: PreprocessResponse) -> int:
    metadata_dic = {}
    metadata_dic["index"] = idx
    tags = preprocess.data['ds'][idx]['ner_tags']
    tokens = preprocess.data['ds'][idx]['tokens']
    # Length of text
    metadata_dic['txt_length'] = len(tags)

    n = max(metadata_dic['txt_length'], 1)

    # count instances
    res = count_instances(tags)
    for k, v in res.items():
        metadata_dic[k+"_inst_cnt"] = v
        metadata_dic[k+"_inst_percentage"] = v/n        # %

    # Avg entities length and %
    res = calc_instances_avg_len(tags)
    for k, v in res.items():
        metadata_dic[k+"_avg_len"] = v
        metadata_dic[k+"_avg_len_percentage"] = v/n         # %

    # Calc total OOV tokens and OOV per entity type
    res = count_oov(tokens, tags)
    for k, v in res.items():
        metadata_dic[k+"_oov_cnt"] = v
        metadata_dic[k+"_oov_percentage"] = v/n         # %
    # Entity capitalized
    res = string_formatting(tokens, tags)
    metadata_dic.update(res)

    res = get_sample_topic(idx, preprocess)
    metadata_dic.update(res)

    # if preprocess.data['subset']== 'ul':
    #     #TODO
    # else:       # Labeled
    return metadata_dic




