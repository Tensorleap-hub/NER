# import spacy
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from NER.utils.ner import _is_entity, _tag_to_entity_type
from tl.metadata_helpers import *
from tl.visualizers import *



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

    # Language
    # TODO
    # Get POS use it also
    # TODO

    # if preprocess.data['subset']== 'ul':
    #     #TODO
    # else:       # Labeled
    return metadata_dic


# def detect_language(text):
#     doc = nlp(text)
#     return doc.lang_