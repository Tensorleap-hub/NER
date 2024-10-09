from code_loader.contract.datasetclasses import PreprocessResponse

from NER.ner import tokenizer
def mark_start_of_instance(text_tokens, labels_names):
    new_text_tokens, new_labels_names = [], []
    for i, (token, label) in enumerate(zip(text_tokens, labels_names)):
        if "B" in label:
            # Add marker '-' before the token and label 'B-' before the label if label is a beginning of instance
            new_text_tokens.append('<S>')     # add sep token
            new_labels_names.append('-B')   # add B label

        # Add the original token and label
        new_text_tokens.append(token)
        new_labels_names.append(label)

    return new_text_tokens, new_labels_names


def decode_text(i: int, subset: PreprocessResponse):
    tokens = subset.data['ds'][i]
    tokenized_inputs = tokenizer(tokens['tokens'], truncation=True, is_split_into_words=True)['input_ids']
    text_data = tokenizer.decode(tokenized_inputs, skip_special_tokens=True)
    return text_data

