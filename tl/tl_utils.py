
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