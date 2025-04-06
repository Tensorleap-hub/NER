from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.patches as patches

# Tensorleap imports
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask, LeapImage
from code_loader.contract.enums import LeapDataType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer

jet = plt.get_cmap("jet")
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

from NER.ner import *
from NER.utils.ner import *
from tl.metadata_helpers import *
from tl.tl_utils import mark_start_of_instance

# TODO: fix take fron the mask vis

@tensorleap_custom_visualizer(name="input_visualizer", visualizer_type=LeapDataType.Text)
def input_visualizer(input_ids: np.ndarray) -> LeapText:
    input_ids = np.squeeze(input_ids)
    input_ids = input_ids[0] if len(input_ids.shape) == 2 else input_ids  # flatten for batch shape
    input_ids = tf.cast(input_ids, dtype=tf.int32)
    text = decode_token_ids(input_ids)
    return LeapText(text)

@tensorleap_custom_visualizer(name="mask_visualizer_gt", visualizer_type=LeapDataType.TextMask)
def text_visualizer_mask_gt(input_ids: np.ndarray, gt_vec_labels: np.ndarray) -> LeapTextMask:
    """ This is mask text visualizer, showing the GT.
     The labels are the classes categories, when we color the beginning of each instance to separate the different instances """

    input_ids = np.squeeze(input_ids)
    gt_vec_labels = np.squeeze(gt_vec_labels)

    # mask by special label -100
    mask = mask_one_hot_labels(gt_vec_labels)
    gt_vec_labels = gt_vec_labels.argmax(-1)  # from one-hot to labels
    gt_vec_labels[~mask] = CONFIG["special_token_id"]

    # To batch
    gt_vec_labels = gt_vec_labels[None, ...]
    gt_vec_labels = postprocess_labels(gt_vec_labels)
    labels_names = gt_vec_labels[0]  # get single sample

    cat_to_int = {c: i for i, c in enumerate(CONFIG["categories"])}
    cat_to_int["B"] = len(CONFIG["categories"])     # give a new color and label to beginning of instance

    # Decode token IDS to text tokens in list
    text_tokens = decode_token_ids(input_ids)

    # We add before each instance "-" text token and label "B"
    text_tokens, labels_names = mark_start_of_instance(text_tokens, labels_names)

    # take the category of each class
    labels_names = [c.split("-")[-1] for c in labels_names]
    mask = np.array([cat_to_int[c] if c != "" else 0 for c in labels_names]).astype(np.uint8)

    return LeapTextMask(text=text_tokens, mask=mask, labels=CONFIG["categories"]+["B"])

@tensorleap_custom_visualizer(name="mask_visualizer_pred", visualizer_type=LeapDataType.TextMask)
def text_visualizer_mask_pred(input_ids: np.ndarray, pred_vec_labels: np.ndarray) -> LeapTextMask:
    """ This is mask text visualizer, showing the Prediction.
     The labels are the classes categories, when we color the beginning of each instance to separate the different instances """
    # To batch
    pred_vec_labels = postprocess_predictions(pred_vec_labels, input_ids)
    # Mask the predictions
    labels_names = pred_vec_labels[0]    # get single sample

    cat_to_int = {c: i for i, c in enumerate(CONFIG["categories"])}
    cat_to_int["B"] = len(CONFIG["categories"])  # give a new color and label to beginning of instance
    # Decode token IDS to text tokens in list
    text_tokens = decode_token_ids(np.squeeze(input_ids))

    # We add before each instance "-" text token and label "B"
    text_tokens, labels_names = mark_start_of_instance(text_tokens, labels_names)

    # take the category of each
    labels_names = [c.split("-")[-1] for c in labels_names]
    mask = np.array([cat_to_int[c] if c != "" else 0 for c in labels_names]).astype(np.uint8)

    return LeapTextMask(text=text_tokens, mask=mask, labels=CONFIG["categories"]+["B"])

@tensorleap_custom_visualizer(name="mask_visualizer_comb", visualizer_type=LeapDataType.TextMask)
def text_visualizer_mask_comb(input_ids: np.ndarray, gt_vec_labels: np.ndarray,
                              pred_vec_labels: np.ndarray) -> LeapTextMask:
    """ This is mask text visualizer, showing the GT and the Prediction together.
     The labels are the classes categories, when we color the beginning of each instance to separate the different instances """

    input_ids = np.squeeze(input_ids)
    gt_vec_labels = np.squeeze(gt_vec_labels)
    pred_vec_labels = np.squeeze(pred_vec_labels)

    gt_vis = text_visualizer_mask_gt(input_ids, gt_vec_labels)
    pred_vis = text_visualizer_mask_pred(input_ids[None, ...], pred_vec_labels[None, ...])

    gt_text, gt_mask, gt_labels = gt_vis.text, gt_vis.mask, gt_vis.labels
    pred_text, pred_mask, pred_labels = pred_vis.text, pred_vis.mask, pred_vis.labels

    # add new line for the prediction tokens
    token_gt_marker = ["GT:\n"]
    token_pred_marker = ["\nPrediction:\n"]
    # merge both text tokens separated by new line
    text = token_gt_marker + gt_text + token_pred_marker + pred_text
    mask = np.concatenate([[0], gt_mask, [0], pred_mask]).astype(dtype=np.uint8)
    return LeapTextMask(text=text, mask=mask, labels=gt_labels)

def plot_vis(vis):

    text_data = vis.text
    mask_data = vis.mask
    labels = vis.labels

    # Create a color map for each label
    colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

    # Adjust figure width dynamically based on the text length
    fig_width = max(10, len(text_data) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title('Leap Text Mask Visualization', color='white')
    ax.axis('off')

    # Set initial position
    x_pos, y_pos = 0.01, 0.5  # Adjusted initial position for better visibility

    # Display the text with colors
    for token, mask_value in zip(text_data, mask_data):
        if mask_value > 0:
            color = colors[mask_value % len(colors)]
            bbox = dict(facecolor=color, edgecolor='none',
                        boxstyle='round,pad=0.3')  # Background color for masked tokens
        else:
            bbox = None

        ax.text(x_pos, y_pos, token, fontsize=12, color='white', ha='left', va='center', bbox=bbox)

        # Update the x position for the next token
        x_pos += len(token) * 0.03 + 0.02  # Adjust the spacing between tokens

    plt.show()

@tensorleap_custom_visualizer(name="loss_visualizer", visualizer_type=LeapDataType.Image)
def loss_visualizer(input_ids: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray) -> LeapImage:
    """
    Description: Computes the combined Categorical Cross-Entropy loss for start and end index predictions.
    Parameters:
    ground_truth (tf.Tensor): Ground truth tensor of shape [B, max_sequence_length, 2].
    prediction (tf.Tensor): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    combined_loss (tf.Tensor): Combined loss for start and end index predictions, computed as the sum of individual Categorical Cross-Entropy losses weighted by alpha.
    """
    input_ids = np.squeeze(input_ids)
    ground_truth = np.squeeze(ground_truth)
    prediction = np.squeeze(prediction)

    mask = mask_based_inputs(input_ids)

    # cut padded tokens
    input_ids = input_ids[mask]
    text_tokens = decode_token_ids(input_ids)

    prediction = transform_prediction(prediction[None, ...])[0]
    # cut padded tokens
    prediction = prediction[mask]
    ground_truth = ground_truth[mask]
    # calc loss per token
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, axis=-1, reduction=tf.keras.losses.Reduction.NONE)
    loss_data = loss(ground_truth, prediction)           # [#tokens, ]
    loss_data = np.expand_dims(loss_data, -1)*100
    # get heatmap based on loss

    lower_percentile = 1  # Lower percentile (10th percentile)
    upper_percentile = 95  # Upper percentile (90th percentile)
    # Determine the actual values at these percentiles
    clip_min = np.percentile(loss_data, lower_percentile)
    clip_max = np.percentile(loss_data, upper_percentile)

    heatmap = plt.imshow(loss_data, cmap='jet', aspect='auto', vmin=clip_min, vmax=clip_max)
    heatmap_data = heatmap.get_array()  # Get the data used to create the heatmap
    norm = heatmap.norm  # Get the normalization used by the heatmap
    cmap = heatmap.cmap  # Get the colormap used by the heatmap
    # Normalize the loss data to the range [0, 1]
    normalized_data = norm(heatmap_data)
    normalized_data = normalized_data / normalized_data.max()
    # Get the RGB values from the colormap
    rgb_values = cmap(normalized_data[..., 0])
    # Remove the alpha channel (if present)
    rgb_values = rgb_values[..., :3]  # Only keep RGB channels
    # Convert to a NumPy array
    rgb_array = np.array(rgb_values)
    n = len(text_tokens)
    rectangle_height, rectangle_width = 1, 1  # Height of each rectangle
    fig, ax = plt.subplots(figsize=(n, rectangle_height))  # Adjust height based on number of tokens

    # Draw rectangles and add text
    for i, token in enumerate(text_tokens):
        # Create a rectangle for each token
        rect = patches.Rectangle(((i * rectangle_width)/n, 0), rectangle_width, rectangle_height, color=rgb_values[i], linewidth=13)
        ax.add_patch(rect)
        # Add token text
        ax.text((i * rectangle_width)/n, 0.5, f'{token}',
                ha='left', va='center', fontsize=8, color='black')

    # plt.show()
    # Set limits and hide axes
    # ax.set_xlim(0, n * rectangle_width)
    # ax.set_ylim(0, rectangle_height)
    ax.axis('off')  # Hide the axes

    # Save the figure to a BytesIO buffer without borders
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    image = plt.imread(buf)[..., :3]  # Read the image from the buffer
    # plt.imshow(image)
    vis = LeapImage((255*image).astype(np.uint8))
    return vis
