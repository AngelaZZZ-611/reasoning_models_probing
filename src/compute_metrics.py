import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import brier_score_loss
import os
from tqdm import tqdm


def compute_metrics(val_labels, val_preds):
    # compute acc, precision, recall, f1
    # compute f1 and macro f1
    # Calculate Metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds)
    f1_macro = f1_score(val_labels, val_preds, average="macro")

    return {
        "accuracy": accuracy*100,
        "precision": precision*100,
        "recall": recall*100,
        "f1": f1*100,
        "f1_macro": f1_macro*100,
    }

def calculate_ece(y_true, y_pred, n_bins=10):
    """
    Calculate the Estimated Calibration Error (ECE).

    Parameters:
    y_true (np.array): True labels (0 or 1).
    y_pred (np.array): Predicted probabilities.
    n_bins (int): Number of bins to use for calibration.

    Returns:
    float: The Estimated Calibration Error.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Bin the predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which samples fall into this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        if np.sum(in_bin) == 0:
            continue  # Skip empty bins
        # Calculate accuracy and confidence in this bin
        bin_acc = np.mean(y_true[in_bin])
        bin_conf = np.mean(y_pred[in_bin])
        # Update ECE
        ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin)
    # Normalize by the total number of samples
    ece /= len(y_true)
    return ece

def compute_other_metrics(val_labels, val_probs):
    # compute roc_auc
    roc_auc = roc_auc_score(val_labels, val_probs)
    # compute ece
    ece = calculate_ece(val_labels, val_probs)
    # compute brier score
    brier_loss = brier_score_loss(val_labels, val_probs)

    return {
        "roc_auc": roc_auc,
        "ece": ece,
        "brier_loss": brier_loss,
    }

def process_file(file_path):
    res_profile = torch.load(file_path)

    test_preds = res_profile['test']['test_preds'] # len:3380
    test_labels = res_profile['test']['test_labels'] # len: 3380
    test_probs = res_profile['test']['test_probs']# len: 3380
    correctness_metrics = compute_metrics(test_labels, test_preds)
    other_metrics = compute_other_metrics(test_labels, test_probs)
    print(correctness_metrics)
    print(other_metrics)
    print("=====================================")
    return {
        **correctness_metrics,
        **other_metrics
    }

