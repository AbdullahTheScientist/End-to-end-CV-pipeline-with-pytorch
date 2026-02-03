from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_metrics(labels, preds):
    """
    Compute per-class and global metrics.

    Returns a dict with:
      - per_class: pandas.DataFrame with index=class and columns [accuracy, precision, recall, f1, support]
      - global: pandas.DataFrame with rows [macro, micro, weighted] and columns [precision, recall, f1]
      - accuracy: overall accuracy (float)
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    cm = confusion_matrix(labels, preds)
    num_classes = cm.shape[0]
    total = cm.sum()

    # precision, recall, f1, support per class
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    # per-class accuracy = (TP + TN) / total
    per_class_accuracy = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        acc_i = (tp + tn) / total if total > 0 else 0.0
        per_class_accuracy.append(acc_i)

    per_class_df = pd.DataFrame({
        "accuracy": per_class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })

    # global aggregates
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    micro = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    global_df = pd.DataFrame(
        [
            {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
            {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
            {"precision": weighted[0], "recall": weighted[1], "f1": weighted[2]},
        ],
        index=["macro", "micro", "weighted"]
    )

    metrics = {
        "per_class": per_class_df,
        "global": global_df,
        "accuracy": float(accuracy_score(labels, preds)),
        "confusion_matrix": cm
    }
    return metrics


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
