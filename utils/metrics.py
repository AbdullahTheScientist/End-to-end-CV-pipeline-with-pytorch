from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compute_metrics(labels, preds):
    per_class = precision_recall_fscore_support(labels, preds, average=None)
    macro = precision_recall_fscore_support(labels, preds, average='macro')
    weighted = precision_recall_fscore_support(labels, preds, average='weighted')

    metrics = {
        "per_class": per_class,
        "macro": macro,
        "weighted": weighted,
        "accuracy": accuracy_score(labels, preds)
    }
    return metrics

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.close()
