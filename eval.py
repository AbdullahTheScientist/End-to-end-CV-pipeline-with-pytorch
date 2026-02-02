import torch
from utils.metrics import compute_metrics, plot_confusion_matrix

def evaluate(model, dataloader, device="cuda", save_path=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics_dict = compute_metrics(all_labels, all_preds)
    if save_path:
        plot_confusion_matrix(all_labels, all_preds, save_path)

    return metrics_dict
