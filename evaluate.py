"""Validation/test evaluation; test should be run once per final checkpoint."""
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, loader, device, criterion=None, return_confusion=False):
    model.eval()
    predictions, targets = [], []
    total_loss, total_samples = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * y.size(0)
            total_samples += y.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    result = {"accuracy": float(accuracy_score(targets, predictions)),
              "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
              "loss": total_loss / total_samples if criterion is not None else None}
    if return_confusion:
        result["confusion_matrix"] = confusion_matrix(targets, predictions).tolist()
    return result
