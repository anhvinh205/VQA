from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _classification_metrics(
    predictions: list[int], labels: list[int], n_classes: int
) -> dict[str, Any]:
    confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for predicted, label in zip(predictions, labels, strict=False):
        confusion[label][predicted] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for index in range(n_classes):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(index)] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = sum(
        predicted == label for predicted, label in zip(predictions, labels, strict=False)
    ) / len(labels)
    majority_accuracy = max(sum(label == index for label in labels) for index in range(n_classes)) / len(
        labels
    )
    return {
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_accuracy,
        "macro_precision": sum(precisions) / n_classes,
        "macro_recall": sum(recalls) / n_classes,
        "macro_f1": sum(f1s) / n_classes,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, criterion, device: str) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses = []
    predictions, labels_seen = [], []
    for images, questions, labels in data_loader:
        images, questions, labels = images.to(device), questions.to(device), labels.to(device)
        outputs = model(images, questions)
        losses.append(criterion(outputs, labels).item())
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        labels_seen.extend(labels.cpu().tolist())

    if not labels_seen:
        raise ValueError("Cannot evaluate an empty dataloader")
    return sum(losses) / len(losses), _classification_metrics(
        predictions, labels_seen, model.fc2.out_features
    )


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    device: str,
    epochs: int,
    start_epoch: int = 0,
    patience: int = 3,
    min_delta: float = 1e-4,
    checkpoint_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_macro_f1 = float("-inf")
    best_epoch = start_epoch
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(start_epoch, epochs):
        model.train()
        batch_losses = []
        for images, questions, labels in train_loader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, questions), labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        improved = val_metrics["macro_f1"] > best_macro_f1 + min_delta
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "is_best": improved,
        }
        if checkpoint_callback is not None:
            checkpoint_callback(epoch + 1, epoch_result)
        logger.info(
            "epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f%s",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            " [best]" if improved else "",
        )
        if epochs_without_improvement >= patience:
            logger.info("early stopping at epoch %d; best_epoch=%d", epoch + 1, best_epoch)
            break
    if best_model_state is None:
        raise ValueError("Training did not produce a validation checkpoint")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_model_state": best_model_state,
        "last_epoch": start_epoch + len(history["train_loss"]),
    }
