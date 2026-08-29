from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    losses = []
    for images, questions, labels in dataloader:
        images, questions, labels = images.to(device), questions.to(device), labels.to(device)
        outputs = model(images, questions)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return sum(losses) / len(losses), correct / total


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    device: str,
    epochs: int,
) -> dict[str, list[float]]:
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for images, questions, labels in train_loader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            val_acc,
        )

    return history
