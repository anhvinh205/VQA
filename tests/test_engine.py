import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine import _classification_metrics, fit


def test_classification_metrics_include_baseline_and_confusion_matrix():
    metrics = _classification_metrics([0, 1, 1], [0, 0, 1], 2)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["majority_baseline_accuracy"] == 2 / 3
    assert metrics["confusion_matrix"] == [[1, 1], [0, 1]]
    assert "macro_f1" in metrics


def test_fit_stops_after_patience_and_saves_epoch_results():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 2)
            self.fc2 = self.layer

        def forward(self, images, questions):
            return self.layer(images)

    model = TinyModel()
    images = torch.zeros(4, 1)
    questions = torch.zeros(4, 1, dtype=torch.long)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, questions, labels), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    saved_epochs = []

    result = fit(
        model,
        loader,
        loader,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        "cpu",
        epochs=5,
        patience=1,
        checkpoint_callback=lambda epoch, _: saved_epochs.append(epoch),
    )

    assert result["last_epoch"] == 2
    assert result["best_epoch"] == 1
    assert saved_epochs == [1, 2]
