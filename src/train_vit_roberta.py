"""Training pipeline for the ViT + RoBERTa multi-modal Transformer model."""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src import config
from src.data.loader import VQASample, load_split
from src.engine import _classification_metrics
from src.models.vit_roberta import Classifier, TextEncoder, VisualEncoder, VQAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIT_CHECKPOINT = config.ARTIFACTS_DIR / "vit_roberta_model.pt"
VIT_BEST_CHECKPOINT = config.ARTIFACTS_DIR / "vit_roberta_best.pt"


class VitRobertaDataset(Dataset):
    """Dataset feeding tokenized text and processed pixel values for ViT + RoBERTa."""

    def __init__(
        self,
        samples: list[VQASample],
        image_dir: Path,
        label2idx: dict[str, int],
        tokenizer: Any,
        image_processor: Any,
        max_seq_len: int = 32,
    ):
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.label2idx = label2idx
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.image_dir / sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        # ViT input
        img_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].squeeze(0)  # (3, 224, 224)

        # RoBERTa input
        text_inputs = self.tokenizer(
            sample["question"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        label = torch.tensor(self.label2idx[sample["answer"]], dtype=torch.long)
        return pixel_values, input_ids, attention_mask, label


def filter_available_samples(samples: list[VQASample], split_path: Path) -> list[VQASample]:
    available = [sample for sample in samples if (config.IMAGE_DIR / sample["image_path"]).is_file()]
    skipped = len(samples) - len(available)
    if skipped:
        logger.warning("%s: skipped %d samples with missing images", split_path.name, skipped)
    if not available:
        raise ValueError(f"No samples with available images in {split_path}")
    return available


def seed(seed_val: int) -> None:
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_vit(
    model: VQAModel, data_loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses = []
    predictions, labels_seen = [], []

    for pixel_values, input_ids, attention_mask, labels in data_loader:
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        img_dict = {"pixel_values": pixel_values}
        text_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        outputs = model(img_dict, text_dict)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        labels_seen.extend(labels.cpu().tolist())

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    metrics = _classification_metrics(predictions, labels_seen, n_classes=2)
    return avg_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT + RoBERTa VQA Model")
    parser.add_argument("--epochs", type=int, default=8, help="Total fine-tuning epochs")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Epochs to train only the classifier")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone learning rate for fine-tuning")
    parser.add_argument("--classifier-lr", type=float, default=1e-4, help="Classifier learning rate")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size (for testing)")
    parser.add_argument("--seed", type=int, default=59, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device=%s for ViT + RoBERTa training", device)

    # 1. Load Data
    train_data = filter_available_samples(load_split(config.TRAIN_SPLIT), config.TRAIN_SPLIT)
    val_data = filter_available_samples(load_split(config.DEV_SPLIT), config.DEV_SPLIT)
    test_data = filter_available_samples(load_split(config.TEST_SPLIT), config.TEST_SPLIT)

    if args.limit:
        train_data = train_data[: args.limit]
        val_data = val_data[: max(1, args.limit // 4)]
        test_data = test_data[: max(1, args.limit // 4)]

    classes = ["no", "yes"]
    label2idx = {label: idx for idx, label in enumerate(classes)}

    # 2. Tokenizer & Image Processor
    vit_model_name = "google/vit-base-patch16-224"
    roberta_model_name = "roberta-base"

    logger.info("Loading preprocessors: %s, %s", vit_model_name, roberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

    train_ds = VitRobertaDataset(train_data, config.IMAGE_DIR, label2idx, tokenizer, image_processor)
    val_ds = VitRobertaDataset(val_data, config.IMAGE_DIR, label2idx, tokenizer, image_processor)
    test_ds = VitRobertaDataset(test_data, config.IMAGE_DIR, label2idx, tokenizer, image_processor)

    num_workers = 2 if device == "cuda" else 0
    pin_mem = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    # 3. Model
    logger.info("Building ViT + RoBERTa model...")
    visual_encoder = VisualEncoder(vit_model_name)
    text_encoder = TextEncoder(roberta_model_name)
    classifier = Classifier(hidden_size=512, dropout_prob=0.3, n_classes=2)
    model = VQAModel(visual_encoder, text_encoder, classifier).to(device)

    criterion = nn.CrossEntropyLoss()

    # 4. Stage 1: Classifier Warmup (Backbones frozen)
    if args.warmup_epochs > 0:
        logger.info("=== Stage 1: Classifier Warmup (%d epochs) ===", args.warmup_epochs)
        model.freeze(visual=True, textual=True, classifier=False)
        warmup_opt = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(args.warmup_epochs):
            model.train()
            batch_losses = []
            for pixel_values, input_ids, attention_mask, labels in train_loader:
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                warmup_opt.zero_grad()
                img_dict = {"pixel_values": pixel_values}
                text_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
                loss = criterion(model(img_dict, text_dict), labels)
                loss.backward()
                warmup_opt.step()
                batch_losses.append(loss.item())

            train_loss = sum(batch_losses) / len(batch_losses)
            val_loss, val_m = evaluate_vit(model, val_loader, criterion, device)
            logger.info(
                "Warmup Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
                epoch + 1,
                args.warmup_epochs,
                train_loss,
                val_loss,
                val_m["accuracy"],
                val_m["macro_f1"],
            )

    # 5. Stage 2: Full Fine-tuning with Differential Learning Rate
    logger.info("=== Stage 2: Full Fine-tuning (%d epochs) ===", args.epochs)
    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    optimizer_params = [
        {"params": visual_encoder.parameters(), "lr": args.lr},
        {"params": text_encoder.parameters(), "lr": args.lr},
        {"params": classifier.parameters(), "lr": args.classifier_lr},
    ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for pixel_values, input_ids, attention_mask, labels in train_loader:
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            img_dict = {"pixel_values": pixel_values}
            text_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = model(img_dict, text_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        train_loss = sum(batch_losses) / len(batch_losses)
        val_loss, val_m = evaluate_vit(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_m["accuracy"])
        history["val_macro_f1"].append(val_m["macro_f1"])

        improved = val_m["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_m["macro_f1"]
            best_epoch = epoch + 1
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "metrics": val_m,
                    "classes": classes,
                    "label2idx": label2idx,
                },
                VIT_BEST_CHECKPOINT,
            )

        logger.info(
            "Fine-tune Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f%s",
            epoch + 1,
            args.epochs,
            train_loss,
            val_loss,
            val_m["accuracy"],
            val_m["macro_f1"],
            " [BEST]" if improved else "",
        )

    # 6. Final Evaluation on Test set
    if VIT_BEST_CHECKPOINT.exists():
        best_ckpt = torch.load(VIT_BEST_CHECKPOINT, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])

    val_loss, final_val_m = evaluate_vit(model, val_loader, criterion, device)
    test_loss, final_test_m = evaluate_vit(model, test_loader, criterion, device)

    logger.info("=== FINAL RESULTS ===")
    logger.info(
        "Validation Loss: %.4f | Accuracy: %.4f | Macro F1: %.4f",
        val_loss,
        final_val_m["accuracy"],
        final_val_m["macro_f1"],
    )
    logger.info(
        "Test Loss:       %.4f | Accuracy: %.4f | Macro F1: %.4f",
        test_loss,
        final_test_m["accuracy"],
        final_test_m["macro_f1"],
    )

    # Save final model
    torch.save(
        {
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "metrics": {"val": final_val_m, "test": final_test_m},
            "history": history,
            "classes": classes,
            "label2idx": label2idx,
        },
        VIT_CHECKPOINT,
    )
    logger.info("Saved final checkpoint to %s", VIT_CHECKPOINT)


if __name__ == "__main__":
    main()
