from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import VQADataset, build_transforms
from src.data.loader import VQASample, load_split
from src.data.vocab import Vocab
from src.engine import evaluate, fit
from src.models.cnn_lstm import VQAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def filter_available_samples(samples: list[VQASample], split_path: Path) -> list[VQASample]:
    available = [sample for sample in samples if (config.IMAGE_DIR / sample["image_path"]).is_file()]
    skipped = len(samples) - len(available)
    if skipped:
        logger.warning("%s: skipped %d samples with missing images", split_path.name, skipped)
    if not available:
        raise ValueError(f"No samples with available images in {split_path}")
    return available


def update_artifact_manifest() -> None:
    manifest_path = config.ARTIFACTS_DIR / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    else:
        manifest = {}

    manifest.update(
        {
            "model": config.CNN_LSTM_CHECKPOINT.name,
            "model_type": "CNN+BiLSTM",
            "task": "binary yes/no VQA",
            "source": "local training run",
            "sha256": {},
        }
    )
    for artifact_path in (config.CNN_LSTM_CHECKPOINT, config.LABELS_PATH, config.VOCAB_PATH):
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest().upper()
        manifest["sha256"][artifact_path.name] = digest
    manifest["update_command"] = "Generated automatically by src.train_cnn_lstm"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(path: Path, device: str) -> dict:
    return torch.load(path, map_location=device, weights_only=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.CNN_LSTM_CFG.epochs)
    parser.add_argument("--limit", type=int, default=0, help="Truncate dataset size, 0 = full")
    parser.add_argument("--seed", type=int, default=config.CNN_LSTM_CFG.seed, help="Random seed")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained image backbone")
    parser.add_argument("--resume", type=Path, help="Resume from an epoch checkpoint")
    # Phase 2: thêm flag tuỳ chỉnh optimizer
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    return args

def main() -> None:
    args = parse_args()
    cfg = config.CNN_LSTM_CFG
    seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    train_data = filter_available_samples(load_split(config.TRAIN_SPLIT), config.TRAIN_SPLIT)
    val_data = filter_available_samples(load_split(config.DEV_SPLIT), config.DEV_SPLIT)
    test_data = filter_available_samples(load_split(config.TEST_SPLIT), config.TEST_SPLIT)
    if args.limit:
        train_data = train_data[: args.limit]
        val_data = val_data[: max(1, args.limit // 4)]
        test_data = test_data[: max(1, args.limit // 4)]

    vocab = Vocab.build((s["question"] for s in train_data), min_freq=cfg.min_freq, specials=cfg.specials)
    classes = sorted({s["answer"] for s in train_data})
    if set(classes) != {"no", "yes"}:
        raise ValueError("Training data must contain both 'no' and 'yes' answers")
    label2idx = {label: idx for idx, label in enumerate(classes)}

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tfms = build_transforms(cfg.image_size)
    train_ds = VQADataset(train_data, vocab, label2idx, config.IMAGE_DIR, tfms["train"], cfg.max_seq_len)
    val_ds = VQADataset(val_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)
    test_ds = VQADataset(test_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)

    # Phase 2: num_workers=2 trên Colab/Linux để tăng tốc data loading
    num_workers = 2 if device == "cuda" else 0
    train_batch = args.batch_size or cfg.train_batch_size
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    effective_pretrained = args.pretrained or cfg.pretrained_backbone
    model = VQAModel(
        vocab_size=len(vocab),
        n_classes=len(classes),
        img_model_name=cfg.img_model_name,
        embedding_dim=cfg.embedding_dim,
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        drop_p=cfg.dropout,
        pad_idx=vocab["<pad>"],
        pretrained_backbone=effective_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 2: AdamW thay Adam + CosineAnnealingLR thay StepLR
    effective_lr = args.lr or cfg.lr
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": effective_lr}]
    if backbone_params:
        # Phase 3: Differential Learning Rate (Backbone học chậm hơn 10x để tránh overfit và bảo toàn pretrained weights)
        param_groups.append({"params": backbone_params, "lr": effective_lr * 0.1})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    if args.resume:
        resume = load_checkpoint(args.resume, device)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = int(resume["epoch"])
        logger.info("resumed training from %s at epoch %d", args.resume, start_epoch)

    def save_epoch_checkpoint(epoch: int, result: dict) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {**cfg.__dict__, "pretrained_backbone": effective_pretrained},
            "classes": classes,
            "vocab_size": len(vocab),
            "label2idx": label2idx,
            "metrics": result,
        }
        torch.save(checkpoint, config.ARTIFACTS_DIR / f"cnn_lstm_epoch_{epoch}.pt")
        if result["is_best"]:
            torch.save(checkpoint, config.ARTIFACTS_DIR / "cnn_lstm_best.pt")

    training = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.epochs,
        start_epoch=start_epoch,
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        checkpoint_callback=save_epoch_checkpoint,
        # Phase 2: bật gradient clipping
        grad_clip=1.0,
    )
    model.load_state_dict(training["best_model_state"])
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info("Validation Loss: %.4f, metrics=%s", val_loss, val_metrics)
    logger.info("Test Loss: %.4f, metrics=%s", test_loss, test_metrics)
    vocab.save(config.VOCAB_PATH)
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label2idx, f, indent=4)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {**cfg.__dict__, "pretrained_backbone": effective_pretrained},
            "classes": classes,
            "vocab_size": len(vocab),
            "label2idx": label2idx,
            "metrics": {"val": val_metrics, "test": test_metrics},
            "best_epoch": training["best_epoch"],
            "training_history": training["history"],
        },
        config.CNN_LSTM_CHECKPOINT,
    )
    update_artifact_manifest()
    logger.info("Saved checkpoint to %s", config.CNN_LSTM_CHECKPOINT)

if __name__ == "__main__":
    main()
