from __future__ import annotations

import argparse
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
from src.data.loader import load_split
from src.data.vocab import Vocab
from src.engine import evaluate, fit
from src.models.cnn_lstm import VQAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.CNN_LSTM_CFG.epochs)
    parser.add_argument("--limit", type=int, default=0, help="Truncate dataset size, 0 = full")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained image backbone")
    parser.add_argument("--resume", type=Path, help="Resume from an epoch checkpoint")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = config.CNN_LSTM_CFG
    seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    train_data = load_split(config.TRAIN_SPLIT)
    val_data = load_split(config.DEV_SPLIT)
    test_data = load_split(config.TEST_SPLIT)
    if args.limit:
        train_data = train_data[: args.limit]
        val_data = val_data[: max(1, args.limit // 4)]
        test_data = test_data[: max(1, args.limit // 4)]

    vocab = Vocab.build((s["question"] for s in train_data), min_freq=cfg.min_freq, specials=cfg.specials)
    classes = sorted({s["answer"] for s in train_data})
    label2idx = {label: idx for idx, label in enumerate(classes)}

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vocab.save(config.VOCAB_PATH)
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label2idx, f, indent=4)

    tfms = build_transforms(cfg.image_size)
    train_ds = VQADataset(train_data, vocab, label2idx, config.IMAGE_DIR, tfms["train"], cfg.max_seq_len)
    val_ds = VQADataset(val_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)
    test_ds = VQADataset(test_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(args.epochs * 0.8)), gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
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
    )
    model.load_state_dict(training["best_model_state"])
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info("Validation Loss: %.4f, metrics=%s", val_loss, val_metrics)
    logger.info("Test Loss: %.4f, metrics=%s", test_loss, test_metrics)
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
    logger.info("Saved checkpoint to %s", config.CNN_LSTM_CHECKPOINT)

if __name__ == "__main__":
    main()