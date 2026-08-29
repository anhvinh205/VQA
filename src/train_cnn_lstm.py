"""Train the CNN + BiLSTM baseline and save a deployable checkpoint.

Usage:
    python -m src.train_cnn_lstm [--epochs 5] [--limit 0]

`--limit` truncates the train/val sets (useful for a fast smoke test
or for CI). Use 0 for the full dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import random

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.CNN_LSTM_CFG.epochs)
    parser.add_argument("--limit", type=int, default=0, help="Truncate dataset size, 0 = full")
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained backbone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config.CNN_LSTM_CFG
    set_seed(cfg.seed)
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
    label2idx = {c: i for i, c in enumerate(classes)}

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vocab.save(config.VOCAB_PATH)
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label2idx, f)

    tfms = build_transforms(cfg.image_size)
    train_ds = VQADataset(train_data, vocab, label2idx, config.IMAGE_DIR, tfms["train"], cfg.max_seq_len)
    val_ds = VQADataset(val_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)
    test_ds = VQADataset(test_data, vocab, label2idx, config.IMAGE_DIR, tfms["eval"], cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)

    model = VQAModel(
        vocab_size=len(vocab),
        n_classes=len(classes),
        img_model_name=cfg.img_model_name,
        embedding_dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
        hidden_size=cfg.hidden_size,
        drop_p=cfg.dropout,
        pretrained_backbone=args.pretrained or cfg.pretrained_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(args.epochs * 0.8)), gamma=0.1
    )

    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("val_acc=%.4f test_acc=%.4f", val_acc, test_acc)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "classes": classes,
            "vocab_size": len(vocab),
        },
        config.CNN_LSTM_CHECKPOINT,
    )
    logger.info("saved checkpoint to %s", config.CNN_LSTM_CHECKPOINT)


if __name__ == "__main__":
    main()
