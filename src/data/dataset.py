from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.loader import VQASample
from src.data.vocab import Vocab

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size , image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }
    
class VQADataset(Dataset):
    "yes/no label"
    def __init__(
        self,
        samples: list[VQASample],
        vocab: Vocab,
        label2idx: dict[str, int],
        image_dir: Path,
        transform: transforms.Compose,
        max_seq_len: int = 20,
    ):
        self.samples = samples
        self.vocab = vocab
        self.label2idx = label2idx
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.max_seq_len = max_seq_len
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = self.samples[index],
        image = Image.open(self.image_dir / sample.image_id).convert("RGB")
        image = self.transform(image)
        
        question_ids = self.vocab.encode(sample.question, self.max_seq_len)
        question_ids = torch.tensor(question_ids, dtype=torch.long)
        
        label = torch.tensor(self.label2idx[sample.answer], dtype=torch.long)
        
        return image, question_ids, label