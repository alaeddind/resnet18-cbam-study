"""Configuration for reproducible ResNet18/CBAM experiments."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetName = Literal["cifar10", "eurosat"]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetName = "cifar10"
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.1  # Used only for EuroSAT
    image_size: int = 64
    pretrained: bool = False
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.dataset not in ("cifar10", "eurosat"):
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if not (0 < self.val_fraction < 1):
            raise ValueError("val_fraction must be between 0 and 1")
        if not (0 < self.test_fraction < 1):
            raise ValueError("test_fraction must be between 0 and 1")
        if self.dataset == "eurosat" and self.val_fraction + self.test_fraction >= 1:
            raise ValueError("EuroSAT validation and test fractions must sum to less than 1")
        if self.batch_size < 1 or self.image_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid batch_size, image_size or num_workers")
        if self.pretrained:
            raise ValueError("This baseline uses random initialization; set pretrained=False")

    def create_output_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
