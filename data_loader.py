"""BreakHis breast cancer data loading and non-IID client partitioning.

Expects the dataset at ``data/breakhis/`` with subdirectories per class::

    data/breakhis/
    ├── benign/
    │   ├── SOB_B_A-14-22549AB-100-001.png
    │   └── ...
    └── malignant/
        ├── SOB_M_DC-14-2980-100-001.png
        └── ...
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from config import (
    BREAKHIS_MEAN,
    BREAKHIS_PATH,
    BREAKHIS_STD,
    DIRICHLET_ALPHA,
    IMG_SIZE,
    NUM_CLIENTS,
    RANDOM_SEED,
    TEST_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)


def _train_transform():
    """Augmentation pipeline for training histopathology images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(BREAKHIS_MEAN, BREAKHIS_STD),
    ])


def _test_transform():
    """Deterministic pipeline for evaluation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(BREAKHIS_MEAN, BREAKHIS_STD),
    ])


def _augmented_train_transform():
    """Stronger augmentation pipeline for the regularization solution."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(BREAKHIS_MEAN, BREAKHIS_STD),
    ])


_GLOBAL_IMAGE_CACHE = {}


class CachedImageFolder(datasets.ImageFolder):
    """A subclass of ImageFolder that caches resized PIL images in memory to speed up I/O."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if path in _GLOBAL_IMAGE_CACHE:
            return _GLOBAL_IMAGE_CACHE[path], target

        sample, target = super().__getitem__(index)
        # Resize to target image size immediately to save memory in cache
        sample = sample.resize((IMG_SIZE, IMG_SIZE))
        _GLOBAL_IMAGE_CACHE[path] = sample
        return sample, target


class TransformSubset(Dataset):
    """Wraps a Subset to apply a custom transform instead of the parent's."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # ``subset[idx]`` may return a PIL Image if the parent dataset's
        # transform was set to None before creating the Subset.  If a
        # transform was already applied (returning a Tensor), we skip.
        if self.transform is not None and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        return image, label


def _dirichlet_partition(targets, num_clients, alpha, seed):
    """Create non-IID partitions using a Dirichlet distribution.

    Lower ``alpha`` values produce more skewed (non-IID) distributions.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    num_classes = len(np.unique(targets))

    client_indices = defaultdict(list)

    for class_id in range(num_classes):
        class_mask = np.where(targets == class_id)[0]
        rng.shuffle(class_mask)

        # Draw a Dirichlet distribution to decide how many samples each
        # client gets from this class.
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()  # ensure sums to 1

        split_points = (np.cumsum(proportions) * len(class_mask)).astype(int)
        parts = np.split(class_mask, split_points[:-1])

        for client_id, part in enumerate(parts):
            client_indices[client_id].extend(part.tolist())

    # Shuffle each client's indices
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return [client_indices[i] for i in range(num_clients)]


def create_non_iid_breakhis(
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA,
    test_fraction=0.2,
    augmented=False,
):
    """Load BreakHis and return non-IID train loaders + a global test loader.

    Parameters
    ----------
    num_clients : int
        Number of federated clients.
    alpha : float
        Dirichlet concentration parameter (lower → more non-IID).
    test_fraction : float
        Fraction of the full dataset reserved for testing.
    augmented : bool
        If True, use stronger augmentation for training (regularization solution).

    Returns
    -------
    train_loaders : list[DataLoader]
        One DataLoader per client with non-IID training data.
    test_loader : DataLoader
        A single global test DataLoader.
    """
    # Load without transforms so we can apply them per-subset later
    full_dataset = CachedImageFolder(root=str(BREAKHIS_PATH), transform=None)

    # Deterministic train/test split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    total = len(full_dataset)
    test_size = int(total * test_fraction)
    train_size = total - test_size
    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    # Extract targets for the training subset
    train_targets = [full_dataset.targets[i] for i in train_subset.indices]

    # Partition training data across clients using Dirichlet
    client_index_lists = _dirichlet_partition(
        train_targets,
        num_clients=num_clients,
        alpha=alpha,
        seed=RANDOM_SEED,
    )

    # Map partition indices (relative to train_subset) back to full_dataset indices
    train_indices = list(train_subset.indices)

    train_transform = _augmented_train_transform() if augmented else _train_transform()
    test_transform = _test_transform()

    train_loaders = []
    for partition_indices in client_index_lists:
        # Map relative indices to full dataset indices
        absolute_indices = [train_indices[i] for i in partition_indices]
        subset = Subset(full_dataset, absolute_indices)
        wrapped = TransformSubset(subset, transform=train_transform)
        train_loaders.append(
            DataLoader(wrapped, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
        )

    test_wrapped = TransformSubset(test_subset, transform=test_transform)
    test_loader = DataLoader(
        test_wrapped, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loaders, test_loader


def print_client_distribution(train_loaders, class_names=None):
    """Print class distribution per client for debugging / reporting."""
    if class_names is None:
        class_names = ["benign", "malignant"]

    print("\n--- Client Data Distribution (Non-IID) ---")
    for client_id, loader in enumerate(train_loaders):
        counts = defaultdict(int)
        for _, labels in loader:
            for label in labels:
                counts[label.item()] += 1
        total = sum(counts.values())
        dist = ", ".join(
            f"{class_names[cls]}: {counts[cls]} ({counts[cls]/total*100:.1f}%)"
            for cls in sorted(counts.keys())
        )
        print(f"  Client {client_id}: {total} samples → {dist}")
    print()
