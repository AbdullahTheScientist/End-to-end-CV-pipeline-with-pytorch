import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_dataloaders(
    name="cifar10",
    batch_size=16,
    num_workers=0,
    fast_dev_run=False,
    debug_samples=100
):
    """
    Returns: train_loader, val_loader, num_classes
    CPU-friendly and pipeline-test ready
    """

    name = name.lower()

    if name == "cifar10":
        num_classes = 10
        dataset_class = datasets.CIFAR10
        train_args = dict(train=True)
        val_args = dict(train=False)

    elif name == "cifar100":
        num_classes = 100
        dataset_class = datasets.CIFAR100
        train_args = dict(train=True)
        val_args = dict(train=False)

    elif name == "stl10":
        num_classes = 10
        dataset_class = datasets.STL10
        train_args = dict(split="train")
        val_args = dict(split="test")

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # -------------------------
    # Transforms (CPU safe)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_set = dataset_class(
        root="./data",
        download=True,
        transform=transform,
        **train_args
    )

    val_set = dataset_class(
        root="./data",
        download=True,
        transform=transform,
        **val_args
    )

    # -------------------------
    # FAST DEV RUN (tiny subset)
    # -------------------------
    if fast_dev_run:
        train_set = torch.utils.data.Subset(train_set, range(debug_samples))
        val_set = torch.utils.data.Subset(val_set, range(debug_samples))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,   # keep 0 for weak CPU
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, num_classes
