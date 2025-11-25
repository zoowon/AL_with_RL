"""
명령어 : python main.py --dataset cifar10 --gpu 1
데이터셋은 cifar10, cifar100, fashionmnist에서 고르면 됨
"""

# Python
import argparse
import os
import random
from typing import Tuple, List

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST

# Utils
from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from data.sampler import SubsetSequentialSampler

# Seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name in {"cifar10", "cifar100"}:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    elif dataset_name == "fashionmnist":
        mean = [0.2860, 0.2860, 0.2860]
        std = [0.3530, 0.3530, 0.3530]
        train_transform = T.Compose(
            [
                T.Resize(32),
                T.Grayscale(num_output_channels=3),
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

        test_transform = T.Compose(
            [
                T.Resize(32),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_transform, test_transform


def get_datasets(dataset_name: str, data_root: str):
    train_transform, test_transform = get_transforms(dataset_name)
    dataset_path = os.path.join(data_root, dataset_name.lower())

    if dataset_name.lower() == "cifar10":
        train_set = CIFAR10(dataset_path, train=True, download=True, transform=train_transform)
        test_set = CIFAR10(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        train_set = CIFAR100(dataset_path, train=True, download=True, transform=train_transform)
        test_set = CIFAR100(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset_name.lower() == "fashionmnist":
        train_set = FashionMNIST(dataset_path, train=True, download=True, transform=train_transform)
        test_set = FashionMNIST(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # If NUM_TRAIN is smaller than the dataset size, we only use the first NUM_TRAIN samples
    if NUM_TRAIN is not None and NUM_TRAIN < len(train_set):
        # torchvision CIFAR / MNIST 계열은 data & targets 속성을 가짐
        if hasattr(train_set, "data"):
            train_set.data = train_set.data[:NUM_TRAIN]
        if hasattr(train_set, "targets"):
            train_set.targets = train_set.targets[:NUM_TRAIN]
        elif hasattr(train_set, "labels"):
            train_set.labels = train_set.labels[:NUM_TRAIN]

    return train_set, test_set, num_classes


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)  # ResNet18 in resnet.py returns (logits, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def random_sampling_step(labeled_indices: List[int], unlabeled_indices: List[int], addendum: int):
    if len(unlabeled_indices) == 0:
        return labeled_indices, unlabeled_indices

    add_count = min(addendum, len(unlabeled_indices))
    newly_labeled = random.sample(unlabeled_indices, add_count)

    new_labeled = labeled_indices + newly_labeled
    remaining_unlabeled = [idx for idx in unlabeled_indices if idx not in newly_labeled]
    return new_labeled, remaining_unlabeled


def active_learning_random(dataset_name: str, data_root: str, device: torch.device):
    train_set, test_set, num_classes = get_datasets(dataset_name, data_root)

    num_train = len(train_set)
    all_indices = list(range(num_train))

    for trial in range(TRIALS):
        print(f"=== Trial {trial + 1}/{TRIALS} ===")
        # Different seed per trial for fair randomness
        set_seed(RANDOM_SEED + trial)

        random.shuffle(all_indices)
        if INITIAL_LABELED > len(all_indices):
            raise ValueError("INITIAL_LABELED is larger than the available training samples.")

        labeled_set = all_indices[:INITIAL_LABELED]
        unlabeled_indices = all_indices[INITIAL_LABELED:]

        # Model for this trial
        model = resnet.ResNet18(num_classes=num_classes).to(device)

        for cycle in range(CYCLES):
            print(f"\n[Trial {trial + 1}/{TRIALS}] Cycle {cycle + 1}/{CYCLES}")
            print(f"Labeled set size: {len(labeled_set)}, Unlabeled set size: {len(unlabeled_indices)}")

            # DataLoaders for current labeled set and test set
            train_loader = DataLoader(
                train_set,
                batch_size=BATCH,
                sampler=SubsetRandomSampler(labeled_set),
                num_workers=NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
            )
            test_loader = DataLoader(
                test_set,
                batch_size=BATCH,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
            )

            # Optimization setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY
            )
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

            # Train for EPOCH epochs on current labeled set
            for epoch in range(EPOCH):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                scheduler.step()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(
                        f"  Epoch {epoch + 1:3d}/{EPOCH} | "
                        f"loss {train_loss:.4f} | acc {train_acc * 100:.2f}%"
                    )

            # Evaluate on test set
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(
                f"  >> Test: loss {test_loss:.4f} | acc {test_acc * 100:.2f}%"
            )

            # If no unlabeled data remains, stop AL cycles
            if len(unlabeled_indices) == 0:
                print("  No unlabeled samples left. Stopping active learning cycles.")
                break

            # Randomly add ADDENDUM new labeled samples
            labeled_set, unlabeled_indices = random_sampling_step(
                labeled_set, unlabeled_indices, ADDENDUM
            )

        # Optionally: save model after last cycle of each trial
        save_dir = os.path.join("./checkpoints", dataset_name.lower())
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"resnet18_random_trial{trial + 1}.pth")
        torch.save({"state_dict": model.state_dict()}, save_path)
        print(f"Saved checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning with Random Sampling (image classification)."
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "fashionmnist"]
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    active_learning_random(args.dataset, data_root, device)


if __name__ == "__main__":
    main()
