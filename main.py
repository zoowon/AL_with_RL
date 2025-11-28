"""
명령어 : python main.py --dataset {dataset name} --method {sampling method name} --gpu {gpu num}
데이터셋은 cifar10, cifar100, fashionmnist 중에서 선택
Sampling method는 random, DQN, ~~ 중에서 선택
GPU는 0, 1, 2 중 빈 곳으로 선택
"""

# Python
import argparse
import os
import random
from datetime import datetime
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
import logging

# Custom
import models.resnet as resnet
from models.sampler import SubsetSequentialSampler
from config import *

# Methods
from methods.random import random_sampling
from methods.DQN import DQNAgent, DQN_sampling

# Seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(dataset_name: str, method: str) -> logging.Logger:
    logger = logging.getLogger("active_learning")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    log_dir = os.path.join("logs", dataset_name.lower())
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{method}_{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"로그 파일 경로: {log_path}")

    return logger


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

    if NUM_TRAIN is not None and NUM_TRAIN < len(train_set):
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
        outputs, _ = model(inputs)
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


def active_learning(dataset_name: str, data_root: str, device: torch.device, method:str, initial_labeled: int, addendum: int, logger: logging.Logger):
    train_set, test_set, num_classes = get_datasets(dataset_name, data_root)

    num_train = len(train_set)
    all_indices = list(range(num_train))
    set_seed(RANDOM_SEED)

    random.shuffle(all_indices)
    if initial_labeled > len(all_indices):
        raise ValueError("INITIAL_LABELED is larger than the available training samples.")

    labeled_set = all_indices[:initial_labeled]
    unlabeled_indices = all_indices[initial_labeled:]

    model = resnet.ResNet18(num_classes=num_classes).to(device)
    dqn_agent = None

    for cycle in range(CYCLES):
        logger.info(f"Cycle {cycle + 1}/{CYCLES}")
        logger.info(f"Labeled set size: {len(labeled_set)}, Unlabeled set size: {len(unlabeled_indices)}")

        # Cycle 별로 달라지는 labeled set과 test set에 대한 DataLoader 재정의 
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

        # Cycle 별 labeled set에 대한 train
        for epoch in range(EPOCH):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1:3d}/{EPOCH} | "
                    f"loss {train_loss:.4f} | acc {train_acc * 100:.2f}%"
                )

        # test set에 대한 evaluation 
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(
            f"  >> Test: loss {test_loss:.4f} | acc {test_acc * 100:.2f}%"
        )

        # Unlabeled data가 남아있지 않으면 AL 정지
        if len(unlabeled_indices) == 0:
            logger.info("  No unlabeled samples left. Stopping active learning cycles.")
            break

        # 여러 가지 방식으로 addendum만큼의 새로운 labeled samples 선택
        if method == "random":
            labeled_set, unlabeled_indices = random_sampling(labeled_set, unlabeled_indices, addendum)
        elif method =="DQN":
            labeled_set, unlabeled_indices = DQN_sampling(labeled_set, unlabeled_indices, addendum, model, train_set, device, agent=dqn_agent, num_classes=num_classes)

    # 마지막 cycle 이후 model 저장
    save_dir = os.path.join("./checkpoints", dataset_name.lower())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"resnet18_{method}.pth")
    torch.save({"state_dict": model.state_dict()}, save_path)
    logger.info(f"Saved checkpoint to: {save_path}")


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
        "--method",
        default="random",
        choices=["random", "DQN"]  # 강화학습 방식 선택 가능하도록 추가 필수
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None
    )

    args = parser.parse_args()
    logger = setup_logger(args.dataset, args.method)

    # GPU 선택
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(args.gpu)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # 데이터셋 선택에 따라 initial labeled data 개수 및 cycle 별 추가 labeled data 개수 설정
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if args.dataset.lower() == "cifar100":
        initial_labeled = 2000
        addendum = 2000
    else:
        initial_labeled = INITIAL_LABELED
        addendum = ADDENDUM

    # Sampling 방법 선택
    method = args.method

    active_learning(args.dataset, data_root, device, method, initial_labeled, addendum, logger)


if __name__ == "__main__":
    main()
