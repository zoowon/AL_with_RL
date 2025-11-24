# Python
import argparse
import os
import random

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
import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from data.sampler import SubsetSequentialSampler

# Seed
random.seed("Reinforcement Learning")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

## 데이터 로더
def get_transforms(dataset_name):
    if dataset_name in {"cifar10", "cifar100"}:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif dataset_name == "fashionmnist":
        mean = [0.2860, 0.2860, 0.2860]
        std = [0.3530, 0.3530, 0.3530]
        train_transform = T.Compose([
            T.Resize(32),
            T.Grayscale(num_output_channels=3),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        test_transform = T.Compose([
            T.Resize(32),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset_name}")

    return train_transform, test_transform


def get_datasets(dataset_name, data_root):
    train_transform, test_transform = get_transforms(dataset_name)
    dataset_path = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        train_set = CIFAR10(dataset_path, train=True, download=True, transform=train_transform)
        unlabeled_set = CIFAR10(dataset_path, train=True, download=True, transform=test_transform)
        test_set = CIFAR10(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = CIFAR100(dataset_path, train=True, download=True, transform=train_transform)
        unlabeled_set = CIFAR100(dataset_path, train=True, download=True, transform=test_transform)
        test_set = CIFAR100(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset_name == "fashionmnist":
        train_set = FashionMNIST(dataset_path, train=True, download=True, transform=train_transform)
        unlabeled_set = FashionMNIST(dataset_path, train=True, download=True, transform=test_transform)
        test_set = FashionMNIST(dataset_path, train=False, download=True, transform=test_transform)
        num_classes = 10
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset_name}")

    return train_set, unlabeled_set, test_set, num_classes


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


# Train Utils
iters = 0

def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss.item(),
                m_module_loss.item(),
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset 선택을 위한 Active Learning 실험 스크립트')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'fashionmnist'],
                        help='사용할 데이터셋 이름')
    parser.add_argument('--data-root', default='../data', help='데이터 다운로드 및 로딩 경로')
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    data_root = args.data_root

    train_set, unlabeled_set, test_set, num_classes = get_datasets(dataset_name, data_root)
    num_train = len(train_set)

    vis = visdom.Visdom(server='http://localhost', port=9000)
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}
    checkpoint_dir = os.path.join('.', dataset_name, 'train', 'weights')

    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(num_train))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_indices = indices[ADDENDUM:]

        train_loader = DataLoader(train_set, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18 = resnet.ResNet18(num_classes=num_classes).cuda()
        loss_module = lossnet.LossNet().cuda()
        models = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data, checkpoint_dir)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample from unlabeled data points
            random.shuffle(unlabeled_indices)
            subset = unlabeled_indices[:min(SUBSET, len(unlabeled_indices))]

            if len(subset) == 0:
                print('남은 비라벨 데이터가 없어 사이클을 종료합니다.')
                break

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            add_count = min(ADDENDUM, len(subset))
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-add_count:].numpy())
            unlabeled_indices = list(torch.tensor(subset)[arg][:-add_count].numpy()) + unlabeled_indices[len(subset):]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(train_set, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                os.path.join(checkpoint_dir, 'active_resnet18_{}_trial{}.pth'.format(dataset_name, trial)))
