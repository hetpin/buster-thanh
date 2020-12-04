import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np


def get_dataloader(root='./data', root_real='./data/real', root_fake='./data/fake', train_ratio=0.7, batch_size=2):
    """get data loader

    Args:
        root (root, optional): root dir
        train_ratio (float, optional): train ratio over full set
        batch_size (int, optional): 

    Returns:
        TYPE: Description
    """
    # Load full dataset
    labels = [0, 1]
    full_set = datasets.ImageFolder(
        root=root, transform=transforms.Compose([transforms.ToTensor()]))

    # split train test
    torch.manual_seed(1000)  # reproducibility
    no_train_sample = int(train_ratio * len(full_set))
    indices = torch.randperm(len(full_set))

    train_set = torch.utils.data.Subset(
        dataset=full_set, indices=indices[0:no_train_sample])
    test_set = torch.utils.data.Subset(
        dataset=full_set, indices=indices[no_train_sample:])
    # print(len(train_set), len(test_set), no_train_sample)

    # batch using dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
