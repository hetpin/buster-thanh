import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models.net_classify import NetClassify
from utils import get_dataloader

import numpy as np


def main():
    print('a simply classifier')

    # 1. innit hyper params
    no_epoch = 10
    no_freq = 2
    batch_size = 2
    no_classes = 2
    lr = 1e-6
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 2. dataloader
    train_loader, test_loader = get_dataloader(batch_size=batch_size)
    # train_loader.to(device)
    # test_loader.to(device)
    # visualize the data loader for testing later

    # 3. model
    net = NetClassify().to(device)

    # 4. training
    opt = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    global_counter = 0
    for epoch in range(no_epoch):
        running_loss = 0
        for images, labels in iter(train_loader):
            # size prob
            labels = torch.nn.functional.one_hot(
                labels, num_classes=no_classes)
            labels = labels.type(torch.float32)

            # update model
            opt.zero_grad()
            preds = net(images)
            print(labels.dtype, preds.dtype)

            loss = criterion(preds, labels)
            loss.backward()
            opt.step()

            # analyze: print loss in each no_freq
            running_loss += loss.item() / no_freq
            global_counter += 1
            if not global_counter % no_freq:
                print('global_counter %d: running_loss %f' %
                      (global_counter, running_loss))
                running_loss = 0

    # 5. evaluation


if __name__ == '__main__':
    main()
