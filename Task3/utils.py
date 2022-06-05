import numpy as np
from sched import scheduler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.nn.parallel

def dataloading(mean, std, dataset="CIFAR100", batch_size=128, num_workers=16, resize=False):
    normalize = transforms.Normalize(mean=mean, std=std)
    if resize:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


    if dataset=="CIFAR100":
        train_images = datasets.CIFAR100('cifar100', train=True, download=True, transform=transform_train)
        test_images = datasets.CIFAR100('cifar100', train=False, download=True, transform=transform_test)
    else:
        train_images = datasets.CIFAR10('cifar10', train=True, download=True, transform=transform_train)
        test_images = datasets.CIFAR10('cifar10', train=False, download=True, transform=transform_test)

    train_data = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = DataLoader(test_images, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data, test_data
    
def train(train_data, net, criterion, optimizer, epoch, prob=0, beta=20, aug_type="cutmix", verbose=True, dit=False):
    temp_loss = AverageMeter()
    temp_correct = AverageMeter()
    # print("Mock(2)")
    net.train()
    for X, y in train_data:
        X = X.cuda()
        y = y.cuda()
        # print(X.shape)
        # print(y.shape)
        if prob == 0 or aug_type == "baseline" or dit:
            if dit:
                loss = net(X, y)
                with torch.no_grad():
                    y_hat = net.student(X)
                    loss_train = criterion(y_hat, y)
            else:
                y_hat = net(X)
                loss = criterion(y_hat, y)
        else:
            r = np.random.rand(1)
            if r < prob:
                if aug_type == "cutmix":
                    """
                    Using cutmix augmentation
                    """
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(X.size()[0]).cuda()
                    target_a = y
                    target_b = y[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                    X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (X.size()[-1] * X.size()[-2])

                    y_hat = net(X)
                    loss = criterion(y_hat, target_a) * lam + criterion(y_hat, target_b) * (1. - lam)
                elif aug_type == "cutout":
                    """
                    Using cutout augmentation
                    """ 
                    lam = np.random.beta(beta, beta)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                    X[:, :, bbx1:bbx2, bby1:bby2] = 0.0

                    y_hat = net(X)
                    loss = criterion(y_hat, y)
                elif aug_type == "mixup":
                    """
                    Using mixup augmentation
                    """
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(X.size()[0]).cuda()
                    target_a = y
                    target_b = y[rand_index]
                    X = X * lam + X[rand_index] * (1. - lam)

                    y_hat = net(X)
                    loss = criterion(y_hat, target_a) * lam + criterion(y_hat, target_b) * (1. - lam)
            else:
                y_hat = net(X)
                loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct,  = accuracy(y_hat.data, y)
        temp_correct.update(correct.item(), X.shape[0])
        if dit:
            temp_loss.update(loss_train.item(), X.shape[0])
        else:
            temp_loss.update(loss.item(), X.shape[0])
        loss.detach()
            
    # temp_loss = temp_loss / len(train_data.dataset)
    # temp_correct = temp_correct / len(train_data.dataset) * 100.0
    if verbose:
        print('epoch: %d, train loss: %.3f, train accuracy: %.2f' % (epoch, temp_loss.avg, temp_correct.avg))
    return temp_loss.avg, temp_correct.avg
     
def test(test_data, net, criterion, epoch, verbose=True, dit=False):
    net.eval()
    temp_loss = AverageMeter()
    temp_correct = AverageMeter()
    with torch.no_grad():
        for X, y in test_data:
            X = X.cuda()
            y = y.cuda()
            if dit:
                y_hat = net.student(X)
                loss = criterion(y_hat, y)
            else:
                y_hat = net(X)
                loss = criterion(y_hat, y)

            correct, = accuracy(y_hat.data, y)
            temp_correct.update(correct.item(), X.shape[0])
            temp_loss.update(loss.item(), X.shape[0]) 

    # temp_loss = temp_loss / len(test_data.dataset)
    # temp_correct = temp_correct / len(test_data.dataset) * 100.0
    if verbose:
        print('epoch: %d, test loss: %.3f, test accuracy: %.2f' % (epoch, temp_loss.avg, temp_correct.avg))
    return temp_loss.avg, temp_correct.avg

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
