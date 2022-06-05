from colossalai.utils import get_dataloader

from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
from colossalai.registry import LOSSES
import torch


def build_cifar(batch_size, root, padding=None, pad_if_needed=False, crop=224, resize=224):
    transform_train = transforms.Compose([
        transforms.RandomCrop(crop, padding=padding, pad_if_needed=pad_if_needed),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=root, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader

@LOSSES.register_module
class MixupLoss(nn.Module):
    def __init__(self, loss_fn_cls):
        super().__init__()
        self.loss_fn = loss_fn_cls()

    def forward(self, inputs, targets_a, targets_b, lam):
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)


class MixupAccuracy(nn.Module):
    def forward(self, logits, targets):
        targets = targets['targets_a']
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        return correct