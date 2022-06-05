from colossalai.utils import get_dataloader

from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageNet
import torch.nn as nn
from colossalai.registry import LOSSES, HOOKS
from colossalai.trainer import hooks
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

    train_dataset = CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=root, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader

def build_imagenet(batch_size, root, padding=None, pad_if_needed=False, crop=224, resize=224):
    transform_train = transforms.Compose([
        transforms.RandomCrop(crop, padding=padding, pad_if_needed=pad_if_needed),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = ImageNet(root=root, train=True, download=True, transform=transform_train)
    test_dataset = ImageNet(root=root, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


def save_checkpoint(model, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, checkpoint_file):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if __name__ == "__main__":
    build_cifar(16, './data')