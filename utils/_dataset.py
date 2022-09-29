from typing import Tuple
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader

def get_dataset(args):
    if '10' in args.dataset:
        return cifar_10_dataset(args)
    elif '100' in args.dataset:
        return cifar_100_dataset(args)


def cifar_10_dataset(args) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = CIFAR10( root=args.dataset_path, train=True, transform=train_transform, download=True )
    testset = CIFAR10( root=args.dataset_path, train=False, transform=test_transform, download=True )

    trainloader = DataLoader( dataset = trainset, batch_size = args.batch, num_workers=6, shuffle=True)
    testloader = DataLoader( dataset = testset, batch_size = args.batch, num_workers=6, shuffle=True)
    return trainloader, testloader


def cifar_100_dataset(args) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = CIFAR100( root=args.dataset_path, train=True, transform=train_transform, download=True )
    testset = CIFAR100( root=args.dataset_path, train=False, transform=test_transform, download=True )

    trainloader = DataLoader( dataset = trainset, batch_size = args.batch, num_workers=6, shuffle=True)
    testloader = DataLoader( dataset = testset, batch_size = args.batch, num_workers=6, shuffle=True)

    return trainset, testset