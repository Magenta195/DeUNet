from typing import Tuple
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os

def get_dataset(args):
    if '10' in args.dataset:
        return cifar_10_dataset(args)
    elif '100' in args.dataset:
        return cifar_100_dataset(args)
    elif 'mnist' in args.dataset:
        return mnist_dataset(args)
    elif 'medical' in args.dataset :
        return medical_dataset(args)


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

    return trainloader, testloader

def mnist_dataset(args) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = MNIST( root=args.dataset_path, train=True, transform=train_transform, download=True )
    testset = MNIST( root=args.dataset_path, train=False, transform=test_transform, download=True )

    trainloader = DataLoader( dataset = trainset, batch_size = args.batch, num_workers=6, shuffle=True)
    testloader = DataLoader( dataset = testset, batch_size = args.batch, num_workers=6, shuffle=True)
    return trainloader, testloader

def medical_dataset(args) -> Tuple[DataLoader, DataLoader]:
    DATA_PATH = os.path.join(args.dataset_path, "chest_xray")
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    VAL_PATH = os.path.join(DATA_PATH, "val")
    TEST_PATH = os.path.join(DATA_PATH, "test")

    train_val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    trainset = ImageFolder( root = TRAIN_PATH, transform = train_val_transform)
    valset = ImageFolder( root = VAL_PATH, transform = train_val_transform)
    testset = ImageFolder( root = TEST_PATH, transform = test_transform)

    trainset = ConcatDataset([trainset, valset])

    trainloader = DataLoader( dataset = trainset, batch_size = args.batch, num_workers=6, shuffle=True)
    testloader = DataLoader( dataset = testset, batch_size = args.batch, num_workers=6, shuffle=True)

    return trainloader, testloader