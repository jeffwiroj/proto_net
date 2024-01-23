import numpy as np
from PIL import Image

#  Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# TorchVision
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms as T

DATA_PATH = "data/"
DATA_MEANS = torch.Tensor([0.5183975, 0.49192241, 0.44651328])
DATA_STD = torch.Tensor([0.26770132, 0.25828985, 0.27961241])

test_transform = T.Compose([T.ToTensor(), T.Normalize(DATA_MEANS, DATA_STD)])

# For training, we add some augmentation.
train_transform = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(DATA_MEANS, DATA_STD),
    ]
)


class ImageDataset(data.Dataset):
    def __init__(self, imgs, targets, img_transform):
        super().__init__()

        self.img_transform = img_transform
        self.imgs = imgs
        self.targets = targets

    def __getitem__(self, idx):
        img, target = self.imgs[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, target

    def __len__(self):
        return self.imgs.shape[0]


def dataset_from_labels(imgs, targets, class_set, **kwargs):
    class_mask = (targets[:, None] == class_set[:None]).any(dim=-1)
    return ImageDataset(imgs=imgs[class_mask], targets=targets[class_mask], **kwargs)


def get_dataset(seed: int = 0):
    # Download Cifar Dataset
    train_set = CIFAR100(
        root=DATA_PATH, train=True, download=True, transform=T.ToTensor()
    )
    test_set = CIFAR100(
        root=DATA_PATH, train=False, download=True, transform=T.ToTensor()
    )

    # Combine the images and targets from train and test into one
    all_images = np.concatenate([train_set.data, test_set.data], axis=0)
    all_targets = torch.LongTensor(train_set.targets + test_set.targets)

    #
    torch.manual_seed(seed)
    classes = torch.randperm(100)
    train_classes, val_classes, test_classes = (
        classes[:80],
        classes[80:90],
        classes[90:100],
    )

    train_set = dataset_from_labels(
        all_images, all_targets, train_classes, img_transform=train_transform
    )

    val_set = dataset_from_labels(
        all_images, all_targets, val_classes, img_transform=test_transform
    )

    test_set = dataset_from_labels(
        all_images, all_targets, test_classes, img_transform=test_transform
    )

    return train_set, val_set, test_set


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    _, _, test_set = get_dataset()
