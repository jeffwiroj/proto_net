import numpy as np
from PIL import Image
from collections import defaultdict

#  Torch
import torch
import torch.utils.data as data

# TorchVision
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


class FewShotBatchSampler:
    def __init__(self, dataset_targets, N, K, include_query=False, shuffle=True):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N
        self.K_shot = K
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.class_indices = {}
        self.class_num_batches = (
            {}
        )  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.class_indices[c] = torch.where(self.dataset_targets == c)[0]
            self.class_num_batches[c] = self.class_indices[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.class_num_batches.values()) // (self.N_way)
        self.class_list = [
            c for c in self.classes for _ in range(self.class_num_batches[c])
        ]

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.class_indices[c].shape[0])
            self.class_indices[c] = self.class_indices[c][perm]

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        for _ in range(self.iterations):
            class_batch = np.random.choice(self.classes, self.N_way, replace=False)
            index_batch = []
            for c in class_batch:
                index_batch.extend(self.class_indices[c][: self.K_shot])
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    _, _, test_set = get_dataset()
    dataloader = data.DataLoader(
        test_set,
        batch_sampler=FewShotBatchSampler(
            test_set.targets,
            include_query=True,
            N=5,
            K=2,
            shuffle=True,
            shuffle_once=True,
        ),
        num_workers=0,
    )
    pdb.set_trace()
