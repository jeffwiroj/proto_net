import math

# ML + CV related packages
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torchvision
from data import split_batch


def get_resnet(output_size):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(1000,output_size)
    return model


def get_densenet(output_size):
    convnet = torchvision.models.DenseNet(
        growth_rate=32,
        block_config=(6, 6, 6, 6),
        bn_size=2,
        num_init_features=64,
        num_classes=output_size,  
    )
    return convnet


class ProtoNet(nn.Module):
    def __init__(self,model_type="resnet",proto_dim:int=100):
        super().__init__()
        if model_type == "resnet":
            self.model = get_resnet(proto_dim)
        else:
            self.model= get_densenet(proto_dim)

    @staticmethod
    def calculate_prototypes(features, targets):
        classes, _ = torch.unique(targets).sort()
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(0)
            prototypes.append(p)
        return torch.stack(prototypes, dim=0), classes

    def classify_feats(self, prototypes, classes, feats, targets):
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        return preds, labels

    def forward(self, batch):
        imgs, targets = batch
        features = self.model(imgs)
        support_feats, query_feats, support_targets, query_targets = split_batch(
            features, targets
        )
        prototypes, classes = ProtoNet.calculate_prototypes(
            support_feats, support_targets
        )
        preds, labels = self.classify_feats(
            prototypes, classes, query_feats, query_targets
        )
        return preds, labels
