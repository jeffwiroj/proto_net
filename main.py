from torch.utils.data import DataLoader, Dataset
import torch, torch.nn as nn

from data import get_dataset, FewShotBatchSampler
from model import ProtoNet
from util import AverageMeter, get_accuracy

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    model = model.to(DEVICE)

    avg_loss = AverageMeter("Loss")
    avg_acc = AverageMeter("Accuracy")

    for batch in dataloader:
        optimizer.zero_grad()

        batch = [e.to(DEVICE) for e in batch]
        preds, labels = model(batch)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        avg_loss.update(loss.item())
        avg_acc.update(get_accuracy(preds, labels))
    return avg_loss.avg, avg_acc.avg


def evaluate(model, dataloader):
    model.eval()
    avg_acc = AverageMeter("Accuracy")
    with torch.inference_mode():
        for batch in dataloader:
            batch = [e.to(DEVICE) for e in batch]
            preds, labels = model(batch)
            avg_acc.update(get_accuracy(preds, labels))

    model.train()
    return avg_acc.avg


if __name__ == "__main__":
    train_set, val_set, _ = get_dataset()

    train_loader = DataLoader(
        train_set,
        batch_sampler=FewShotBatchSampler(
            train_set.targets,
            include_query=True,
            N=5,
            K=4,
            shuffle=True,
        ),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=FewShotBatchSampler(
            val_set.targets,
            include_query=True,
            N=5,
            K=4,
            shuffle=True,
        ),
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    model = ProtoNet(200)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for epoch in range(64):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch + 1
        )
        val_acc = evaluate(model, val_loader)
        print(f"Epoch: {epoch}, loss: {tr_loss}, acc:{tr_acc}, val_acc: {val_acc}")
