import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def make_dataloaders(data_dir, batch_size=16, num_workers=4, seed=42):
    train_transform, eval_transform = get_transforms()

    # ImageFolder는 폴더명을 class 이름으로 자동 인식한다.
    full_dataset = datasets.ImageFolder(root=data_dir)

    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # random_split으로 나눈 subset은 원본 dataset을 공유한다.
    # split별 transform을 다르게 쓰기 위해 dataset을 따로 만들어서 indices만 공유한다.
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    train_subset.dataset = train_dataset
    val_subset.dataset = val_dataset
    test_subset.dataset = test_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes, class_names


def build_resnet34(num_classes, pretrained=True):
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def set_finetuning_mode(model, mode):
    """
    mode:
        head_only       : fc layer만 학습
        layer4          : layer4 + fc 학습
        layer3_layer4   : layer3 + layer4 + fc 학습
        all             : 전체 layer 학습
    """

    for param in model.parameters():
        param.requires_grad = False

    if mode == "head_only":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "layer3_layer4":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "all":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]

"""
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc
"""
def top5_accuracy(outputs, labels):
    """
    outputs: model logits, shape = [batch_size, num_classes]
    labels : ground-truth labels, shape = [batch_size]
    """
    _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))

    top5_correct = correct.any(dim=1).float().sum().item()
    batch_size = labels.size(0)

    return top5_correct / batch_size

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    top5_correct = 0.0
    total_samples = 0

    if epoch is not None and total_epochs is not None:
        desc = f"Train Epoch {epoch}/{total_epochs}"
    else:
        desc = "Train"

    progress_bar = tqdm(
        dataloader,
        desc=desc,
        leave=True,
        ncols=120
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        current_loss = running_loss / len(all_labels)
        current_acc = accuracy_score(all_labels, all_preds)
        top5_correct += top5_accuracy(outputs, labels) * labels.size(0)
        total_samples += labels.size(0)
        current_top5=top5_correct / total_samples if total_samples > 0 else 0

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}",
            "top5_acc": f"{current_top5:.4f}",
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, current_top5

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    top5_correct = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        top5_correct += top5_accuracy(outputs, labels) * labels.size(0)
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics = {
        "loss": epoch_loss,
        "accuracy": acc,
        "top5_accuracy": top5_correct / total_samples if total_samples > 0 else 0,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    return metrics, all_labels, all_preds


def plot_learning_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train accuracy")
    plt.plot(epochs, history["val_acc"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / "accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_experiment(
    mode,
    data_dir,
    result_dir,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    num_workers=4,
    seed=42,
    pretrained=True,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes, class_names = make_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = build_resnet34(num_classes=num_classes, pretrained=pretrained)
    set_finetuning_mode(model, mode)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        get_trainable_parameters(model),
        lr=lr,
        weight_decay=1e-4,
    )

    save_path = Path(result_dir) / mode
    save_path.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5_acc": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
    }

    best_val_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n========== Experiment: {mode} ==========")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(epochs):
        train_loss, train_acc, train_top5_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch + 1,
            total_epochs=epochs,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_top5_acc"].append(val_metrics["top5_accuracy"])

        print(
            f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "mode": mode,
            "num_classes": num_classes,
        },
        save_path / "best_model.pth",
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(save_path / "history.csv", index=False)

    plot_learning_curve(history, save_path)

    test_metrics, test_labels, test_preds = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        zero_division=0,
    )

    with open(save_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nTest Result")
    print(test_metrics)
    print(report)

    result = {
        "mode": mode,
        "pretrained": pretrained,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "best_val_f1_macro": best_val_f1,
    }

    return result


def main():
    print("Starting Pokémon Training Experiment...")
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    experiment_settings = [
        {
            "mode": "head_only",
            "pretrained": True,
            "lr": args.lr,
        },
        {
            "mode": "layer4",
            "pretrained": True,
            "lr": args.lr,
        },
        {
            "mode": "layer3_layer4",
            "pretrained": True,
            "lr": args.lr,
        },
        {
            "mode": "all",
            "pretrained": False,
            "lr": args.lr,
        },
    ]

    all_results = []

    for exp in experiment_settings:
        result = run_experiment(
            mode=exp["mode"],
            pretrained=exp["pretrained"],
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=exp["lr"],
            num_workers=args.num_workers,
            seed=args.seed,
        )
        all_results.append(result)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(Path(args.result_dir) / "summary_results.csv", index=False)

    print("\n========== Summary ==========")
    print(result_df)


if __name__ == "__main__":
    main()