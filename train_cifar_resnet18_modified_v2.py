#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ResNet-18 on CIFAR-100 using labels from cifar100.csv.

Image root:
    /home/work-base/datasets/cifar100/img/
        train/
        test/

CSV label file:
    /home/work-base/datasets/cifar100/data/cifar100.csv

Expected CSV format (no header required):
    column 0: image path, e.g.
              cifar100/img/train/bos_taurus_s_000507.png
    column 1: class name, e.g.
              cattle

Example:
    cifar100/img/train/bos_taurus_s_000507.png,cattle

The script matches CSV entries using the path after train/ or test/, so the
different root prefix in the CSV does not matter.

If duplicate CSV paths have conflicting labels, the first label is kept
deterministically and all conflicts are written to csv_label_conflicts.csv.

Default settings:
- ResNet18
- weights=None (no pretrained model)
- 100 epochs
- basic augmentation only:
    RandomCrop(32, padding=4)
    RandomHorizontalFlip()
- evaluation on test set every 10 epochs
- logs loss/accuracy/lr/time to train.log and metrics.csv
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on CIFAR-100 with CSV labels."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/work-base/datasets/cifar100/img",
        help="Image root containing train/ and test/.",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="/home/work-base/datasets/cifar100/data/cifar100.csv",
        help="CSV file containing image path and class name.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/work-base/Tolerence-Boundary/outputs/resnet18_cifar100",
        help="Output directory for logs and checkpoints.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv_labels(csv_path):
    """
    Read:
        column 0 -> image path
        column 1 -> class name

    Returns:
        records: list[(csv_image_path, class_name)]
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV label file does not exist: {csv_path}"
        )

    records = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.reader(f)

        for row_id, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            image_path = row[0].strip()
            class_name = row[1].strip()

            if not image_path or not class_name:
                continue

            # Automatically ignore a possible header.
            lower_path = image_path.lower()
            lower_label = class_name.lower()

            if row_id == 1 and (
                "path" in lower_path
                or "image" in lower_path
            ) and (
                "label" in lower_label
                or "class" in lower_label
            ):
                continue

            records.append(
                (image_path, class_name)
            )

    if not records:
        raise RuntimeError(
            f"No valid labels found in {csv_path}"
        )

    return records


def build_label_maps(records):
    """
    Build a stable alphabetical mapping:
        class_name -> class_index
    """
    classes = sorted(
        {class_name for _, class_name in records}
    )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    return classes, class_to_idx


def build_split_label_map(records, split):
    """
    Build:
        relative_path_inside_split -> class_name

    Expected CSV row:
        cifar100/img/train/bos_taurus_s_000507.png,cattle

    Notes
    -----
    Some exported CIFAR CSV files may contain duplicate image paths.
    If the same path appears more than once with different labels, this
    function keeps the FIRST label deterministically and records the conflict
    instead of stopping training.

    Returns
    -------
    label_map : dict
        relative image path -> class name
    conflicts : list[dict]
        conflicting duplicate-label records
    duplicates : int
        number of duplicate rows with the same path and same label
    """
    split = split.lower()
    label_map = {}
    conflicts = []
    duplicates = 0

    for row_idx, (image_path, class_name) in enumerate(records, start=1):
        normalized = image_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")
        lower_parts = [p.lower() for p in parts]

        if split not in lower_parts:
            continue

        split_idx = lower_parts.index(split)

        if split_idx >= len(parts) - 1:
            continue

        relative_path = "/".join(parts[split_idx + 1:])

        if relative_path in label_map:
            old_label = label_map[relative_path]

            if old_label == class_name:
                duplicates += 1
                continue

            conflicts.append({
                "row": row_idx,
                "split": split,
                "relative_path": relative_path,
                "kept_label": old_label,
                "ignored_label": class_name,
                "csv_image_path": image_path,
            })

            # Keep the first occurrence deterministically.
            continue

        label_map[relative_path] = class_name

    if not label_map:
        raise RuntimeError(
            f"No '{split}' samples found in CSV labels."
        )

    return label_map, conflicts, duplicates


class CIFARCSVImageDataset(Dataset):
    def __init__(
        self,
        image_root,
        split,
        label_map,
        class_to_idx,
        transform=None,
    ):
        self.image_root = Path(image_root) / split
        self.split = split
        self.label_map = label_map
        self.class_to_idx = class_to_idx
        self.transform = transform

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_root}"
            )

        image_paths = sorted([
            p
            for p in self.image_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not image_paths:
            raise RuntimeError(
                f"No images found under: {self.image_root}"
            )

        self.samples = []
        missing_labels = []

        for image_path in image_paths:
            relative_path = image_path.relative_to(self.image_root).as_posix()

            if relative_path not in label_map:
                missing_labels.append(str(image_path))
                continue

            class_name = label_map[relative_path]

            if class_name not in class_to_idx:
                raise RuntimeError(
                    f"Unknown class '{class_name}' for {image_path}"
                )

            target = class_to_idx[class_name]

            self.samples.append(
                (image_path, target)
            )

        if missing_labels:
            preview = "\n".join(missing_labels[:10])
            raise RuntimeError(
                f"{len(missing_labels)} images under '{split}' "
                f"do not have labels in the CSV.\n"
                f"Examples:\n{preview}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found for split '{split}'."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_dataloaders(
    data_root,
    csv_path,
    batch_size,
    workers,
):
    records = load_csv_labels(csv_path)

    classes, class_to_idx = build_label_maps(records)

    train_label_map, train_conflicts, train_duplicates = build_split_label_map(
        records,
        split="train",
    )

    test_label_map, test_conflicts, test_duplicates = build_split_label_map(
        records,
        split="test",
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR100_MEAN,
            CIFAR100_STD,
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR100_MEAN,
            CIFAR100_STD,
        ),
    ])

    train_dataset = CIFARCSVImageDataset(
        image_root=data_root,
        split="train",
        label_map=train_label_map,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    test_dataset = CIFARCSVImageDataset(
        image_root=data_root,
        split="test",
        label_map=test_label_map,
        class_to_idx=class_to_idx,
        transform=test_transform,
    )

    print(
        f"[CSV check] train labels={len(train_label_map)}, "
        f"test labels={len(test_label_map)}, "
        f"classes={len(classes)}"
    )
    print(
        f"[CSV check] train duplicate rows={train_duplicates}, "
        f"train conflicting rows={len(train_conflicts)}"
    )
    print(
        f"[CSV check] test duplicate rows={test_duplicates}, "
        f"test conflicting rows={len(test_conflicts)}"
    )

    if train_conflicts:
        print(
            "[Warning] Conflicting TRAIN labels were found in the CSV. "
            "The first label for each duplicated image path is kept."
        )
        for item in train_conflicts[:10]:
            print(
                f"  {item['relative_path']}: "
                f"keep='{item['kept_label']}', "
                f"ignore='{item['ignored_label']}'"
            )

    if test_conflicts:
        print(
            "[Warning] Conflicting TEST labels were found in the CSV. "
            "The first label for each duplicated image path is kept."
        )
        for item in test_conflicts[:10]:
            print(
                f"  {item['relative_path']}: "
                f"keep='{item['kept_label']}', "
                f"ignore='{item['ignored_label']}'"
            )

    if len(classes) != 100:
        print(
            f"[Warning] Expected 100 CIFAR-100 classes, "
            f"but found {len(classes)} unique labels in CSV."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    csv_diagnostics = {
        "train_conflicts": train_conflicts,
        "test_conflicts": test_conflicts,
        "train_duplicates": train_duplicates,
        "test_duplicates": test_duplicates,
    }

    return (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
        classes,
        class_to_idx,
        csv_diagnostics,
    )


def build_model(num_classes):
    model = models.resnet18(
        weights=None
    )

    # CIFAR 32x32 stem.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        dynamic_ncols=True,
    )

    for images, targets in pbar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            logits.argmax(dim=1)
            == targets
        ).sum().item()

        total_samples += batch_size

        avg_loss = (
            total_loss / total_samples
        )

        avg_acc = (
            100.0
            * total_correct
            / total_samples
        )

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}%",
        )

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Test Epoch {epoch}",
        dynamic_ncols=True,
    )

    for images, targets in pbar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            logits.argmax(dim=1)
            == targets
        ).sum().item()

        total_samples += batch_size

        avg_loss = (
            total_loss / total_samples
        )

        avg_acc = (
            100.0
            * total_correct
            / total_samples
        )

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}%",
        )

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    args,
    classes,
    class_to_idx,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "args": vars(args),
        },
        path,
    )


def append_csv(csv_path, row):
    file_exists = csv_path.exists()

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "epoch_time_sec",
    ]

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_text_log(
    log_path,
    message,
):
    with log_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(message + "\n")


def main():
    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_log_path = (
        output_dir / "metrics.csv"
    )

    text_log_path = (
        output_dir / "train.log"
    )

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=4,
            ensure_ascii=False,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
        classes,
        class_to_idx,
        csv_diagnostics,
    ) = build_dataloaders(
        data_root=args.data_root,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    num_classes = len(classes)

    model = build_model(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    header = (
        f"Device        : {device}\n"
        f"Data root     : {args.data_root}\n"
        f"CSV labels    : {args.csv_path}\n"
        f"Output dir    : {args.output_dir}\n"
        f"Train samples : {len(train_dataset)}\n"
        f"Test samples  : {len(test_dataset)}\n"
        f"Classes       : {num_classes}\n"
        f"Epochs        : {args.epochs}\n"
        f"Batch size    : {args.batch_size}\n"
        f"Initial LR    : {args.lr}\n"
        f"Eval interval : {args.eval_interval}\n"
        f"Pretrained    : False"
    )

    print("=" * 70)
    print(header)
    print("=" * 70)

    append_text_log(
        text_log_path,
        "=" * 70,
    )

    append_text_log(
        text_log_path,
        header,
    )

    append_text_log(
        text_log_path,
        "=" * 70,
    )

    # Save class mapping for later inference/evaluation.
    with (
        output_dir / "class_to_idx.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class_to_idx,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # Persist CSV diagnostics so label conflicts are auditable.
    with (
        output_dir / "csv_diagnostics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            csv_diagnostics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    conflict_csv = output_dir / "csv_label_conflicts.csv"
    all_conflicts = (
        csv_diagnostics["train_conflicts"]
        + csv_diagnostics["test_conflicts"]
    )

    if all_conflicts:
        with conflict_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            fieldnames = [
                "row",
                "split",
                "relative_path",
                "kept_label",
                "ignored_label",
                "csv_image_path",
            ]
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(all_conflicts)

    best_acc = 0.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_loss = ""
        test_acc = ""

        should_eval = (
            epoch % args.eval_interval == 0
            or epoch == args.epochs
        )

        if should_eval:
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
            )

            test_loss = test_metrics["loss"]
            test_acc = test_metrics["acc"]

            if test_acc > best_acc:
                best_acc = test_acc

                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_acc,
                    args,
                    classes,
                    class_to_idx,
                )

        scheduler.step()

        epoch_time = (
            time.time() - epoch_start
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_sec": epoch_time,
        }

        append_csv(
            csv_log_path,
            row,
        )

        if should_eval:
            log_line = (
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['acc']:.2f}% | "
                f"test_loss={test_loss:.4f} | "
                f"test_acc={test_acc:.2f}% | "
                f"best_acc={best_acc:.2f}% | "
                f"time={epoch_time:.1f}s"
            )
        else:
            log_line = (
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['acc']:.2f}% | "
                f"test=skipped | "
                f"time={epoch_time:.1f}s"
            )

        print(log_line)

        append_text_log(
            text_log_path,
            log_line,
        )

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            args,
            classes,
            class_to_idx,
        )

    final_msg = (
        f"Training finished. "
        f"Best test accuracy: {best_acc:.2f}%"
    )

    print(final_msg)

    append_text_log(
        text_log_path,
        final_msg,
    )


if __name__ == "__main__":
    main()
