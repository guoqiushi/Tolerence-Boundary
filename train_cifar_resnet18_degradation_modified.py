#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ResNet-18 on CIFAR-100 with controllable degradation replacement.

Original:
    /home/work-base/datasets/cifar100/img/train
    /home/work-base/datasets/cifar100/img/test

CSV:
    /home/work-base/datasets/cifar100/data/cifar100.csv

Degradation:
    /home/work-base/datasets/cifar100/degradtion/
        gaussian_noise/<level>/train/
        gaussian_blur/<level>/train/
        jpeg_compression/<level>/train/

Examples:
    # 10% severe gaussian noise
    python train_cifar_resnet18_degradation_modified.py \
        --degrade-type gaussian_noise \
        --level severe \
        --degrade-ratio 10

    # 30% moderate jpeg compression
    python train_cifar_resnet18_degradation_modified.py \
        --degrade-type jpeg_compression \
        --level moderate \
        --degrade-ratio 30

    # 30% severe mixed degradation
    python train_cifar_resnet18_degradation_modified.py \
        --degrade-type mixed \
        --level severe \
        --degrade-ratio 30
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

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEGRADATION_TYPES = [
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/work-base/datasets/cifar100/img",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="/home/work-base/datasets/cifar100/data/cifar100.csv",
    )

    parser.add_argument(
        "--degradation-root",
        type=str,
        default="/home/work-base/datasets/cifar100/degradtion",
    )

    parser.add_argument(
        "--degrade-type",
        type=str,
        default="gaussian_noise",
        choices=[
            "gaussian_noise",
            "gaussian_blur",
            "jpeg_compression",
            "mixed",
        ],
    )

    parser.add_argument(
        "--level",
        type=str,
        default="severe",
        choices=["mild", "moderate", "severe"],
    )

    parser.add_argument(
        "--degrade-ratio",
        type=float,
        default=10.0,
        help="10 means 10% degraded + 90% original.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/work-base/Tolerence-Boundary/outputs/resnet18_cifar100_degradation",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv_labels(csv_path):
    records = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if len(row) < 2:
                continue

            image_path = row[0].strip()
            class_name = row[1].strip()

            if not image_path or not class_name:
                continue

            if row_idx == 0:
                low0 = image_path.lower()
                low1 = class_name.lower()

                if (
                    ("path" in low0 or "image" in low0)
                    and ("label" in low1 or "class" in low1)
                ):
                    continue

            records.append((image_path, class_name))

    if not records:
        raise RuntimeError("No valid labels found in CSV.")

    return records


def build_label_map(records, split):
    """
    CSV example:
        cifar100/img/train/bos_taurus_s_000507.png,cattle

    Becomes:
        bos_taurus_s_000507.png -> cattle

    For conflicting duplicates, keep the first label.
    """
    label_map = {}
    conflicts = []

    for image_path, class_name in records:
        normalized = image_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")
        lower = [p.lower() for p in parts]

        if split not in lower:
            continue

        idx = lower.index(split)

        if idx >= len(parts) - 1:
            continue

        relative_path = "/".join(parts[idx + 1:])

        if relative_path in label_map:
            if label_map[relative_path] != class_name:
                conflicts.append({
                    "relative_path": relative_path,
                    "kept_label": label_map[relative_path],
                    "ignored_label": class_name,
                })
            continue

        label_map[relative_path] = class_name

    return label_map, conflicts


def collect_images(root):
    root = Path(root)

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class TrainDataset(Dataset):
    def __init__(
        self,
        original_root,
        degradation_root,
        degrade_type,
        level,
        degrade_ratio,
        label_map,
        class_to_idx,
        seed,
        transform,
    ):
        self.original_root = Path(original_root)
        self.degradation_root = Path(degradation_root)
        self.degrade_type = degrade_type
        self.level = level
        self.transform = transform

        if not 0 <= degrade_ratio <= 100:
            raise ValueError("--degrade-ratio must be in [0, 100].")

        if degrade_type == "mixed":
            self.active_types = DEGRADATION_TYPES
        else:
            self.active_types = [degrade_type]

        self.type_roots = {
            dtype: (
                self.degradation_root
                / dtype
                / level
                / "train"
            )
            for dtype in self.active_types
        }

        if degrade_ratio > 0:
            for dtype, root in self.type_roots.items():
                if not root.exists():
                    raise FileNotFoundError(
                        f"Missing degradation directory: {root}"
                    )

        original_images = collect_images(self.original_root)

        if not original_images:
            raise RuntimeError(
                f"No training images found: {self.original_root}"
            )

        self.samples = []

        for image_path in original_images:
            relative_path = image_path.relative_to(
                self.original_root
            ).as_posix()

            if relative_path not in label_map:
                raise RuntimeError(
                    f"Missing CSV label for: {relative_path}"
                )

            class_name = label_map[relative_path]
            target = class_to_idx[class_name]

            degraded_paths = {
                dtype: self.type_roots[dtype] / relative_path
                for dtype in self.active_types
            }

            self.samples.append({
                "relative_path": relative_path,
                "original_path": image_path,
                "degraded_paths": degraded_paths,
                "target": target,
            })

        requested_num = round(
            len(self.samples) * degrade_ratio / 100.0
        )

        # For mixed mode, require all 3 degraded counterparts to exist.
        valid_indices = []

        for idx, sample in enumerate(self.samples):
            if all(
                sample["degraded_paths"][dtype].exists()
                for dtype in self.active_types
            ):
                valid_indices.append(idx)

        if requested_num > len(valid_indices):
            raise RuntimeError(
                f"Requested {requested_num} degraded samples, "
                f"but only {len(valid_indices)} valid degraded counterparts exist."
            )

        rng = random.Random(seed)

        selected = rng.sample(
            valid_indices,
            requested_num,
        )

        self.assignment = {}

        if degrade_type == "mixed":
            rng.shuffle(selected)

            for pos, idx in enumerate(selected):
                dtype = DEGRADATION_TYPES[
                    pos % len(DEGRADATION_TYPES)
                ]
                self.assignment[idx] = dtype
        else:
            for idx in selected:
                self.assignment[idx] = degrade_type

        self.num_degraded = len(self.assignment)
        self.num_original = len(self.samples) - self.num_degraded

        self.type_counts = {
            dtype: 0
            for dtype in DEGRADATION_TYPES
        }

        for dtype in self.assignment.values():
            self.type_counts[dtype] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if idx in self.assignment:
            dtype = self.assignment[idx]
            image_path = sample["degraded_paths"][dtype]
        else:
            image_path = sample["original_path"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, sample["target"]


class TestDataset(Dataset):
    def __init__(
        self,
        root,
        label_map,
        class_to_idx,
        transform,
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for image_path in collect_images(self.root):
            relative_path = image_path.relative_to(
                self.root
            ).as_posix()

            if relative_path not in label_map:
                raise RuntimeError(
                    f"Missing CSV test label for: {relative_path}"
                )

            class_name = label_map[relative_path]
            target = class_to_idx[class_name]

            self.samples.append(
                (image_path, target)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


def build_dataloaders(args):
    records = load_csv_labels(args.csv_path)

    classes = sorted(
        {label for _, label in records}
    )

    class_to_idx = {
        name: idx
        for idx, name in enumerate(classes)
    }

    train_labels, train_conflicts = build_label_map(
        records,
        "train",
    )

    test_labels, test_conflicts = build_label_map(
        records,
        "test",
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
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

    train_dataset = TrainDataset(
        original_root=Path(args.data_root) / "train",
        degradation_root=args.degradation_root,
        degrade_type=args.degrade_type,
        level=args.level,
        degrade_ratio=args.degrade_ratio,
        label_map=train_labels,
        class_to_idx=class_to_idx,
        seed=args.seed,
        transform=train_transform,
    )

    test_dataset = TestDataset(
        root=Path(args.data_root) / "test",
        label_map=test_labels,
        class_to_idx=class_to_idx,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    diagnostics = {
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "classes": len(classes),
        "original_train_samples": train_dataset.num_original,
        "degraded_train_samples": train_dataset.num_degraded,
        "degradation_counts": train_dataset.type_counts,
        "train_csv_conflicts": train_conflicts,
        "test_csv_conflicts": test_conflicts,
    }

    return (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
        classes,
        class_to_idx,
        diagnostics,
    )


def build_model(num_classes):
    model = models.resnet18(weights=None)

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
    total_num = 0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}",
        dynamic_ncols=True,
    )

    for images, labels in pbar:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        bs = labels.size(0)

        total_loss += loss.item() * bs
        total_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total_num += bs

        pbar.set_postfix(
            loss=f"{total_loss / total_num:.4f}",
            acc=f"{100 * total_correct / total_num:.2f}%",
        )

    return (
        total_loss / total_num,
        100.0 * total_correct / total_num,
    )


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
    total_num = 0

    for images, labels in tqdm(
        loader,
        desc=f"Test {epoch}",
        dynamic_ncols=True,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = labels.size(0)

        total_loss += loss.item() * bs
        total_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total_num += bs

    return (
        total_loss / total_num,
        100.0 * total_correct / total_num,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()

    set_seed(args.seed)

    experiment_name = (
        f"{args.degrade_type}_"
        f"{args.level}_"
        f"ratio_{args.degrade_ratio:g}"
    )

    output_dir = (
        Path(args.output_dir)
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        diagnostics,
    ) = build_dataloaders(args)

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        output_dir / "data_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            diagnostics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        output_dir / "class_to_idx.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class_to_idx,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # Save exact replacement list.
    with open(
        output_dir / "degraded_samples.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "relative_path",
            "source",
            "degradation_type",
            "level",
        ])

        for idx, sample in enumerate(
            train_dataset.samples
        ):
            if idx in train_dataset.assignment:
                dtype = train_dataset.assignment[idx]

                writer.writerow([
                    sample["relative_path"],
                    "degraded",
                    dtype,
                    args.level,
                ])
            else:
                writer.writerow([
                    sample["relative_path"],
                    "original",
                    "",
                    "",
                ])

    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Degrade type     : {args.degrade_type}")
    print(f"Level            : {args.level}")
    print(f"Degrade ratio    : {args.degrade_ratio}%")
    print(f"Train total      : {len(train_dataset)}")
    print(f"Original         : {train_dataset.num_original}")
    print(f"Degraded         : {train_dataset.num_degraded}")
    print(
        f"Noise / Blur / JPEG : "
        f"{train_dataset.type_counts['gaussian_noise']} / "
        f"{train_dataset.type_counts['gaussian_blur']} / "
        f"{train_dataset.type_counts['jpeg_compression']}"
    )
    print(f"Test total       : {len(test_dataset)}")
    print(f"Classes          : {len(classes)}")
    print(f"Output           : {output_dir}")
    print("=" * 70)

    model = build_model(
        len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    metrics_file = output_dir / "metrics.csv"
    log_file = output_dir / "train.log"

    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "epoch_time_sec",
        ])

    best_acc = 0.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        start_time = time.time()

        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )

        test_loss = ""
        test_acc = ""

        do_eval = (
            epoch % args.eval_interval == 0
            or epoch == args.epochs
        )

        if do_eval:
            test_loss, test_acc = evaluate(
                model,
                test_loader,
                criterion,
                device,
                epoch,
            )

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
                )

        scheduler.step()

        elapsed = time.time() - start_time

        with open(
            metrics_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                lr,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                elapsed,
            ])

        if do_eval:
            msg = (
                f"[{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}% "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.2f}% "
                f"best={best_acc:.2f}%"
            )
        else:
            msg = (
                f"[{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}%"
            )

        print(msg)

        with open(
            log_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(msg + "\n")

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            args,
        )

    print(
        f"Finished. Best test accuracy: {best_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
