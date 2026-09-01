#!/usr/bin/env python3
"""Train ResNet-18 on clean and low-quality CIFAR-100 or ImageNet-100 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from low_quality import DEGRADATION_TYPES, collect_images

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cifar100", "imagenet100"))
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing train/ and test/ (CIFAR) or train/ and val/ (ImageNet).",
    )
    parser.add_argument("--low-quality-root", type=Path, required=True)
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Optional CIFAR CSV: first column image path, second column class name.",
    )
    parser.add_argument("--val-split", choices=("test", "val"))

    parser.add_argument("--clean-ratio", type=float, default=100.0)
    parser.add_argument("--gaussian-blur-ratio", type=float, default=0.0)
    parser.add_argument("--gaussian-noise-ratio", type=float, default=0.0)
    parser.add_argument("--jpeg-compression-ratio", type=float, default=0.0)
    parser.add_argument(
        "--blur-level", choices=("mild", "moderate", "severe"), default="severe"
    )
    parser.add_argument(
        "--noise-level", choices=("mild", "moderate", "severe"), default="severe"
    )
    parser.add_argument(
        "--jpeg-level", choices=("mild", "moderate", "severe"), default="severe"
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/low_quality"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and write the manifest only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_config(args) -> dict[str, float]:
    ratios = {
        "clean": args.clean_ratio,
        "gaussian_blur": args.gaussian_blur_ratio,
        "gaussian_noise": args.gaussian_noise_ratio,
        "jpeg_compression": args.jpeg_compression_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("All ratios must be non-negative.")
    if not math.isclose(sum(ratios.values()), 100.0, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 100, got {sum(ratios.values()):g}.")
    return ratios


def exact_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation, so counts always add up to dataset size."""
    raw = {name: total * ratio / 100.0 for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def load_csv_labels(csv_path: Path, split: str) -> dict[str, str]:
    labels = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 2:
                continue
            raw_path, class_name = row[0].strip(), row[1].strip()
            if not raw_path or not class_name:
                continue
            if (
                row_number == 1
                and ("path" in raw_path.lower() or "image" in raw_path.lower())
                and ("class" in class_name.lower() or "label" in class_name.lower())
            ):
                continue
            parts = raw_path.replace("\\", "/").strip("/").split("/")
            lowered = [part.lower() for part in parts]
            if split.lower() not in lowered:
                continue
            relative = "/".join(parts[lowered.index(split.lower()) + 1 :])
            previous = labels.setdefault(relative, class_name)
            if previous != class_name:
                raise ValueError(
                    f"Conflicting CSV labels for {relative}: {previous}, {class_name}"
                )
    if not labels:
        raise RuntimeError(f"No {split!r} labels found in {csv_path}")
    return labels


def infer_folder_labels(root: Path) -> dict[str, str]:
    labels = {}
    for path in collect_images(root):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise RuntimeError(
                f"Cannot infer class for {relative}; use class subdirectories or --csv-path."
            )
        labels[relative.as_posix()] = relative.parts[0]
    return labels


def labels_for_split(root: Path, split: str, csv_path: Path | None) -> dict[str, str]:
    return load_csv_labels(csv_path, split) if csv_path else infer_folder_labels(root)


class MixedQualityDataset(Dataset):
    def __init__(
        self,
        clean_root: Path,
        low_quality_root: Path,
        labels: dict[str, str],
        class_to_idx: dict[str, int],
        ratios: dict[str, float],
        levels: dict[str, str],
        seed: int,
        transform,
    ):
        self.transform = transform
        clean_paths = collect_images(clean_root)
        if not clean_paths:
            raise RuntimeError(f"No training images found under {clean_root}")

        self.samples = []
        for clean_path in clean_paths:
            relative = clean_path.relative_to(clean_root).as_posix()
            if relative not in labels:
                raise RuntimeError(f"Missing label for training image: {relative}")
            class_name = labels[relative]
            if class_name not in class_to_idx:
                raise RuntimeError(f"Unknown class {class_name!r} for {relative}")
            self.samples.append(
                {
                    "relative_path": relative,
                    "clean_path": clean_path,
                    "target": class_to_idx[class_name],
                }
            )

        counts = exact_counts(len(self.samples), ratios)
        indices = list(range(len(self.samples)))
        random.Random(seed).shuffle(indices)
        self.assignment = {}
        cursor = 0
        for source in ("clean",) + DEGRADATION_TYPES:
            for index in indices[cursor : cursor + counts[source]]:
                self.assignment[index] = source
            cursor += counts[source]

        self.paths = []
        missing = []
        for index, sample in enumerate(self.samples):
            source = self.assignment[index]
            if source == "clean":
                path = sample["clean_path"]
            else:
                path = (
                    low_quality_root
                    / source
                    / levels[source]
                    / "train"
                    / sample["relative_path"]
                )
            if not path.is_file():
                missing.append(path)
            self.paths.append(path)
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} assigned image(s) are missing. First paths:\n{preview}"
            )
        self.source_counts = dict(Counter(self.assignment.values()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.samples[index]["target"]


class EvaluationDataset(Dataset):
    def __init__(self, root: Path, labels, class_to_idx, transform):
        self.transform = transform
        self.samples = []
        for path in collect_images(root):
            relative = path.relative_to(root).as_posix()
            if relative not in labels:
                raise RuntimeError(f"Missing label for evaluation image: {relative}")
            class_name = labels[relative]
            if class_name not in class_to_idx:
                raise RuntimeError(
                    f"Evaluation-only class {class_name!r} in {relative}"
                )
            self.samples.append((path, class_to_idx[class_name]))
        if not self.samples:
            raise RuntimeError(f"No evaluation images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def build_transforms(dataset: str):
    if dataset == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return train_transform, eval_transform


def build_data(args):
    train_root = args.data_root / "train"
    val_split = args.val_split or ("test" if args.dataset == "cifar100" else "val")
    val_root = args.data_root / val_split
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(f"Expected {train_root} and {val_root}")

    train_labels = labels_for_split(train_root, "train", args.csv_path)
    val_labels = labels_for_split(val_root, val_split, args.csv_path)
    classes = sorted(set(train_labels.values()))
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    train_transform, eval_transform = build_transforms(args.dataset)
    levels = {
        "gaussian_blur": args.blur_level,
        "gaussian_noise": args.noise_level,
        "jpeg_compression": args.jpeg_level,
    }
    train_dataset = MixedQualityDataset(
        train_root,
        args.low_quality_root,
        train_labels,
        class_to_idx,
        ratio_config(args),
        levels,
        args.seed,
        train_transform,
    )
    val_dataset = EvaluationDataset(val_root, val_labels, class_to_idx, eval_transform)
    return train_dataset, val_dataset, classes, class_to_idx, val_split


def build_model(dataset: str, num_classes: int):
    model = models.resnet18(weights=None)
    if dataset == "cifar100":
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def choose_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, dynamic_ncols=True)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(1) == targets).sum().item()
            total += batch_size
            progress.set_postfix(
                loss=f"{loss_sum / total:.4f}", acc=f"{100 * correct / total:.2f}%"
            )
    return loss_sum / total, 100.0 * correct / total


def save_manifest(path: Path, dataset: MixedQualityDataset, levels: dict[str, str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "source", "level", "resolved_path"])
        for index, sample in enumerate(dataset.samples):
            source = dataset.assignment[index]
            writer.writerow(
                [
                    sample["relative_path"],
                    source,
                    "" if source == "clean" else levels[source],
                    dataset.paths[index],
                ]
            )


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be at least 1.")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError(
            "--batch-size must be positive and --workers must be non-negative."
        )
    set_seed(args.seed)
    ratios = ratio_config(args)
    levels = {
        "gaussian_blur": args.blur_level,
        "gaussian_noise": args.noise_level,
        "jpeg_compression": args.jpeg_level,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, classes, class_to_idx, val_split = build_data(args)
    save_manifest(args.output_dir / "sample_manifest.csv", train_dataset, levels)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        config = vars(args).copy()
        config.update(
            {
                key: str(value)
                for key, value in config.items()
                if isinstance(value, Path)
            }
        )
        json.dump(config, handle, indent=2)
    with (args.output_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_samples": len(train_dataset),
                "evaluation_samples": len(val_dataset),
                "evaluation_split": val_split,
                "num_classes": len(classes),
                "requested_ratios": ratios,
                "actual_counts": train_dataset.source_counts,
                "class_to_idx": class_to_idx,
            },
            handle,
            indent=2,
        )
    print(f"Train/eval/classes: {len(train_dataset)}/{len(val_dataset)}/{len(classes)}")
    print(f"Assigned sources: {train_dataset.source_counts}")
    if args.dry_run:
        print("Dry run complete; no training was started.")
        return

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.dataset, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "eval_loss",
                "eval_acc",
                "seconds",
            ]
        )

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            f"Train {epoch}/{args.epochs}",
        )
        eval_loss = eval_acc = None
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_loss, eval_acc = run_epoch(
                model, val_loader, criterion, device, desc=f"Eval {epoch}"
            )
        scheduler.step()
        is_best = eval_acc is not None and eval_acc > best_acc
        if is_best:
            best_acc = eval_acc
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
            "args": vars(args),
        }
        torch.save(checkpoint, args.output_dir / "last.pth")
        if is_best:
            torch.save(checkpoint, args.output_dir / "best.pth")
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    lr,
                    train_loss,
                    train_acc,
                    eval_loss,
                    eval_acc,
                    time.time() - started,
                ]
            )
        message = f"[{epoch:03d}/{args.epochs}] train={train_acc:.2f}%"
        if eval_acc is not None:
            message += f" eval={eval_acc:.2f}% best={best_acc:.2f}%"
        print(message)


if __name__ == "__main__":
    main()
