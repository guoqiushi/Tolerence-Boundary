#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DATASET_ROOT = Path("/home/work-base/datasets/cifar100")
INPUT_ROOT = DATASET_ROOT / "img" / "train"
OUTPUT_ROOT = DATASET_ROOT / "degradtion"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


DEGRADE_CONFIG = {
    "gaussian_noise": {
        "mild": {"sigma": 10.0},
        "moderate": {"sigma": 25.0},
        "severe": {"sigma": 50.0},
    },

    "gaussian_blur": {
        "mild": {
            "kernel_size": 3,
            "sigma": 0.5,
        },
        "moderate": {
            "kernel_size": 5,
            "sigma": 1.0,
        },
        "severe": {
            "kernel_size": 9,
            "sigma": 2.0,
        },
    },

    "jpeg_compression": {
        "mild": {"quality": 75},
        "moderate": {"quality": 40},
        "severe": {"quality": 10},
    },
}


def gaussian_noise(image, level):
    sigma = DEGRADE_CONFIG["gaussian_noise"][level]["sigma"]

    image = image.astype(np.float32)

    noise = np.random.normal(
        0.0,
        sigma,
        image.shape
    ).astype(np.float32)

    degraded = image + noise

    return np.clip(
        degraded,
        0,
        255
    ).astype(np.uint8)


def gaussian_blur(image, level):
    config = DEGRADE_CONFIG["gaussian_blur"][level]

    kernel_size = config["kernel_size"]
    sigma = config["sigma"]

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def jpeg_compression(image, level):
    quality = DEGRADE_CONFIG["jpeg_compression"][level]["quality"]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )


def apply_degradation(image, degrade_type, level):
    if degrade_type == "gaussian_noise":
        return gaussian_noise(image, level)

    if degrade_type == "gaussian_blur":
        return gaussian_blur(image, level)

    if degrade_type == "jpeg_compression":
        return jpeg_compression(image, level)

    raise ValueError(
        f"Unsupported degradation type: {degrade_type}"
    )


def collect_images(root):
    return sorted([
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def process_images(degrade_type, level):
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Train directory does not exist: {INPUT_ROOT}"
        )

    output_root = (
        OUTPUT_ROOT
        / degrade_type
        / level
        / "train"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = collect_images(INPUT_ROOT)

    if not image_paths:
        print(f"No images found under: {INPUT_ROOT}")
        return

    print("=" * 60)
    print(f"Input root      : {INPUT_ROOT}")
    print(f"Output root     : {output_root}")
    print(f"Degradation type: {degrade_type}")
    print(f"Level           : {level}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Config          : {DEGRADE_CONFIG[degrade_type][level]}")
    print("=" * 60)

    failed = []

    for image_path in tqdm(
        image_paths,
        desc=f"{degrade_type}-{level}",
    ):
        relative_path = image_path.relative_to(INPUT_ROOT)
        output_path = output_root / relative_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            failed.append(str(image_path))
            continue

        try:
            degraded = apply_degradation(
                image,
                degrade_type,
                level,
            )

            success = cv2.imwrite(
                str(output_path),
                degraded,
            )

            if not success:
                failed.append(str(image_path))

        except Exception as e:
            print(
                f"\nFailed: {image_path}\n"
                f"Reason: {e}"
            )
            failed.append(str(image_path))

    print("\nDone.")
    print(f"Processed: {len(image_paths) - len(failed)}")
    print(f"Failed   : {len(failed)}")
    print(f"Saved to : {output_root}")

    if failed:
        print("\nFailed files:")
        for path in failed[:20]:
            print(path)

        if len(failed) > 20:
            print(f"... and {len(failed) - 20} more.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate degraded CIFAR-100 train images."
    )

    parser.add_argument(
        "-type",
        "--type",
        dest="degrade_type",
        required=True,
        choices=[
            "gaussian_noise",
            "gaussian_blur",
            "jpeg_compression",
        ],
        help="Degradation type.",
    )

    parser.add_argument(
        "-level",
        "--level",
        required=True,
        choices=[
            "mild",
            "moderate",
            "severe",
        ],
        help="Degradation strength.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    process_images(
        args.degrade_type,
        args.level,
    )


if __name__ == "__main__":
    main()
