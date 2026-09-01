#!/usr/bin/env python3
"""Shared low-quality image generation utilities."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEGRADATION_TYPES = ("gaussian_blur", "gaussian_noise", "jpeg_compression")
LEVELS = ("mild", "moderate", "severe")

DEGRADATION_CONFIG = {
    "gaussian_blur": {
        "mild": {"radius": 0.5},
        "moderate": {"radius": 1.0},
        "severe": {"radius": 2.0},
    },
    "gaussian_noise": {
        "mild": {"sigma": 10.0},
        "moderate": {"sigma": 25.0},
        "severe": {"sigma": 50.0},
    },
    "jpeg_compression": {
        "mild": {"quality": 75},
        "moderate": {"quality": 40},
        "severe": {"quality": 10},
    },
}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _rng_for(seed: int, relative_path: str, degradation: str, level: str):
    material = f"{seed}\0{relative_path}\0{degradation}\0{level}".encode()
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def degrade_image(
    image: Image.Image,
    degradation: str,
    level: str,
    *,
    rng: np.random.Generator,
) -> Image.Image:
    """Return an RGB degraded copy of ``image``."""
    image = image.convert("RGB")
    config = DEGRADATION_CONFIG[degradation][level]

    if degradation == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=config["radius"]))

    if degradation == "gaussian_noise":
        array = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, config["sigma"], size=array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))

    if degradation == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=config["quality"], subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    raise ValueError(f"Unsupported degradation type: {degradation}")


def _save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    save_kwargs = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95, "subsampling": 0}
    image.save(output_path, **save_kwargs)


def generate_low_quality_data(
    input_root: Path,
    output_root: Path,
    degradations: Sequence[str],
    levels: Sequence[str],
    *,
    seed: int = 42,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, int]:
    """Create parallel degraded trees while preserving relative image paths."""
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output root must not be inside the input root.")

    images = collect_images(input_root)
    if not images:
        raise RuntimeError(f"No supported images found under: {input_root}")

    jobs: Iterable[tuple[Path, str, str]] = (
        (path, degradation, level)
        for degradation in degradations
        for level in levels
        for path in images
    )
    total = len(images) * len(degradations) * len(levels)
    if progress:
        from tqdm import tqdm

        jobs = tqdm(jobs, total=total, desc="Generating", dynamic_ncols=True)

    counts = {"created": 0, "skipped": 0, "failed": 0}
    failures = []
    for input_path, degradation, level in jobs:
        relative = input_path.relative_to(input_root)
        output_path = output_root / degradation / level / "train" / relative
        if output_path.exists() and not overwrite:
            counts["skipped"] += 1
            continue

        try:
            with Image.open(input_path) as source:
                result = degrade_image(
                    source,
                    degradation,
                    level,
                    rng=_rng_for(seed, relative.as_posix(), degradation, level),
                )
            _save_image(result, output_path)
            counts["created"] += 1
        except Exception as exc:  # noqa: BLE001 - collect per-file failures
            counts["failed"] += 1
            failures.append(f"{input_path}: {exc}")

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"Failed to process {len(failures)} image(s). First failures:\n{preview}"
        )
    return counts


def add_generation_arguments(parser, *, default_input: str, default_output: str):
    parser.add_argument("--input-root", type=Path, default=Path(default_input))
    parser.add_argument("--output-root", type=Path, default=Path(default_output))
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(DEGRADATION_TYPES),
        choices=DEGRADATION_TYPES,
        help="One or more degradations (default: all three).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(LEVELS),
        choices=LEVELS,
        help="One or more severity levels (default: all three).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser
