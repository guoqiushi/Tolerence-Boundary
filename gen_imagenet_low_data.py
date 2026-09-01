#!/usr/bin/env python3
"""Generate Gaussian blur/noise/JPEG versions of the ImageNet-100 train set."""

import argparse

from low_quality import add_generation_arguments, generate_low_quality_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(
        parser,
        default_input="/home/work-base/datasets/imagenet100/train",
        default_output="/home/work-base/datasets/imagenet100/low_quality",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    counts = generate_low_quality_data(
        args.input_root,
        args.output_root,
        args.types,
        args.levels,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Done: {counts}")
    print(f"Output root: {args.output_root.expanduser().resolve()}")


if __name__ == "__main__":
    main()
