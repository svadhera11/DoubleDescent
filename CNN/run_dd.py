#!/usr/bin/env python
import argparse
import os

from dd import (
    run_model_wise_grid,
    plot_model_wise_results,
    run_epoch_wise_multi,
    plot_epoch_wise_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run double-descent experiments using dd.py helpers."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["model-wise", "epoch-wise"],
        default="model-wise",
        help="Which experiment to run.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "mnist"],
        required=True,
        help="Dataset to use.",
    )

    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 96, 128],
        help="List of widths k for CNN5 (model-wise mode).",
    )
    parser.add_argument(
        "--noise-probs",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2],
        help="Label noise probabilities p.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to average over.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500_000,
        help="Number of SGD steps to train for.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10_000,
        help="Evaluate every this many steps.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.1,
        help="Initial learning rate for inverse-sqrt schedule.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="results",
        help="Root directory for output (plots + pickles).",
    )

    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Train sizes to run in model-wise mode. "
            "If omitted, use full dataset (train_size=None)."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Width k for epoch-wise experiments.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.2,
        help="Label noise p for epoch-wise experiments.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Train size for epoch-wise experiments (None = full dataset).",
    )
    parser.add_argument(
        "--epoch-tag",
        type=str,
        default=None,
        help="Tag for epoch-wise plots; default constructed from params.",
    )

    return parser.parse_args()


def main() -> None:
    
    args = parse_args()

    augment = not args.no_augment

    if args.mode == "model-wise":
        if args.train_sizes is None or len(args.train_sizes) == 0:
            train_sizes = [None]
        else:
            train_sizes = args.train_sizes

        for ts in train_sizes:
            if ts is None:
                ts_label = "full"
            else:
                ts_label = str(ts)

            tag = f"{args.dataset}_n{ts_label}"
            out_dir = os.path.join(args.out_root, tag)

            print(f"Running model-wise grid for {args.dataset}, train_size={ts_label}")

            results = run_model_wise_grid(
                dataset=args.dataset,
                widths=args.widths,
                noise_probs=args.noise_probs,
                seeds=args.seeds,
                train_size=ts,
                augment=augment,
                num_steps=args.num_steps,
                eval_every=args.eval_every,
                lr0=args.lr0,
                batch_size=args.batch_size,
                data_root=args.data_root,
            )

            plot_model_wise_results(
                results=results,
                widths=args.widths,
                noise_probs=args.noise_probs,
                out_dir=out_dir,
                tag=tag,
            )

    elif args.mode == "epoch-wise":
        ts = args.train_size
        if ts is None:
            ts_label = "full"
        else:
            ts_label = str(ts)

        tag = args.epoch_tag
        if tag is None:
            tag = f"{args.dataset}_k{args.width}_p{args.noise_prob}_n{ts_label}"

        out_dir = os.path.join(args.out_root, tag)

        print(
            f"Running epoch-wise multi for {args.dataset}, "
            f"k={args.width}, p={args.noise_prob}, train_size={ts_label}"
        )

        epoch_result = run_epoch_wise_multi(
            dataset=args.dataset,
            width=args.width,
            noise_prob=args.noise_prob,
            seeds=args.seeds,
            train_size=ts,
            augment=augment,
            num_steps=args.num_steps,
            eval_every=args.eval_every,
            lr0=args.lr0,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )

        plot_epoch_wise_result(
            epoch_result=epoch_result,
            out_dir=out_dir,
            tag=tag,
        )


if __name__ == "__main__":
    main()
