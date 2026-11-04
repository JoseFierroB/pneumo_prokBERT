#!/usr/bin/env python3
"""Thin CLI: tunable pipeline with argparse-only hyperparameters.

Usage:
  pixi run python -m tunable.cli --preset replica-1ep \\
      --data-dir datasets/high5050_stratified_801010 \\
      --hf-split-spec local:species801010 --output-dir finetuning_outputs/tunable_smoke
  pixi run python -m tunable.cli --preset replica-1ep --data-dir ... --lr ... --gap \\
      --gap-test-csv datasets/hf_splits/species801010/test_segments.csv
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tunable")


def main(argv=None):
    from tunable import config as C
    from tunable import train as T
    from tunable import evaluate as E

    args = C.parse_args(argv)
    res = T.run(args)
    if res.get("dry_run"):
        return res
    if getattr(args, "loss_overlay", True):
        try:
            E.loss_overlay(args.output_dir,
                           out_path=f"{args.output_dir.rstrip('/')}/loss_overlay.png")
        except SystemExit as e:
            print(f"loss overlay skipped: {e}")
    if args.gap:
        if not args.gap_test_csv:
            raise SystemExit("--gap needs --gap-test-csv")
        E.run_gap(f"{args.output_dir.rstrip('/')}/best",
                  alias=args.output_dir.rstrip("/").split("/")[-1],
                  hf_split_dir="datasets/hf_splits/" + args.hf_split_spec.split("local:", 1)[1]
                  if args.hf_split_spec.startswith("local:") else args.hf_split_spec,
                  train_csv=f"{args.data_dir}/train.csv",
                  test_csv=args.gap_test_csv, test_name=args.gap_test_name,
                  out_dir=args.gap_out_dir or "reports/gaps/tunable",
                  batch_size=args.gap_batch_size, gpu=args.gpu or 0,
                  log_path=f"{args.output_dir.rstrip('/')}/gap.log")
    return res


if __name__ == "__main__":
    main()
