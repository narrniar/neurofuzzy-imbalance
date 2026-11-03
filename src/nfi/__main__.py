"""Command line entry point.

  python -m nfi run configs/quick.yaml [--settings a:b,c:d] [--models tsk,lgbm]
  python -m nfi report configs/quick.yaml
  python -m nfi datasets                      # sizes, positive rates, split sizes
"""
from __future__ import annotations

import argparse
import sys

from .data import DATASETS, Setting, load_table, make_split
from .experiment import Experiment, Runner


def cmd_run(args: argparse.Namespace) -> None:
    exp = Experiment.load(args.config)
    settings = args.settings.split(",") if args.settings else None
    models = args.models.split(",") if args.models else None
    Runner(exp, settings, models, threads=args.threads).run()


def cmd_report(args: argparse.Namespace) -> None:
    from .report import build_report

    exp = Experiment.load(args.config)
    build_report(exp)


def cmd_datasets(args: argparse.Namespace) -> None:
    for name in DATASETS:
        table = load_table(name)
        rate = table.y.mean()
        print(f"{name}: rows={len(table.y):,} features={table.frame.shape[1]} "
              f"(categorical={len(table.categorical)}) positives={int(table.y.sum()):,} rate={rate:.4%}")
        for r in ["natural", "0.05", "0.01"]:
            if r != "natural" and float(r) >= rate:
                continue
            split = make_split(table, Setting(name, r), 0)
            sizes = {k: f"{len(v):,}/{int(table.y[v].sum())}" for k, v in split.as_dict().items()}
            print(f"   rate={r:<8} rows/positives {sizes} dup(train,test)={split.duplicates_train_test}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nfi")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run search, final seeds and ablations for a config file")
    p_run.add_argument("config")
    p_run.add_argument("--settings", help="comma-separated subset, e.g. fraud:natural,bank_marketing:0.05")
    p_run.add_argument("--models", help="comma-separated subset of logreg,lgbm,mlp,tsk")
    p_run.add_argument("--threads", type=int, help="override the thread count from the config file")
    p_run.set_defaults(func=cmd_run)
    p_rep = sub.add_parser("report", help="aggregate runs into tables and figures")
    p_rep.add_argument("config")
    p_rep.set_defaults(func=cmd_report)
    p_ds = sub.add_parser("datasets", help="print dataset and split statistics")
    p_ds.set_defaults(func=cmd_datasets)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
