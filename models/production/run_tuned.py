#!/usr/bin/env python
"""Fit a theory-mapped variant of a shared-kappa production model.

Example (step 2 of the staged comparison, plan-guided, context-fixed):
    python run_tuned.py --model K-HO --output fits/step2 \
        --drop-dead-terms --unscaled-size-contrast --size-blur-multiplier 2
"""
from __future__ import annotations

import argparse
import dataclasses
from argparse import Namespace
from functools import partial
from pathlib import Path

import run_corrected_primary as primary
import tuned_model as tm

FLAGS = tuple(f.name for f in dataclasses.fields(tm.TuningSpec)
              if f.name != "size_blur_multiplier")


def parse_args() -> tuple[argparse.Namespace, tm.TuningSpec]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=tm.SUPPORTED_MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-tree-depth", type=int, default=6)
    parser.add_argument("--size-blur-multiplier", type=float)
    parser.add_argument("--colour-reliability", type=float,
                        help="colour semantic value (default .59)")
    for flag in FLAGS:
        parser.add_argument("--" + flag.replace("_", "-"), action="store_true")
    args = parser.parse_args()
    spec = tm.TuningSpec(size_blur_multiplier=args.size_blur_multiplier,
                         **{flag: getattr(args, flag) for flag in FLAGS})
    return args, spec


def main() -> None:
    args, spec = parse_args()
    primary.run(Namespace(
        model=args.model, output=args.output, warmup=args.warmup, samples=args.samples,
        batch=args.batch, max_tree_depth=args.max_tree_depth, recovery=False,
        drop_condition_terms=False, terminal_utility=False, drop_order_term=False,
        drop_informativeness=False, drop_prefix_salience=False, input=None,
        input_sha256=None, colour_reliability=args.colour_reliability, form_reliability=None,
        model_override=tm.make_tuned_model(args.model, spec),
        precompute_override=partial(tm.precompute_tuned_features, spec=spec),
        tuning=dataclasses.asdict(spec)))


if __name__ == "__main__":
    main()
