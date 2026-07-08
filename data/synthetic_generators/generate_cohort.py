"""CLI entrypoint to generate the full synthetic MSME cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.synthetic_generators.persona_builder import DEFAULT_COHORT_COUNTS, generate_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MSME alternate-data cohort")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for per-MSME folders",
    )
    args = parser.parse_args()

    generated = generate_cohort(seed=args.seed, output_dir=args.output_dir)
    print(f"Generated {len(generated)} MSMEs into {args.output_dir}")
    for persona, count in DEFAULT_COHORT_COUNTS.items():
        print(f"  {persona}: {count}")


if __name__ == "__main__":
    main()
