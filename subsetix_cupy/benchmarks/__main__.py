from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

import numpy as np

from . import load_cases, run_case


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="subsetix CuPy benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Substring / regex used to filter benchmark names",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override number of repetitions (per benchmark)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override number of warmup iterations (per benchmark)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks without executing",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to dump JSON results",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.pattern:
        pattern = re.compile(args.pattern)
        cases = [case for case in cases if pattern.search(case.name)]

    if not cases:
        raise SystemExit("no benchmarks matched the provided pattern")

    if args.list:
        for case in cases:
            print(f"{case.name:>28s}  |  {case.description}")
        return

    results = []
    for case in cases:
        print(f"Running {case.name} …", end="", flush=True)
        result, metadata, repeat_used, warmup_used = run_case(case, repeat=args.repeat, warmup=args.warmup)
        times = np.asarray(result.gpu_times, dtype=np.float32).reshape(-1)
        avg_ms = float(times.mean() * 1e3) if times.size else 0.0
        std_ms = float(times.std(ddof=0) * 1e3) if times.size else 0.0

        input_intervals = None
        ns_per_interval = None
        if metadata is not None and isinstance(metadata, dict):
            raw = metadata.get("input_intervals")
            if raw is not None:
                input_intervals = int(raw)
                if input_intervals > 0:
                    ns_per_interval = (avg_ms * 1e6) / input_intervals

        summary = f" {avg_ms:8.3f} ms ± {std_ms:6.3f} ms (repeat={repeat_used}, warmup={warmup_used})"
        if ns_per_interval is not None:
            summary += f" | {ns_per_interval:7.2f} ns/int (input={input_intervals})"
        print(summary)
        record: Dict[str, object] = {
            "name": case.name,
            "description": case.description,
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "repeat": repeat_used,
            "warmup": warmup_used,
            "gpu_times": [float(t) for t in times],
            "cpu_times": [float(t) for t in np.asarray(result.cpu_times, dtype=np.float32).reshape(-1)],
            "metadata": metadata,
        }
        if input_intervals is not None:
            record["input_intervals"] = input_intervals
        if ns_per_interval is not None:
            record["ns_per_interval"] = ns_per_interval
        results.append(record)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
