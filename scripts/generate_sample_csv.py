from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
import random


def generate(path: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, 8, 0, 0)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "time_s",
                "case_alpha",
                "case_beta",
                "case_gamma",
                "aux_temperature",
                "aux_pressure",
                "operator_note",
            ]
        )
        elapsed = 0.0
        for index in range(rows):
            elapsed += 0.8 + rng.random() * 0.5
            timestamp = start + timedelta(seconds=elapsed)
            baseline = 20.0 + 0.002 * elapsed
            event_1 = 18.0 * math.exp(-((elapsed - 1100.0) / 95.0) ** 2)
            event_2 = 24.0 * math.exp(-((elapsed - 2600.0) / 160.0) ** 2)
            event_3 = 15.0 * math.exp(-((elapsed - 4200.0) / 120.0) ** 2)

            alpha = baseline + 3.0 * math.sin(elapsed / 80.0) + event_1 + rng.uniform(-0.4, 0.4)
            beta = baseline * 0.9 + 2.2 * math.cos(elapsed / 100.0) + event_2 + rng.uniform(-0.5, 0.5)
            gamma = baseline * 1.05 + 1.5 * math.sin(elapsed / 45.0) + event_3 + rng.uniform(-0.35, 0.35)
            temperature = 68.0 + 5.0 * math.sin(elapsed / 900.0) + rng.uniform(-0.25, 0.25)
            pressure = 101.3 + 0.8 * math.cos(elapsed / 700.0) + rng.uniform(-0.08, 0.08)
            note = "maintenance" if index % 1400 == 0 and index else ""

            writer.writerow(
                [
                    timestamp.isoformat(sep=" "),
                    f"{elapsed:.3f}",
                    f"{alpha:.5f}",
                    f"{beta:.5f}",
                    f"{gamma:.5f}",
                    f"{temperature:.5f}",
                    f"{pressure:.5f}",
                    note,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic Event Analyzer CSV file.")
    parser.add_argument("--out", type=Path, default=Path("sample_timeseries.csv"))
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate(args.out, args.rows, args.seed)
    print(f"Wrote {args.rows:,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

