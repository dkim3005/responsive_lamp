from __future__ import annotations

import sqlite3
import statistics
from collections import defaultdict

from config import DB_PATH


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[idx]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT metric, value_ms FROM latency_log ORDER BY ts").fetchall()
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric, value in rows:
        grouped[metric].append(float(value))

    print("| Metric | n | mean ms | p50 ms | p95 ms |")
    print("|---|---:|---:|---:|---:|")
    for metric in sorted(grouped):
        values = grouped[metric]
        print(
            f"| {metric} | {len(values)} | {statistics.mean(values):.1f} | "
            f"{percentile(values, 0.50):.1f} | {percentile(values, 0.95):.1f} |"
        )


if __name__ == "__main__":
    main()

