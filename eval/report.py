from __future__ import annotations

import sqlite3
import statistics
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_PATH
from memory.store import MemoryStore


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[idx]


def main() -> None:
    MemoryStore(DB_PATH).close()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT metric, value_ms FROM latency_log ORDER BY ts").fetchall()
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric, value in rows:
        grouped[metric].append(float(value))

    print("## Latency")
    print("| Metric | n | mean ms | p50 ms | p95 ms |")
    print("|---|---:|---:|---:|---:|")
    for metric in sorted(grouped):
        values = grouped[metric]
        print(
            f"| {metric} | {len(values)} | {statistics.mean(values):.1f} | "
            f"{percentile(values, 0.50):.1f} | {percentile(values, 0.95):.1f} |"
        )

    print()
    print("## Engagement Reliability")
    labels = conn.execute("SELECT truth, predicted, detected, method FROM engagement_labels ORDER BY ts").fetchall()
    sample_count = conn.execute("SELECT COUNT(*) FROM engagement_samples").fetchone()[0]
    if not labels:
        print(f"No labeled engagement rows yet. Automatic prediction samples recorded: {sample_count}.")
        print("During a demo, press E when you are looking at the lamp and D when you are looking away.")
        return

    truth = [bool(row["truth"]) for row in labels]
    pred = [bool(row["predicted"]) for row in labels]
    tp = sum(t and p for t, p in zip(truth, pred))
    tn = sum((not t) and (not p) for t, p in zip(truth, pred))
    fp = sum((not t) and p for t, p in zip(truth, pred))
    fn = sum(t and (not p) for t, p in zip(truth, pred))
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detected_rate = sum(bool(row["detected"]) for row in labels) / len(labels)

    print("| Labels | samples | accuracy | precision | recall | F1 | detected rate |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {len(labels)} | {sample_count} | {accuracy:.2f} | {precision:.2f} | "
        f"{recall:.2f} | {f1:.2f} | {detected_rate:.2f} |"
    )
    print()
    print("| truth/pred | engaged | disengaged |")
    print("|---|---:|---:|")
    print(f"| engaged | {tp} | {fn} |")
    print(f"| disengaged | {fp} | {tn} |")


if __name__ == "__main__":
    main()
