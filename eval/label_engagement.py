from __future__ import annotations

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple engagement metrics from CSV labels.")
    parser.add_argument("csv_path", help="CSV with columns truth,pred where values are 0 or 1")
    args = parser.parse_args()

    tp = tn = fp = fn = 0
    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            truth = row["truth"].strip() == "1"
            pred = row["pred"].strip() == "1"
            if truth and pred:
                tp += 1
            elif truth and not pred:
                fn += 1
            elif not truth and pred:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"precision={precision:.3f} recall={recall:.3f} accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()

