"""Parse a training log or metrics CSV and draw train/val loss."""
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt

STEP_RE = re.compile(
    r"Step\s+(\d+)/\d+\s+\|\s+train:\s+([\d.]+)\s+\|\s+val:\s+([\d.]+)"
)


def parse_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                rows.append(
                    {
                        "step": int(m.group(1)),
                        "train": float(m.group(2)),
                        "val": float(m.group(3)),
                    }
                )
    if not rows:
        raise SystemExit(f"no Step lines found in {path}")
    return rows


def parse_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = [
            {
                "step": int(r["step"]),
                "train": float(r["train"]),
                "val": float(r["val"]),
            }
            for r in csv.DictReader(f)
        ]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train", "val"])
        w.writeheader()
        w.writerows(rows)


def plot_curve(rows, path, title):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    steps = [r["step"] for r in rows]
    train = [r["train"] for r in rows]
    val = [r["val"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=140)
    ax.plot(steps, train, color="#1f77b4", linewidth=1.4, label="train")
    ax.plot(
        steps,
        val,
        color="#ff7f0e",
        linewidth=1.4,
        alpha=0.85,
        label="val (one random batch)",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None, help="stdout log with 'Step N/M | train:' lines")
    p.add_argument("--from_csv", default=None, help="metrics csv with step,train,val")
    p.add_argument("--csv", default="results/metrics_3000.csv")
    p.add_argument("--out", default="results/loss_curve.png")
    p.add_argument(
        "--title",
        default="Smoke run: 3000 steps (not the 170k checkpoint)",
    )
    args = p.parse_args()

    if args.from_csv:
        rows = parse_csv(args.from_csv)
    elif args.log:
        rows = parse_log(args.log)
        write_csv(rows, args.csv)
    else:
        rows = parse_log("train_run.log")
        write_csv(rows, args.csv)

    plot_curve(rows, args.out, args.title)
    print(
        f"plotted {len(rows)} points → {args.out}\n"
        f"step {rows[0]['step']}: train {rows[0]['train']:.4f} val {rows[0]['val']:.4f}\n"
        f"step {rows[-1]['step']}: train {rows[-1]['train']:.4f} val {rows[-1]['val']:.4f}"
    )


if __name__ == "__main__":
    main()
