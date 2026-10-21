"""Plot bw_sweep CSVs: bandwidth (GB/s) against working-set size.

    tools/.venv/bin/python tools/plot_sweep.py results/phase-1/raw/sweep-*.csv \
        -o results/phase-1/sweep.png

Each CSV becomes one line, labelled from its `# kernel ...` header. The x axis
is log2 (each step is a doubling), so cache boundaries show up as cliffs at
fixed horizontal positions whatever the sweep range.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    label, xs, ys, lo, hi = path.stem, [], [], [], []
    with open(path) as f:
        rows = []
        for line in f:
            if line.startswith("# kernel"):
                m = re.search(r"kernel (\w+) \| cores (\[[^\]]*\]) \| (\d+) thread", line)
                if m:
                    label = f"{m[1]}, cores {m[2]}, {m[3]} thr"
            elif not line.startswith("#"):
                rows.append(line)
    for r in csv.DictReader(rows):
        xs.append(int(r["working_set_bytes"]))
        ys.append(float(r["median_gbps"]))
        lo.append(float(r["worst_gbps"]))
        hi.append(float(r["best_gbps"]))
    return label, xs, ys, lo, hi


def size_label(b):
    for unit, s in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= s:
            return f"{b / s:g} {unit}"
    return f"{b} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--title", default="Read bandwidth vs working set (iQOO 13, SM8750)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in args.csv:
        label, xs, ys, lo, hi = load(p)
        ax.plot(xs, ys, marker="o", ms=3, label=label)
        ax.fill_between(xs, lo, hi, alpha=0.15)  # worst..best sample band
    ax.set_xscale("log", base=2)
    ticks = [1 << k for k in range(10, 31, 2)]
    ax.set_xticks(ticks, [size_label(t) for t in ticks])
    ax.set_xlabel("working set (log scale)")
    ax.set_ylabel("GB/s (median of samples; band = worst..best)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
