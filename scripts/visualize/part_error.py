"""
Simple script to visualize pair error rates from a result file.

Example usage:
    python part_error.py --file logs/my_model/train/result.txt --out plots/error_rates.png
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


LineData = Tuple[float, float]


def parse_result_file(file_path: str) -> Dict[int, LineData]:
    """
    Parse a result.txt-like file and extract (acc, total_samples) for indices 0..7.

    Returns a dict: {index: (acc, total_samples)}. Stops after the first complete 0..7 set.
    """
    index_pattern = re.compile(
        r"^(?P<idx>[0-7]):\s*acc:\s*(?P<acc>[0-9]+\.[0-9]+),\s*total_samples:\s*(?P<total>[0-9]+\.[0-9]+)",
    )
    extracted: Dict[int, LineData] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = index_pattern.match(line.strip())
            if not m:
                continue
            idx = int(m.group("idx"))
            acc = float(m.group("acc"))
            total = float(m.group("total"))
            # Only keep the first occurrence per index (assume the first full set 0..7 is desired)
            if idx not in extracted:
                extracted[idx] = (acc, total)
            # Break once we have a complete set
            if len(extracted) == 8:
                break
    if len(extracted) < 8:
        missing = sorted(set(range(8)) - set(extracted.keys()))
        raise ValueError(
            f"File {file_path} does not contain a complete first set of indices 0..7. Missing: {missing}"
        )
    return extracted


def compute_pair_error_rates(index_to_data: Dict[int, LineData]) -> List[float]:
    """
    Compute error rates for pairs: (0+4), (1+5), (2+6), (3+7).

    Error count for an index i: (100 - acc) * total / 100
    Pair error rate: (error_i + error_j) / (total_i + total_j)
    """
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    rates: List[float] = []
    for a, b in pairs:
        acc_a, total_a = index_to_data[a]
        acc_b, total_b = index_to_data[b]
        error_a = (100.0 - acc_a) * total_a / 100.0
        error_b = (100.0 - acc_b) * total_b / 100.0
        denom = total_a + total_b
        rate = 0.0 if denom == 0 else (error_a + error_b) / denom
        rates.append(rate)
    return rates


def plot_error_rates(
    labels: List[str],
    values: List[float],
    title: Optional[str],
    output_path: str,
) -> None:
    """Plot error rates as a simple bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = list(range(len(labels)))
    
    ax.bar(x, values, color='#abd3e1', edgecolor='#8693a0', linewidth=0.4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Error rate", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot pair error rates from a result file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python part_error.py --file logs/my_model/train/result.txt --out plots/error_rates.png
        """,
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to result.txt file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title (leave empty for no title)",
    )
    args = parser.parse_args()

    data = parse_result_file(args.file)
    rates = compute_pair_error_rates(data)

    labels = ["Start", "Middle", "End", "Unit"]

    # Print rates for quick inspection
    print(f"Error rates: {[round(r, 6) for r in rates]}")

    plot_error_rates(
        labels=labels,
        values=rates,
        title=args.title,
        output_path=args.out,
    )
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()


