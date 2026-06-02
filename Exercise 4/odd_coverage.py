"""
Exercise 4.5 — ODD Coverage with k-Projection Coverage


"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# kprojection.py is bundled in this folder (from github.com/kkirchheim/odd-coverage)
sys.path.insert(0, str(Path(__file__).parent))
from kprojection import KProjectionCoverage

# ── ODD definition ─────────────────────────────────────────────────────────────
# These are the dimensions and their possible values as defined in Sheet 2.
# They reflect the *intended* operating domain of the AV system.
ODD_DESCRIPTION = {
    "weather":                ["clear", "cloudy", "rain", "fog"],
    "lighting":               ["day", "dusk_dawn", "night"],
    "road_type":              ["urban_mapped", "urban_unmapped"],
    "speed_range":            ["low", "medium", "high"],
    "pedestrian_present":     ["yes", "no"],
    "vehicle_present":        ["yes", "no"],
    "traffic_light_present":  ["yes", "no"],
}

# Total ODD space size for reference
def odd_space_size(desc: dict) -> int:
    s = 1
    for v in desc.values():
        s *= len(v)
    return s


def load_split_scenarios(split_root: Path,
                         weather: str,
                         lighting: str,
                         road_type: str = "urban_mapped") -> list[dict]:
    """
    Load labels.csv and convert each row to an ODD scenario dict.


    """
    csv = split_root / "labels.csv"
    if not csv.exists():
        print(f"  [skip] no labels.csv at {csv}")
        return []

    df = pd.read_csv(csv)

    # Try to get speed from actions.feather if present
    speed_col = None
    actions_path = split_root / "actions.feather"
    if actions_path.exists():
        try:
            import pyarrow.feather as feather
            actions = feather.read_feather(actions_path)
            # CARLA actions typically has a 'speed' column in m/s
            if "speed" in actions.columns and "frame" in actions.columns:
                speed_col = actions.set_index("frame")["speed"]
        except Exception:
            pass

    scenarios = []
    for _, row in df.iterrows():
        # Map pixel speed or metadata to speed_range
        if speed_col is not None:
            frame = int(row["frame"])
            try:
                speed_ms = float(speed_col.loc[frame])
                speed_kmh = speed_ms * 3.6
            except (KeyError, TypeError):
                speed_kmh = 25.0  # fallback
        else:
            speed_kmh = 25.0  # fallback: assume urban medium

        if speed_kmh < 20:
            speed_range = "low"
        elif speed_kmh < 40:
            speed_range = "medium"
        else:
            speed_range = "high"

        scenarios.append({
            "weather":               weather,
            "lighting":              lighting,
            "road_type":             road_type,
            "speed_range":           speed_range,
            "pedestrian_present":    "yes" if bool(row["has_pedestrian"]) else "no",
            "vehicle_present":       "yes" if bool(row["has_vehicle"]) else "no",
            "traffic_light_present": "yes" if bool(row["has_traffic_light"]) else "no",
        })
    return scenarios


def compute_coverage_for_scenarios(scenarios: list[dict],
                                   desc: dict,
                                   ks: list[int]) -> dict[int, object]:
    results = {}
    for k in ks:
        cov = KProjectionCoverage(k=k, desc=desc)
        cov.add_scenarios(scenarios)
        results[k] = cov.compute()
    return results


def plot_coverage(results_by_split: dict[str, dict],
                  ks: list[int],
                  out_path: Path) -> None:
    """Bar chart: coverage per split per k."""
    n_splits = len(results_by_split)
    x = range(len(ks))
    width = 0.8 / max(n_splits, 1)
    colors = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (split_name, res_dict) in enumerate(results_by_split.items()):
        vals = [res_dict[k].coverage for k in ks]
        offset = (i - n_splits / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=split_name, color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("k-projection coverage")
    ax.set_ylim(0, 1.1)
    ax.set_title("ODD k-projection coverage by split")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--include-ood", action="store_true",
                    help="Also process test-fog and test-night splits if present")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ks = [1, 2, 3]

    # Define which splits to load and what their ODD metadata is.
    # The 'test' split is sunny-day urban. The OOD splits extend coverage.
    split_configs = [
        ("test (clear/day)",  "test",       "clear", "day"),
    ]
    if args.include_ood:
        split_configs += [
            ("test-fog",       "test-fog",       "fog",   "day"),
            ("test-night",     "test-night",     "clear", "night"),
            ("test-town-01",   "test-town-01",   "clear", "day"),
        ]

    lines = ["ODD k-Projection Coverage Report", "=" * 60]
    lines.append(f"\nODD dimensions: {list(ODD_DESCRIPTION.keys())}")
    lines.append(f"ODD total space: {odd_space_size(ODD_DESCRIPTION):,} combinations\n")

    all_results: dict[str, dict] = {}
    all_scenarios: list[dict] = []

    for label, folder, weather, lighting in split_configs:
        split_root = args.data_root / folder
        scenarios = load_split_scenarios(split_root, weather, lighting)
        if not scenarios:
            continue
        lines.append(f"Split: {label}  ({len(scenarios)} frames)")
        results = compute_coverage_for_scenarios(scenarios, ODD_DESCRIPTION, ks)
        all_results[label] = results
        all_scenarios.extend(scenarios)
        for k in ks:
            r = results[k]
            lines.append(f"  k={k}: coverage={r.coverage:.4f}  "
                         f"({r.covered}/{r.total} projections covered)")
        lines.append("")

    if len(all_results) > 1:
        lines.append("Combined (all splits):")
        combined = compute_coverage_for_scenarios(all_scenarios, ODD_DESCRIPTION, ks)
        all_results["combined"] = combined
        for k in ks:
            r = combined[k]
            lines.append(f"  k={k}: coverage={r.coverage:.4f}  "
                         f"({r.covered}/{r.total} projections covered)")

    summary = "\n".join(lines)
    print(summary)
    out_txt = args.out_dir / "4.5_odd_coverage.txt"
    out_txt.write_text(summary)
    print(f"\n  wrote {out_txt}")

    plot_coverage(all_results, ks, args.out_dir / "4.5_odd_coverage.png")


if __name__ == "__main__":
    main()
