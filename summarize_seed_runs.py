"""
Parses nsga2_scenario_search.py's multi-seed terminal output and
computes a robustness summary -- coefficient of variation across seeds,
comparison against the greedy baseline's known maximum (223,803.4 t C
at disruption 0.6), and whether the "NSGA-II beats greedy everywhere"
claim holds for EVERY seed, not just the best one.

Usage: paste the full terminal output into a text file and run:
    python3 summarize_seed_runs.py path/to/output.txt
"""
import re
import sys
import numpy as np

GREEDY_MAX = 223803.4  # from baseline_comparison.py's disruption=0.6 result

def parse_seed_runs(text):
    seed_blocks = re.split(r"=+\s*SEED (\d+)\s*=+", text)[1:]  # drop preamble
    results = {}
    for i in range(0, len(seed_blocks), 2):
        seed = int(seed_blocks[i])
        block = seed_blocks[i + 1]
        rows = re.findall(r"^\s*([\d.]+)\s+([\d.]+)\s*$", block, re.MULTILINE)
        if not rows:
            continue
        carbon_vals = [float(c) for _, c in rows]
        disruption_vals = [float(d) for d, _ in rows]
        best_idx = int(np.argmax(carbon_vals))
        results[seed] = {
            "best_carbon": carbon_vals[best_idx],
            "best_disruption": disruption_vals[best_idx],
            "n_points": len(rows),
            "disruption_range": (min(disruption_vals), max(disruption_vals)),
        }
    return results


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        text = f.read()

    results = parse_seed_runs(text)
    if not results:
        print("No SEED blocks found -- paste the FULL multi-seed output "
              "(including the '==== SEED N ====' headers), not a fragment.")
        sys.exit(1)

    print(f"Parsed {len(results)} seed runs\n")
    bests = []
    for seed, r in sorted(results.items()):
        beats_greedy = r["best_carbon"] > GREEDY_MAX
        print(f"seed={seed}: best={r['best_carbon']:,.1f} t C at disruption="
              f"{r['best_disruption']:.1f}, front spans "
              f"{r['disruption_range'][0]:.1f}-{r['disruption_range'][1]:.1f}, "
              f"n_points={r['n_points']}, beats greedy max: {beats_greedy}")
        bests.append(r["best_carbon"])

    bests = np.array(bests)
    cv = 100 * bests.std() / bests.mean()
    print(f"\nBest-case carbon gain across seeds: {bests.min():,.1f} to {bests.max():,.1f} t C")
    print(f"Mean: {bests.mean():,.1f}, std: {bests.std():,.1f}, CV: {cv:.1f}%")
    print(f"All seeds beat greedy's max ({GREEDY_MAX:,.1f} t C): {(bests > GREEDY_MAX).all()}")
