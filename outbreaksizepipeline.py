#!/usr/bin/env python3
"""
Script to run targeted epidemics from the hub node of each network
for multiple infection rates and count outbreaks exceeding 90% recovery.
"""

import os
import re
import glob
import subprocess
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import pipeline.data as data

# ============================================================
# Fixed simulation parameters (consistent with original pipeline)
# ============================================================
N = 1000
K = 20               # average degree
GAMMA = 2.1          # for config and S¹/H²
BETA = 2.1           # for S¹/H²
R_RATE = 1.0
LIMIT_TIME = 1e10
MODEL_TYPE = 2       # SIR
BASE_SEED = 42075    # base for epidemic seeds
NUM_EPIDEMIC_SEEDS = 10000

# Infection rates from 0.01 to 0.09 step 0.005
I_RATES = np.arange(0.1, 0.95, 0.05)

# Networks to process
MODELS = ['er', 'ba', 'config', 's1h2']
NETWORK_SEEDS = [12345, 12346, 12347, 12348, 12349]

# ============================================================
# Helper functions
# ============================================================
def get_network_file(model, net_seed, extension):
    """Return the full path to the edge file for a given model and network seed."""
    base = f'generated-nets/{model}-s={net_seed}'
    if model == 'er':
        return f'{base}/er-n={N}-k={K}-s={net_seed}.{extension}'
    elif model == 'ba':
        return f'{base}/ba-n={N}-k={K}-s={net_seed}.{extension}'
    elif model == 'config':
        return f'{base}/conf-n={N}-k={K}-g={GAMMA}-s={net_seed}.{extension}'
    elif model == 's1h2':
        return f'{base}/s1h2-n={N}-k={K}-g={GAMMA}-b={BETA}-s={net_seed}.{extension}'
    else:
        raise ValueError(f'Unknown model: {model}')

def get_edges_file(model, net_seed):
    return get_network_file(model, net_seed, 'edge')

def get_coords_file(model, net_seed):
    return get_network_file(model, net_seed, 'gen_coord' if model == 's1h2' else 'inf_coord')

def generate_batch_file(output_dir, hub_node, i_rates, num_seeds, base_seed):
    """
    Create a batch file for the epidemics tool.
    Returns the path to the batch file.
    """
    batch_file = os.path.join(output_dir, 'batch_targeted_sir_hub.txt')
    with open(batch_file, 'w') as f:
        f.write("# infection_rate recovery_rate seed limit_time model_type start_node\n")
        seed_counter = base_seed
        for i_rate in i_rates:
            for _ in range(num_seeds):
                f.write(f"{i_rate:.6f} {R_RATE:.6f} {seed_counter} {LIMIT_TIME:.1f} {MODEL_TYPE} {hub_node}\n")
                seed_counter += 1
    return batch_file

def run_epidemics(batch_file, weight_file, output_dir):
    """
    Execute the epidemics tool with the given batch file.
    No edge weights (-ev) are used.
    """
    cmd = [
        "./tools/epidemics",
        "-b", batch_file,
        "-o", output_dir,
        "-st",           # write statistics only
        "-ss", "10000",     # sampling steps (kept as in original)
        "-w",weight_file
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def parse_stats_file(filepath):
    """
    Read a stats file produced by the epidemics tool.
    Returns the final recovered density (last row of the 'recovered_density' column).
    """
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None,
                     names=['time', 'infected_density', 'recovered_density',
                            'actual_infection_rate', 'actual_recovery_rate'])
    if df.empty:
        return None
    return df['recovered_density'].iloc[-1]


def extract_parameters_from_filename(filename):
    """
    Extract i_rate, seed and start_node from a stats/events filename.
    Expected format (from build_epidemics_filename):
        stats-<edges_stem>-<w?><model>-I=  0.01000-R=  1.00000-S=42075-SN=00042.dat
    Returns (i_rate, seed, start_node) or None if pattern not found.
    """
    base = os.path.basename(filename)
    # Allow optional spaces due to fixed-width formatting (e.g., :10.5f, :5d, :05d)
    # Pattern: -I= followed by optional spaces, a float, then -R= optional spaces float,
    # then -S= optional spaces integer, then -SN= optional spaces integer (maybe with leading zeros)
    match = re.search(
        r'-I=\s*([\d\.]+)\s+-R=\s*[\d\.]+\s+-S=\s*(\d+)\s+-SN=\s*(\d+)',
        base
    )
    if match:
        i_rate = float(match.group(1))
        seed = int(match.group(2))
        start_node = int(match.group(3))
        return i_rate, seed, start_node
    return None

def count_successes(output_dir, hub_node, expected_i_rates, num_seeds):
    """
    Scan all stats files in output_dir, compute final recovery density,
    and count how many seeds per infection rate have final recovery > 0.9.
    Returns a dict {i_rate: success_count}.
    """
    # Counters
    success = {ir: 0 for ir in expected_i_rates}
    total = {ir: 0 for ir in expected_i_rates}

    # Find all .dat files (stats files)
    stats_files = glob.glob(os.path.join(output_dir, "*.dat"))
    for fpath in stats_files:
        params = extract_parameters_from_filename(fpath)
        if params is None:
            continue
        i_rate, seed, start_node = params
        # Only consider simulations that started from the hub node
        if start_node != hub_node:
            continue
        # Update totals
        total[i_rate] += 1
        final_rec = parse_stats_file(fpath)
        if final_rec is not None and final_rec > 0.9:
            success[i_rate] += 1

    # Sanity check: all i_rates should have exactly num_seeds files
    for ir, cnt in total.items():
        if cnt != num_seeds:
            print(f"  Warning: for i_rate={ir:.4f} only {cnt}/{num_seeds} stats files found")
    return success

def save_results(output_dir, successes, num_seeds):
    """Write the success counts to a file."""
    out_file = os.path.join(output_dir, 'success_counts.dat')
    with open(out_file, 'w') as f:
        f.write("# infection_rate  successes  total_simulations\n")
        for i_rate in I_RATES:
            f.write(f"{i_rate:.6f} {successes[i_rate]} {num_seeds}\n")
    print(f"  Results saved to {out_file}")

# ============================================================
# Main loop
# ============================================================
def main():
    for model in MODELS:
        for net_seed in NETWORK_SEEDS:
            print(f"\n=== Processing {model} with network seed {net_seed} ===")
            edge_file = get_edges_file(model, net_seed)
            if not os.path.exists(edge_file):
                print(f"  Edge file not found: {edge_file}. Skipping.")
                continue

            # Find the hub (highest degree) node
            G, df, params = data.read_hyperbolic_data(get_coords_file(model, net_seed), edge_file, model == 's1h2')
            hub_node = data.get_most_popular_node(G)
            print(f"  Hub node: {hub_node} (degree {G.degree()[hub_node]})")

            # Create output directory for this network
            base_folder = os.path.dirname(edge_file)
            output_dir = os.path.join(base_folder, 'outbreak-size-epidemics')
            os.makedirs(output_dir, exist_ok=True)

            # Generate batch file and run epidemics
            batch_file = generate_batch_file(output_dir, hub_node, I_RATES,
                                             NUM_EPIDEMIC_SEEDS, BASE_SEED)
            print("  Running epidemics (this may take a while)...")
            weight_file = f"{edge_file}_weight_-1x10^-1"
            run_epidemics(batch_file, weight_file, output_dir)
            print("  Epidemics finished.")

            # Count successful outbreaks (final recovered density > 0.9)
            successes = count_successes(output_dir, hub_node, I_RATES, NUM_EPIDEMIC_SEEDS)
            save_results(output_dir, successes, NUM_EPIDEMIC_SEEDS)

if __name__ == "__main__":
    main()