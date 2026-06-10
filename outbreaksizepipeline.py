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
import pipeline.hyperbolic as hyp
import shutil
# ============================================================
# Fixed simulation parameters (consistent with original pipeline)
# ============================================================
N = 1000
K = 20               # average degree
GAMMA = 2.1          # for conf and S¹/H²
BETA = 2.1           # for S¹/H²
R_RATE = 1.0
LIMIT_TIME = 1e10
MODEL_TYPE = 2       # SIR
BASE_SEED = 42075    # base for epidemic seeds
NUM_EPIDEMIC_SEEDS = 10000

# Infection rates from 0.01 to 0.09 step 0.005
# I_RATES = np.flip(np.arange(0.95, 1.25, 0.02))
I_RATES = np.flip(np.arange(0.002, 0.1, 0.002))

# Networks to process
MODELS = ['er', 'ba', 'conf', 's1h2']
# MODELS = ['conf', ]
NETWORK_SEEDS = [12348, 12349]

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
    elif model == 'conf':
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
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

def parse_stats_file_fast(filepath):
    """
    Lee solo la última línea del archivo de stats y extrae la densidad de recuperados.
    Retorna float (recovered_density) o None si no se puede obtener.
    """
    try:
        with open(filepath, 'rb') as f:
            # Saltar al final del archivo
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            # Leer hacia atrás desde el final hasta encontrar un salto de línea
            buffer = bytearray()
            pos = size - 1
            while pos >= 0 and buffer.count(b'\n') < 2:
                f.seek(pos)
                buffer.append(f.read(1)[0])
                pos -= 1
            # La última línea completa
            last_line = buffer.split(b'\n')[-2].decode('utf-8').strip()
            last_line = last_line[::-1]
            if not last_line:
                return None
            # Formato esperado (separado por espacios):
            # time infected_density recovered_density actual_infection_rate actual_recovery_rate
            # time, infected_density, recovered_density, actual_infection_rate, actual_recovery_rate
            parts = last_line.split()
            if len(parts) >= 3:
                return float(parts[2])   # tercera columna = recovered_density
    except Exception:
        return None
    return None


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
        r'-I=\s*([\d\.]+)\s*-R=\s*[\d\.]+\s*-S=\s*(\d+)\s*-SN=\s*(\d+)',
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
    Escanea todos los archivos .dat en output_dir, calcula la densidad final de recuperados
    y cuenta cuántas semillas por tasa de infección superan 0.9.
    Retorna diccionarios {i_rate: <R>} y {i_rate: <R^2>}.
    """
    success = {}   # suma de R
    success2 = {}  # suma de R^2
    total = {}     # contador de simulaciones válidas

    stats_files = glob.glob(os.path.join(output_dir, '*.dat'))
    for fpath in tqdm(stats_files, desc=f'Procesando {len(stats_files)} archivos'):
        params = extract_parameters_from_filename(fpath)
        if params is None:
            continue
        i_rate, seed, start_node = params

        key = f'{i_rate:0.5f}'
        if key not in success:
            success[key] = 0.0
            success2[key] = 0.0
            total[key] = 0

        final_rec = parse_stats_file_fast(fpath)
        if final_rec is None:
            continue   # omitir este archivo

        success[key] += final_rec
        success2[key] += final_rec * final_rec
        total[key] += 1

    # Verificar que todos los rates tengan el número esperado de archivos
    for ir, cnt in total.items():
        if cnt != num_seeds:
            print(f"  Advertencia: para i_rate={ir} solo {cnt}/{num_seeds} archivos válidos")

    # Calcular promedios
    for key in total:
        if total[key] > 0:
            success[key] /= total[key]
            success2[key] /= total[key]
        else:
            success[key] = float('nan')
            success2[key] = float('nan')

    return success, success2

def save_results(output_dir, successes, successes2):
    """Write the success counts to a file."""
    out_file = os.path.join(output_dir, 'success_counts.dat')
    # print(successes)
    with open(out_file, 'w') as f:
        f.write("# infection_rate <R> <R^2>\n")
        for i_rate in successes:
            f.write(f"{i_rate} {successes[i_rate]} {successes2[i_rate]}\n")
    print(f"  Results saved to {out_file}")

# ============================================================
# Main loop
# ============================================================
def main():
    for net_seed in NETWORK_SEEDS:
        for model in MODELS:
            print(f"\n=== Processing {model} with network seed {net_seed} ===")
            edge_file = get_edges_file(model, net_seed)
            if not os.path.exists(edge_file):
                print(f"  Edge file not found: {edge_file}. Skipping.")
                continue
            
            # Find the hub (highest degree) node
            G, df, params = data.read_hyperbolic_data(get_coords_file(model, net_seed), edge_file, model == 's1h2')

            edges = pd.DataFrame(G.edges, columns=['a', 'b'])
            edges = pd.merge(edges, df[['Vertex', 'Disc.Radius', 'Inf.Theta']], left_on='a', right_on='Vertex', suffixes=('_a', '_b'))
            edges = pd.merge(edges, df[['Vertex', 'Disc.Radius', 'Inf.Theta']], left_on='b', right_on='Vertex', suffixes=('_a', '_b'))
            edges['Theta_Dif'] = np.pi - np.abs(np.pi - np.abs(edges['Inf.Theta_a']-edges['Inf.Theta_b']))

            edges['Distance'] = np.where(edges['Theta_Dif'] == 0, 
                                        np.abs(edges['Disc.Radius_a']- edges['Disc.Radius_b']), 
                                        hyp.hyperbolic_distance_og(edges['Disc.Radius_a'], edges['Disc.Radius_b'], edges['Theta_Dif']))

            R = 2 * np.log(params['nb. vertices']/(params['mu']*np.pi*params['kappa_min']**2))

            n=0
            c = -(n+1)/10
            edges['Epidemic_Func'] = hyp.link_probability_og(edges['Distance'], R, c)

            avg_epidemic_func = np.average(edges['Epidemic_Func'])

            edges['Weight_Multiplier'] = edges['Epidemic_Func']/avg_epidemic_func
            weight_file = f"{edge_file}_weight_-1x10^-1"

            edges.to_csv(weight_file, sep='\t', header=False, index=False, columns=['a', 'b', 'Weight_Multiplier'])

            hub_node = data.get_most_popular_node(G)
            print(f"  Hub node: {hub_node} (degree {G.degree()[hub_node]})")

            # Create output directory for this network
            base_folder = os.path.dirname(edge_file)
            output_dir = os.path.join(base_folder, 'outbreak-size-epidemics-2')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            output_dir = os.path.join(base_folder, 'outbreak-size-epidemics')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Generate batch file and run epidemics
            batch_file = generate_batch_file(output_dir, hub_node, I_RATES,
                                             NUM_EPIDEMIC_SEEDS, BASE_SEED)
            print("  Running epidemics (this may take a while)...")
            run_epidemics(batch_file, weight_file, output_dir)
            print("  Epidemics finished.")

            # Count successful outbreaks (final recovered density > 0.9)
            successes, success2 = count_successes(output_dir, int(hub_node), I_RATES, NUM_EPIDEMIC_SEEDS)
            save_results(output_dir, successes, success2)

if __name__ == "__main__":
    main()
    # print(count_successes("./generated-nets-2/ba-s=12349/outbreak-size-epidemics/stats-ba-n=1000-k=20-s=12349-wSIR-I=   0.70000-R=   1.00000-S=46517-SN=00011*", 11, [0.7], 1))