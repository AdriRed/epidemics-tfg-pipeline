import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pandas as pd
import re

# =============================================================================
# Configuración
# =============================================================================
BASE_DIR = "./generated-nets"
MODELS = ["er", "ba", "conf", "s1h2"]

# Definición de bins
DIST_MIN, DIST_MAX = 0.0, 1.15
TIME_MIN, TIME_MAX = 0.0, 5.0
NUM_DIST_BINS = 60          # resolución espacial
NUM_TIME_BINS = 100          # resolución temporal (0.02 por bin)

dist_bins = np.linspace(DIST_MIN, DIST_MAX, NUM_DIST_BINS + 1)
time_bins = np.linspace(TIME_MIN, TIME_MAX, NUM_TIME_BINS + 1)

# =============================================================================
# Función para leer todos los archivos source_*.dat de una carpeta de seed
# =============================================================================
def read_seed_data(seed_folder, hill_params_file):
    """
    seed_folder: ruta a la carpeta que contiene 'infection_arrivals_by_source'
    Retorna: (distances_array, times_array) para todos los eventos de esa seed
    """
    arrivals_dir = os.path.join(seed_folder, "infection_arrivals_by_source")
    if not os.path.isdir(arrivals_dir):
        return np.array([]), np.array([])
    
    files = glob.glob(os.path.join(arrivals_dir, "source_*.dat"))
    if not files:
        return np.array([]), np.array([])
    
    distances = []
    times = []
    seed = float(re.search(
        r's=\s*(\d+)\s*',
        seed_folder
    ).group(1))

    for fpath in tqdm(files, desc=f"  Leyendo {os.path.basename(seed_folder)}", leave=False):
        start_node = float(re.search(
            r'source_\s*(\d+)\s*',
            fpath
        ).group(1))
        mask = (hill_params_file['start_node'].astype(float) == start_node) & (hill_params_file['seed'].astype(float) == seed)
        masked = hill_params_file[mask]
        if (masked.empty):
            raise "Empty!"
        if (len(masked) > 1):
            raise "More than 1!"
        L, Omega = masked[['L', 'omega']].to_numpy()[0]
        data = np.loadtxt(fpath, comments='#')
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        distances.extend(data[:, 0]/L)
        times.extend(data[:, 1]/Omega)
    
    return np.array(distances), np.array(times)

# =============================================================================
# Función para calcular histograma normalizado por tiempo para un conjunto de eventos
# =============================================================================
def compute_normalized_histogram(distances, times, dist_bins, time_bins):
    """
    distances, times: arrays 1D
    Retorna: histograma 2D normalizado por columna (tiempo)
    """
    if len(distances) == 0:
        # Retornar matriz de ceros si no hay eventos
        return np.zeros((len(dist_bins)-1, len(time_bins)-1))
    
    H, _, _ = np.histogram2d(distances, times, bins=[dist_bins, time_bins])
    total_per_time = H.sum(axis=0, keepdims=True)
    total_per_time[total_per_time == 0] = 1
    H_norm = H / total_per_time
    return H_norm

# =============================================================================
# Procesar cada modelo: buscar todas las seeds, calcular promedio de histogramas
# =============================================================================
results = {}
for model in MODELS:
    print(f"\nProcesando modelo: {model}")
    hill_params_file = pd.read_csv(f'{BASE_DIR}/{model}-hill-params.csv', sep='\t')

    # Buscar todas las carpetas que coincidan con "modelo-s=*"
    pattern = os.path.join(BASE_DIR, f"{model}-s=*")
    seed_folders = glob.glob(pattern)
    if not seed_folders:
        print(f"  No se encontraron carpetas para {model}")
        results[model] = None
        continue
    
    print(f"  Se encontraron {len(seed_folders)} seeds: {[os.path.basename(f) for f in seed_folders]}")
    
    normalized_histograms = []
    for seed_folder in seed_folders:
        dists, times = read_seed_data(seed_folder, hill_params_file)
        print(f"    {os.path.basename(seed_folder)}: {len(dists)} eventos")
        H_norm = compute_normalized_histogram(dists, times, dist_bins, time_bins)
        normalized_histograms.append(H_norm)
    
    # Promediar los histogramas de todas las seeds (media aritmética)
    avg_H = np.mean(normalized_histograms, axis=0)
    results[model] = avg_H
    
    # Guardar el array promedio
    out_file = f"generated-nets/hist2d_{model}_avgOverSeeds_norm.npy"
    np.save(out_file, avg_H)
    print(f"  Histograma promedio guardado en {out_file}")

# # =============================================================================
# # Opcional: Graficar mapas de calor
# # =============================================================================
# def plot_heatmap(H_norm, dist_bins, time_bins, title):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
#     time_centers = (time_bins[:-1] + time_bins[1:]) / 2
#     im = ax.pcolormesh(time_centers, dist_centers, H_norm, 
#                        shading='auto', cmap='viridis', vmin=0, vmax=1)
#     ax.set_xlabel("Tiempo de infección")
#     ax.set_ylabel("Distancia hiperbólica")
#     ax.set_title(title)
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label("Probabilidad (normalizada por tiempo)")
#     plt.tight_layout()
#     return fig

# for model, H in results.items():
#     if H is not None:
#         fig = plot_heatmap(H, dist_bins, time_bins, f"Modelo {model.upper()} (promedio sobre seeds)")
#         fig.savefig(f"heatmap_{model}_avgSeeds.png", dpi=150)
#         plt.close(fig)

# print("\nProceso completado.")