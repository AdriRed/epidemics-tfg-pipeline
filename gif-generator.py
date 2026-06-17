#!/usr/bin/env python3
"""
Generador optimizado de frames para epidemia SIR en el disco hiperbólico.
Uso:
    python anim_epidemia_opt.py coords.dat events.dat output_frames/ --t-start 0 --t-end 50 --step 0.1 [--boost nodo_inicio] [--parallel]
"""

import os
import shutil
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Sin GUI, necesario para paralelismo
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, Tuple, Optional
import networkx as nx
import gc

# ----------------------------------------------------------------------
# Funciones hiperbólicas (reproducidas aquí para independencia)
# ----------------------------------------------------------------------
def hyperbolic_distance_og(r1, r2, delta_theta, zeta=1.0):
    """Distancia hiperbólica nativa entre dos puntos en el disco."""
    rho1 = zeta * r1
    rho2 = zeta * r2
    cosh_dist = np.cosh(rho1) * np.cosh(rho2) - np.sinh(rho1) * np.sinh(rho2) * np.cos(delta_theta)
    cosh_dist = np.clip(cosh_dist, 1.0, None)
    return np.arccosh(cosh_dist) / zeta

def kappa_to_hyperbolic(kappa, kappa_min):
    """Convierte kappa a radio hiperbólico nativo."""
    return (1.0 / (kappa - kappa_min)) ** 0.5  # Aproximación típica, ajusta según tu modelo

def hyperbolic_to_mercator(r_hyp, N, mu, kappa_min):
    """Transforma radio nativo a radio del disco de Poincaré (proyección conforme)."""
    # Ajustar según tu mapeo real. Aquí va una fórmula común:
    return np.tanh(r_hyp / 2.0)  # Mapeo de Beltrami-Poincaré; si no, usa la que tenías

def centrar_en_origen(r, theta, r_centro, theta_centro, zeta=1.0):
    """Isometría hiperbólica que lleva (r_centro, theta_centro) al origen."""
    rho0 = zeta * r_centro
    nx = np.cos(theta_centro)
    ny = np.sin(theta_centro)

    rho = zeta * r
    t = np.cosh(rho)
    x = np.sinh(rho) * np.cos(theta)
    y = np.sinh(rho) * np.sin(theta)

    dot = nx * x + ny * y
    ch0 = np.cosh(rho0)
    sh0 = np.sinh(rho0)

    t_prime = ch0 * t - sh0 * dot
    x_prime = -sh0 * nx * t + x + (ch0 - 1) * nx * dot
    y_prime = -sh0 * ny * t + y + (ch0 - 1) * ny * dot

    t_prime_clip = np.clip(t_prime, 1.0, None)
    r_nuevo = np.arccosh(t_prime_clip) / zeta
    theta_nuevo = np.arctan2(y_prime, x_prime)
    return r_nuevo, theta_nuevo

# ----------------------------------------------------------------------
# Funciones de entrada/salida de datos (reproducidas)
# ----------------------------------------------------------------------
def read_hyperbolic_data(archivo_coords, archivo_edges):
    """Lee el grafo y coordenadas. Devuelve G, df, params."""
    G = nx.read_edgelist(archivo_edges)
    df = pd.read_csv(archivo_coords, sep='\\s+', comment='#',
                     names=["Vertex", "Inf.Kappa", "Inf.Theta", "Inf.Hyp.Rad."])
    df['Vertex'] = df['Vertex'].astype(str)

    # Parámetros desde el archivo de coordenadas
    params = {}
    with open(archivo_coords, 'r') as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                parts = line.strip('# ').split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lstrip('-')
                    try:
                        params[key] = float(parts[1].strip())
                    except ValueError:
                        params[key] = parts[1].strip()

    kappa_min = np.min(df['Inf.Kappa'])
    params['kappa_min'] = kappa_min

    # Calcular radio en el disco
    df['Disc.Radius'] = hyperbolic_to_mercator(
        kappa_to_hyperbolic(df['Inf.Kappa'], kappa_min),
        params['nb. vertices'], params['mu'], kappa_min
    )
    theta = df['Inf.Theta']
    df['Disc.X'] = df['Disc.Radius'] * np.cos(theta)
    df['Disc.Y'] = df['Disc.Radius'] * np.sin(theta)

    return G, df, params

def read_events_data(events_file):
    """Lee archivo de eventos (t, vertex, event)."""
    with open(events_file, 'r') as f:
        lines = f.readlines()
    first_data_line = 0
    for step, line in enumerate(lines):
        if not line.lstrip().startswith('#'):
            first_data_line = step
            break
    events = pd.read_csv(events_file, sep='\\s+', skiprows=first_data_line,
                         names=['t', 'vertex', 'event'])
    events['vertex'] = events['vertex'].astype(str)
    return events

# ----------------------------------------------------------------------
# Función de renderizado de un frame de epidemia
# ----------------------------------------------------------------------
def mercator_epidemic_disc(data, susceptible_coords, infected_coords, recovered_coords,
                           filename=None, time=None):
    """Dibuja los puntos susceptibles, infectados y recuperados en el disco."""
    fig = Figure(figsize=(14, 12), dpi=100)
    ax = fig.add_subplot(111)

    # Preparar coordenadas
    def unpack(coords):
        if len(coords) > 0:
            return zip(*coords)
        else:
            return [], []

    x_sus, y_sus = unpack(susceptible_coords)
    x_inf, y_inf = unpack(infected_coords)
    x_rec, y_rec = unpack(recovered_coords)

    max_val = np.max(np.abs([data['Disc.X'], data['Disc.Y']])) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.scatter(x_sus, y_sus, s=15, alpha=0.5, linewidth=0.3, c='white', edgecolors='black')
    ax.scatter(x_inf, y_inf, s=15, c='red')
    ax.scatter(x_rec, y_rec, s=15, alpha=0.1, c='blue')

    if time is not None:
        ax.set_title(f"t={time:.03f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(filename)
    else:
        plt.close(fig)  # Nunca se usa sin filename en el script, pero por completitud
    plt.close(fig)
    gc.collect()

# ----------------------------------------------------------------------
# Lógica principal: simulación incremental + renderizado paralelo
# ----------------------------------------------------------------------
def generate_epidemic_frames(df, events, output_dir, step=0.1, t_start=None, t_end=None,
                             start_node=None, lorentz_boost=False, parallel=False, max_workers=None):
    """
    Genera frames de la epidemia y los guarda en output_dir.
    """
    # Limpiar y crear carpeta de salida
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Lorentz boost si se pide
    if lorentz_boost and start_node is not None:
        selected = df[df['Vertex'] == start_node].iloc[0]
        r, theta = selected['Disc.Radius'], selected['Inf.Theta']
        r_changed, theta_changed = centrar_en_origen(df['Disc.Radius'].values,
                                                     df['Inf.Theta'].values, r, theta)
        df = df.copy()
        df['Disc.Radius'] = r_changed
        df['Inf.Theta'] = theta_changed
        df['Disc.X'] = df['Disc.Radius'] * np.cos(df['Inf.Theta'])
        df['Disc.Y'] = df['Disc.Radius'] * np.sin(df['Inf.Theta'])

    # Mapeo de vértice a índice para acceso rápido a coordenadas
    x_arr = df['Disc.X'].values
    y_arr = df['Disc.Y'].values
    vertex_to_idx = {v: i for i, v in enumerate(df['Vertex'])}
    all_vertices = set(df['Vertex'])

    def get_coords(vertex_set):
        if not vertex_set:
            return np.empty((0, 2))
        indices = [vertex_to_idx[v] for v in vertex_set]
        return np.column_stack((x_arr[indices], y_arr[indices]))

    # Ventana temporal
    if t_start is None:
        t_start = events["t"].min()
    if t_end is None:
        t_end = events["t"].max()
    n_steps = math.ceil((t_end - t_start) / step)

    # Eventos ordenados
    events_sorted = events.sort_values("t").reset_index(drop=True)
    n_events = len(events_sorted)

    # Estado inicial
    susceptible = all_vertices.copy()
    infected = set()
    recovered = set()

    # Bucle de simulación: avance incremental
    frames = []  # (i, t, susc_coords, inf_coords, rec_coords)
    event_idx = 0
    print(f"Simulando {n_steps} frames...")
    for i in tqdm(range(n_steps), desc="Simulación"):
        t = t_start + (i + 1) * step

        # Procesar eventos nuevos hasta tiempo t
        while event_idx < n_events and events_sorted.iloc[event_idx]['t'] < t:
            ev = events_sorted.iloc[event_idx]
            v = ev['vertex']
            if ev['event'] == 'I':
                susceptible.discard(v)
                recovered.discard(v)
                infected.add(v)
            elif ev['event'] == 'R':
                infected.discard(v)
                susceptible.discard(v)
                recovered.add(v)
            event_idx += 1

        # Obtener coordenadas de cada grupo
        susc_coords = get_coords(susceptible)
        inf_coords = get_coords(infected)
        rec_coords = get_coords(recovered)

        frames.append((i, t, susc_coords, inf_coords, rec_coords))

    # Fase de renderizado
    if parallel:
        print(f"Renderizando {len(frames)} frames en paralelo...")
        def render_frame(args):
            i, t, susc, inf, rec = args
            mercator_epidemic_disc(df, susc, inf, rec,
                                   f"{output_dir}/sim-{i:04d}.png", t)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(render_frame, f) for f in frames]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Renderizado"):
                pass
    else:
        print(f"Renderizando {len(frames)} frames secuencialmente...")
        for i, t, susc, inf, rec in tqdm(frames, desc="Renderizado"):
            mercator_epidemic_disc(df, susc, inf, rec,
                                   f"{output_dir}/sim-{i:04d}.png", t)

    print(f"Frames guardados en {output_dir}")

# ----------------------------------------------------------------------
# Interfaz de línea de comandos
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generador de frames de epidemia hiperbólica")
    parser.add_argument("coords_file", help="Archivo de coordenadas (inf_coord)")
    parser.add_argument("edges_file", help="Archivo de aristas (para leer parámetros)")
    parser.add_argument("events_file", help="Archivo de eventos de la epidemia")
    parser.add_argument("output_dir", help="Directorio de salida para los frames PNG")
    parser.add_argument("--t-start", type=float, default=None, help="Tiempo inicial (por defecto min de eventos)")
    parser.add_argument("--t-end", type=float, default=None, help="Tiempo final (por defecto max de eventos)")
    parser.add_argument("--step", type=float, default=0.1, help="Paso entre frames (default 0.1)")
    parser.add_argument("--boost", type=str, default=None, help="Nodo para centrar en origen (Lorentz boost)")
    parser.add_argument("--parallel", action="store_true", help="Renderizar frames en paralelo")
    parser.add_argument("--max-workers", type=int, default=None, help="Número de hilos (por defecto automático)")
    args = parser.parse_args()

    # Cargar datos
    print("Cargando grafo y coordenadas...")
    G, df, params = read_hyperbolic_data(args.coords_file, args.edges_file)
    print("Cargando eventos...")
    events = read_events_data(args.events_file)

    # Generar frames
    generate_epidemic_frames(
        df=df,
        events=events,
        output_dir=args.output_dir,
        step=args.step,
        t_start=args.t_start,
        t_end=args.t_end,
        start_node=args.boost,
        lorentz_boost=args.boost is not None,
        parallel=args.parallel,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()