#!/usr/bin/env python3
"""
Generador optimizado de frames para epidemia SIR en el disco hiperbólico.
Solo resalta el momento de infección (rojo) durante unos frames, el resto blanco.
Uso:
    python gif-generator.py coords.dat edges.dat events.dat output_frames/ --t-start 0 --t-end 50 --step 0.1 [--boost nodo_inicio] [--parallel]
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

# =============================================================================
# CONFIGURACIÓN VISUAL
# =============================================================================
INFECTED_HIGHLIGHT_FRAMES = 10   # cuántos frames dura el color rojo tras infección

# ----------------------------------------------------------------------
# Funciones hiperbólicas
# ----------------------------------------------------------------------
def hyperbolic_distance_og(r1, r2, delta_theta, zeta=1.0):
    """Distancia hiperbólica nativa entre dos puntos en el disco."""
    rho1 = zeta * r1
    rho2 = zeta * r2
    cosh_dist = np.cosh(rho1) * np.cosh(rho2) - np.sinh(rho1) * np.sinh(rho2) * np.cos(delta_theta)
    cosh_dist = np.clip(cosh_dist, 1.0, None)
    return np.arccosh(cosh_dist) / zeta

def kappa_to_hyperbolic(kappa, kappa_min):
    """Convierte κ a coordenada radial hiperbólica r = ln(κ/κ_min)."""
    return np.log(kappa / kappa_min)

def hyperbolic_to_mercator(r_hiperbolico, edge_count, mu, kappa_min):
    """Convierte radio hiperbólico a coordenada en disco de Poincaré."""
    R = 2 * np.log(edge_count / (mu * np.pi * kappa_min**2))
    return R - 2 * r_hiperbolico

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
# Entrada/salida de datos
# ----------------------------------------------------------------------
def read_hyperbolic_data(archivo_coords: str, archivo_edges: str):
    """Lee el grafo y las coordenadas hiperbólicas del formato S1/H2."""
    G = nx.read_edgelist(archivo_edges)
    gen_coord = 'gen_coord' in archivo_coords

    if gen_coord:
        df = pd.read_csv(archivo_coords, sep='\\s+', comment='#',
                         names=["Vertex", "Inf.Kappa", "Inf.Hyp.Rad.", "Inf.Theta", "RealDeg.", "Exp.Deg."])
    else:
        df = pd.read_csv(archivo_coords, sep='\\s+', comment='#',
                         names=["Vertex", "Inf.Kappa", "Inf.Theta", "Inf.Hyp.Rad."])

    df['Vertex'] = df['Vertex'].astype(str)

    # Parámetros
    params = {}
    if gen_coord:
        with open(archivo_edges, 'r') as f:
            for line in f:
                if line.startswith('#') and ':' in line:
                    parts = line.strip('# ').split(':')
                    if len(parts) == 2:
                        key = parts[0].strip().lstrip('-').strip()
                        try:
                            params[key] = float(parts[1].strip())
                        except ValueError:
                            params[key] = parts[1].strip()
        params['kappa_min'] = np.min(df['Inf.Kappa'])
    else:
        with open(archivo_coords, 'r') as f:
            for line in f:
                if line.startswith('#') and ':' in line:
                    parts = line.strip('# ').split(':')
                    if len(parts) == 2:
                        key = parts[0].strip().lstrip('-').strip()
                        try:
                            params[key] = float(parts[1].strip())
                        except ValueError:
                            params[key] = parts[1].strip()
    print(params)
    df['Disc.Radius'] = hyperbolic_to_mercator(
        kappa_to_hyperbolic(df['Inf.Kappa'], params['kappa_min']),
        params['nb. vertices'], params['mu'], params['kappa_min']
    )

    R = df['Disc.Radius']
    theta = df['Inf.Theta']
    df['x0'] = (1 + R**2) / (1 - R**2)
    df['x1'] = 2 * R * np.cos(theta) / (1 - R**2)
    df['x2'] = 2 * R * np.sin(theta) / (1 - R**2)
    df['Verifi'] = -df['x0']**2 + df['x1']**2 + df['x2']**2
    df['Disc.X'] = df['Disc.Radius'] * np.cos(df['Inf.Theta'])
    df['Disc.Y'] = df['Disc.Radius'] * np.sin(df['Inf.Theta'])

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
# Renderizado de un frame (modificado: solo blanco y rojo)
# ----------------------------------------------------------------------
def mercator_epidemic_disc(data, white_coords, red_coords, filename=None, time=None):
    """Dibuja nodos blancos y resalta infectados recientes en rojo."""
    fig = Figure(dpi=200, figsize=(5, 5))
    ax = fig.add_subplot(111)

    def unpack(coords):
        if len(coords) > 0:
            return zip(*coords)
        else:
            return [], []

    x_white, y_white = unpack(white_coords)
    x_red, y_red = unpack(red_coords)

    max_val = np.max(np.abs([data['Disc.X'], data['Disc.Y']]))
    maxlims = max_val * 1.1
    ax.set_xlim(-maxlims, maxlims)
    ax.set_ylim(-maxlims, maxlims)

    ax.scatter(x_white, y_white, s=20, alpha=0.5, linewidth=0.3, c='white', edgecolors='black')
    ax.scatter(x_red, y_red, s=20, c='red')

    if time is not None:
        ax.annotate(f't={time:.3f} s', (-max_val, max_val))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    if filename:
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(filename)
    plt.close(fig)
    gc.collect()

# ----------------------------------------------------------------------
# Simulación principal
# ----------------------------------------------------------------------
def generate_epidemic_frames(df, events, output_dir, step=0.1, t_start=None, t_end=None,
                             start_node=None, lorentz_boost=False, parallel=False, max_workers=None):
    """Genera frames de la epidemia con solo infecciones resaltadas."""
    # Limpiar y crear carpeta
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Lorentz boost opcional
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

    # Mapeo rápido de coordenadas
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

    # Duración del resaltado en unidades de tiempo
    highlight_duration = INFECTED_HIGHLIGHT_FRAMES * step

    # Estado: diccionario con tiempos de infección
    infection_times = {}
    event_idx = 0
    frames = []  # (i, t, white_coords, red_coords)

    print(f"Simulando {n_steps} frames...")
    for i in tqdm(range(n_steps), desc="Simulación"):
        t = t_start + (i + 1) * step

        # Procesar solo eventos 'I'
        while event_idx < len(events_sorted) and events_sorted.iloc[event_idx]['t'] < t:
            ev = events_sorted.iloc[event_idx]
            v = ev['vertex']
            if ev['event'] == 'I':
                infection_times[v] = ev['t']   # registra momento exacto de infección
            # Ignoramos 'R' completamente
            event_idx += 1

        # Determinar nodos rojos (infectados hace menos de highlight_duration)
        red_set = set()
        for v, t_inf in infection_times.items():
            if 0.0 <= t - t_inf <= highlight_duration:
                red_set.add(v)

        white_set = all_vertices - red_set   # todos los demás son blancos

        white_coords = get_coords(white_set)
        red_coords = get_coords(red_set)

        frames.append((i, t, white_coords, red_coords))

    # Fase de renderizado
    if parallel:
        print(f"Renderizando {len(frames)} frames en paralelo...")
        def render_frame(args):
            i, t, white_c, red_c = args
            mercator_epidemic_disc(df, white_c, red_c,
                                   f"{output_dir}/sim-{i:04d}.png", t)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(render_frame, f) for f in frames]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Renderizado"):
                pass
    else:
        print(f"Renderizando {len(frames)} frames secuencialmente...")
        for i, t, white_c, red_c in tqdm(frames, desc="Renderizado"):
            mercator_epidemic_disc(df, white_c, red_c,
                                   f"{output_dir}/sim-{i:04d}.png", t)

    print(f"Frames guardados en {output_dir}")

# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generador de frames de epidemia hiperbólica")
    parser.add_argument("coords_file", help="Archivo de coordenadas (inf_coord o gen_coord)")
    parser.add_argument("edges_file", help="Archivo de aristas (para leer parámetros)")
    parser.add_argument("events_file", help="Archivo de eventos de la epidemia")
    parser.add_argument("output_dir", help="Directorio de salida para los frames PNG")
    parser.add_argument("--t-start", type=float, default=None, help="Tiempo inicial")
    parser.add_argument("--t-end", type=float, default=None, help="Tiempo final")
    parser.add_argument("--step", type=float, default=0.1, help="Paso entre frames")
    parser.add_argument("--boost", type=str, default=None, help="Nodo para centrar en origen")
    parser.add_argument("--parallel", action="store_true", help="Renderizado en paralelo")
    parser.add_argument("--max-workers", type=int, default=None, help="Número de hilos")
    args = parser.parse_args()

    print("Cargando grafo y coordenadas...")
    G, df, params = read_hyperbolic_data(args.coords_file, args.edges_file)
    print("Cargando eventos...")
    events = read_events_data(args.events_file)

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