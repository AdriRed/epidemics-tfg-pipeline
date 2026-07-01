#!/usr/bin/env python3
"""
Generador optimizado de frames para epidemia SIR en disco hiperbólico 
con layout radial al estilo Brockmann & Helbing (2013).

- Nodo raíz en el centro.
- Capas concéntricas según vecindad (primeros vecinos, segundos, ...).
- Radios proporcionales a la distancia hiperbólica acumulada.
- Nodos recién infectados en rojo durante unos frames; el resto blanco.

Uso:
    python gif-generator.py coords.dat edges.dat events.dat output_frames/ \\
        --t-start 0 --t-end 50 --step 0.1 [--boost nodo_inicio] [--parallel]
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
INFECTED_HIGHLIGHT_FRAMES = 25   # cuántos frames dura el color rojo tras infección

# =============================================================================
# 1. FUNCIONES HIPERBÓLICAS ORIGINALES (para leer datos)
# =============================================================================
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

# =============================================================================
# 2. LECTURA DE DATOS (coordenadas, aristas, eventos)
# =============================================================================
def read_hyperbolic_data(archivo_coords: str, archivo_edges: str):
    """Lee el grafo y las coordenadas hiperbólicas del formato S1/H2."""
    G = nx.read_weighted_edgelist(archivo_edges+"_weight_-1x10^-1")
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

    # Calcular distancias hiperbólicas para las aristas
    # (asumiendo que la distancia ya está en G[u][v]['distance'] o la calculamos)
    # Si no existe, la calculamos con las coordenadas
    total_dist = 0
    total_weight = 0
    for u, v in G.edges():
        if 'distance' not in G[u][v]:
            u_row = df[df['Vertex'] == str(u)].iloc[0]
            v_row = df[df['Vertex'] == str(v)].iloc[0]
            r_u = u_row['Disc.Radius']
            r_v = v_row['Disc.Radius']
            theta_u = u_row['Inf.Theta']
            theta_v = v_row['Inf.Theta']
            delta_theta = abs(theta_u - theta_v)
            dist = hyperbolic_distance_og(r_u, r_v, delta_theta)
            G[u][v]['distance'] = dist
            total_weight += G[u][v]['weight']
            total_dist += dist
    
    for u, v in G.edges():
        G[u][v]['D_ij'] = 1 - np.log(G[u][v]['weight']/total_weight)

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

# =============================================================================
# 3. LAYOUT RADIAL AL ESTILO BROCKMANN & HELBING (2013)
# =============================================================================
import networkx as nx
import numpy as np

def radial_layout_brockmann_helbing(G, root, weight='distance', angle_span=2*np.pi, scale=0.9):
    """
    Genera un layout radial estilo Brockmann & Helbing (2013).

    Parámetros:
    - G: grafo de NetworkX (no dirigido) con atributo 'distance' en aristas.
    - root: nodo raíz (en el centro).
    - weight: nombre del atributo de arista con la distancia.
    - angle_span: ángulo total para distribuir los nodos (2π por defecto).
    - scale: factor de escala para que los nodos no toquen el borde (0.9 por defecto).

    Retorna:
    - pos: dict {nodo: (x, y)} con coordenadas en el disco (normalizadas a [-1,1]).
    """
    # 1. Calcular capas BFS (profundidad topológica) y distancia más corta ponderada (Dijkstra)
    layers = {root: 0}
    queue = [root]

    # Capas mediante BFS (sin pesos) – se usan para agrupar y asignar ángulos
    for node in queue:
        for neighbor in G.neighbors(node):
            if neighbor not in layers:
                layers[neighbor] = layers[node] + 1
                queue.append(neighbor)

    # Distancias más cortas ponderadas desde la raíz usando Dijkstra
    shortest_dist = nx.single_source_dijkstra_path_length(G, root, weight=weight)
    
    # Para nodos no alcanzables (por si acaso) se les asigna una distancia infinita,
    # pero los ignoraremos en el layout.
    for node in G.nodes():
        if node not in layers:
            layers[node] = -1
        if node not in shortest_dist:
            shortest_dist[node] = float('inf')

    # 2. Agrupar nodos por capa (solo los alcanzables)
    nodes_by_layer = {}
    for node, layer in layers.items():
        if layer >= 0 and shortest_dist[node] != float('inf'):
            nodes_by_layer.setdefault(layer, []).append(node)

    # 3. Asignar radios (distancias radiales) normalizadas al intervalo [0, 1]
    #    usando las distancias más cortas ponderadas
    finite_dists = [d for d in shortest_dist.values() if d != float('inf')]
    max_dist = max(finite_dists) if finite_dists else 1.0
    if max_dist == 0:
        max_dist = 1.0

    radii = {}
    for node in G.nodes():
        if node == root:
            radii[node] = 0.0
        else:
            d = shortest_dist.get(node, float('inf'))
            if d == float('inf'):
                radii[node] = 0.0   # nodos no alcanzables se colocan en el centro (o se ignoran)
            else:
                radii[node] = (d / max_dist) * scale

    # 4. Asignar ángulos de forma uniforme dentro de cada capa
    pos = {}
    for layer, nodes in nodes_by_layer.items():
        if layer == 0:
            pos[root] = (0.0, 0.0)
            continue

        n = len(nodes)
        if n == 0:
            continue

        angles = np.linspace(0, angle_span, n, endpoint=False)

        for i, node in enumerate(nodes):
            r = radii[node]
            theta = angles[i]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pos[node] = (x, y)

    return pos
# =============================================================================
# 4. RENDERIZADO DE UN FRAME (con layout radial)
# =============================================================================
def render_frame_radial(ax, white_coords, red_coords, time=None):
    """Dibuja nodos blancos y resalta infectados recientes en rojo."""
    def unpack(coords):
        if len(coords) > 0:
            return zip(*coords)
        else:
            return [], []

    x_white, y_white = unpack(white_coords)
    x_red, y_red = unpack(red_coords)

    # Limpiar ejes
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Dibujar círculo de borde (opcional)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', linewidth=0.5)
    ax.add_artist(circle)

    # Nodos blancos
    if len(x_white) > 0:
        ax.scatter(x_white, y_white, s=10, alpha=0.5, linewidth=0.3,
                   c='white', edgecolors='black')

    # Nodos rojos (recién infectados)
    if len(x_red) > 0:
        ax.scatter(x_red, y_red, s=10, c='red')

    # Mostrar tiempo
    if time is not None:
        ax.annotate(f't={time:.3f} s', (-0.95, 0.95), fontsize=10)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def mercator_epidemic_disc_radial(df, white_coords, red_coords, filename=None, time=None):
    """Versión del renderizado que usa el layout radial."""
    fig = Figure(dpi=200, figsize=(5, 5))
    ax = fig.add_subplot(111)
    render_frame_radial(ax, white_coords, red_coords, time)
    fig.tight_layout()
    if filename:
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(filename)
    plt.close(fig)
    gc.collect()

# =============================================================================
# 5. SIMULACIÓN PRINCIPAL (con layout radial)
# =============================================================================
def generate_epidemic_frames(df, G: nx.Graph, events, output_dir, step=0.1, t_start=None, t_end=None,
                             start_node=None, parallel=False, max_workers=None):
    """
    Genera frames de la epidemia con layout radial al estilo Brockmann & Helbing.
    """
    # Limpiar y crear carpeta
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 1. Elegir nodo raíz (si no se da, el primer infectado)
    if start_node is None:
        # Primer evento 'I' en los eventos
        first_infection = events[events['event'] == 'I'].iloc[0]['vertex']
        root = str(first_infection)
    else:
        root = str(start_node)

    print(f"Nodo raíz (centro): {root}")
    # 2. Calcular layout radial
    pos = radial_layout_brockmann_helbing(G, root, weight='D_ij', scale=0.9)
    
    # 3. Mapeo rápido de coordenadas radiales
    x_arr = np.array([pos[v][0] for v in G.nodes()])
    y_arr = np.array([pos[v][1] for v in G.nodes()])
    vertex_to_idx = {v: i for i, v in enumerate(G.nodes())}
    all_vertices = set(G.nodes())

    def get_coords(vertex_set):
        if not vertex_set:
            return np.empty((0, 2))
        indices = [vertex_to_idx[v] for v in vertex_set]
        return np.column_stack((x_arr[indices], y_arr[indices]))

    # 4. Ventana temporal
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

    # 5. Fase de renderizado
    if parallel:
        print(f"Renderizando {len(frames)} frames en paralelo...")
        def render_frame(args):
            i, t, white_c, red_c = args
            mercator_epidemic_disc_radial(df, white_c, red_c,
                                          f"{output_dir}/sim-{i:04d}.png", t)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(render_frame, f) for f in frames]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Renderizado"):
                pass
    else:
        print(f"Renderizando {len(frames)} frames secuencialmente...")
        for i, t, white_c, red_c in tqdm(frames, desc="Renderizado"):
            mercator_epidemic_disc_radial(df, white_c, red_c,
                                          f"{output_dir}/sim-{i:04d}.png", t)

    print(f"Frames guardados en {output_dir}")

# =============================================================================
# 6. PUNTO DE ENTRADA
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generador de frames de epidemia con layout radial")
    parser.add_argument("coords_file", help="Archivo de coordenadas (inf_coord o gen_coord)")
    parser.add_argument("edges_file", help="Archivo de aristas (para leer parámetros)")
    parser.add_argument("events_file", help="Archivo de eventos de la epidemia")
    parser.add_argument("output_dir", help="Directorio de salida para los frames PNG")
    parser.add_argument("--t-start", type=float, default=None, help="Tiempo inicial")
    parser.add_argument("--t-end", type=float, default=None, help="Tiempo final")
    parser.add_argument("--step", type=float, default=0.1, help="Paso entre frames")
    parser.add_argument("--boost", type=str, default=None, help="Nodo raíz para el layout radial")
    parser.add_argument("--parallel", action="store_true", help="Renderizado en paralelo")
    parser.add_argument("--max-workers", type=int, default=None, help="Número de hilos")
    args = parser.parse_args()

    print("Cargando grafo y coordenadas...")
    G, df, params = read_hyperbolic_data(args.coords_file, args.edges_file)
    print("Cargando eventos...")
    events = read_events_data(args.events_file)

    # Nota: ya no usamos Lorentz boost porque el layout radial ya centra en el nodo raíz.
    # El parámetro --boost se usa para elegir el nodo central.

    generate_epidemic_frames(
        df=df,
        G=G,
        events=events,
        output_dir=args.output_dir,
        step=args.step,
        t_start=args.t_start,
        t_end=args.t_end,
        start_node=args.boost,
        parallel=args.parallel,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()