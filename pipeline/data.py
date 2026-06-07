import numpy as np
import pandas as pd
import networkx as nx
import typing as t

def build_epidemics_filename(type: str, edges_file: str, weighted: bool, model: str, i_rate: float, r_rate: float, seed: int, start_node:int, old=False) -> str:
    from pathlib import Path
    return f'{type}-{Path(edges_file).stem}-{'w' if weighted else ''}{model}-I={i_rate:10.5f}-R={r_rate:10.5f}-S={f'{seed:10d}' if not old else f'{seed:5d}'}-SN={start_node:05d}.dat'

def build_events_filename(edges_file: str, weighted: bool, model: str, i_rate: float, r_rate: float, seed: int, start_node: int, old=False) -> str:
    return build_epidemics_filename('events', edges_file, weighted, model, i_rate, r_rate, seed, start_node, old)

def build_stats_filename(edges_file: str, weighted: bool, model: str, i_rate: float, r_rate: float, seed: int, start_node: int, old=False) -> str:
    return build_epidemics_filename('stats', edges_file, weighted, model, i_rate, r_rate, seed, start_node, old)


def read_events_data(events_file: str) -> pd.DataFrame:
    with open(events_file, 'r') as f:
            lines = f.readlines()
        
        # Encontrar la primera línea de datos
    first_data_line = 0
    for step, line in enumerate(lines):
        if not line.lstrip().startswith('#'):
            first_data_line = step
            break
    events = pd.read_csv(events_file, 
                            sep='\\s+', skiprows=first_data_line, names=['t', 'vertex', 'event'])
    events['vertex'] = events['vertex'].astype(str)

    return events

def read_stats_data(stats_file: str) -> pd.DataFrame:
    with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Encontrar la primera línea de datos
    first_data_line = 0
    for step, line in enumerate(lines):
        if not line.lstrip().startswith('#'):
            first_data_line = step
            break
    stats = pd.read_csv(stats_file, 
                            sep='\\s+', skiprows=first_data_line, names=['t','idens', 'rdens', 'irate', 'rrate'])
    return stats


def read_hyperbolic_data(archivo_coords: str, archivo_edges: str, gen_coord=False) -> t.Tuple[nx.Graph, pd.DataFrame, dict]:
    """
    Lee el grafo y las coordenadas hiperbólicas del formato S1/H2
    """
    import networkx as nx
    import pandas as pd
    from . import hyperbolic as hyp
    import numpy as np
    # Leer grafo
    G = nx.read_edgelist(archivo_edges)
    
    # Leer coordenadas
    df = None
    if (gen_coord):
        # when reading gen_coord instead of inf_coord
        df = pd.read_csv(archivo_coords, sep='\\s+', comment='#', 
                     names=["Vertex", "Inf.Kappa", "Inf.Hyp.Rad.", "Inf.Theta", "RealDeg.", "Exp.Deg."])
    else:
        df = pd.read_csv(archivo_coords, sep='\\s+', comment='#', 
                     names=["Vertex", "Inf.Kappa", "Inf.Theta", "Inf.Hyp.Rad."])
    
    # Convertir Vertex a string
    df['Vertex'] = df['Vertex'].astype(str)
    # df = df.set_index('Vertex')
    # Leer parámetros del archivo
    params = {}
    if (gen_coord):
        # when reading gen_coord instead of inf_coord
        with open(archivo_edges, 'r') as f:
                    for line in f:
                        if line.startswith('#') and ':' in line:
                            parts = line.strip('# ').split(':')
                            if len(parts) == 2:
                                key = parts[0].strip()
                                if (key.startswith('-')):
                                    key = key[1:].strip()
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
                        key = parts[0].strip()
                        if (key.startswith('-')):
                            key = key[1:].strip()
                        try:
                            params[key] = float(parts[1].strip())
                        except ValueError:
                            params[key] = parts[1].strip()
    
    df['Disc.Radius'] = hyp.hyperbolic_to_mercator(hyp.kappa_to_hyperbolic(df['Inf.Kappa'], params['kappa_min']), params['nb. vertices'], params['mu'], params['kappa_min'])
    
    R = df['Disc.Radius']
    theta = df['Inf.Theta']
    df['x0'] = (1+R**2)/(1-R**2)
    df['x1'] = 2*R*np.cos(theta)/(1-R**2)
    df['x2'] = 2*R*np.sin(theta)/(1-R**2)

    df['Verifi'] = -df['x0']**2+df['x1']**2+df['x2']**2
    df['Disc.X'] = df['Disc.Radius']*np.cos(df['Inf.Theta'])
    df['Disc.Y'] = df['Disc.Radius']*np.sin(df['Inf.Theta'])
    
    return G, df, params

def get_most_popular_node(G: nx.Graph) -> str:
    return max(G.degree(), key=lambda x: x[1])[0]
