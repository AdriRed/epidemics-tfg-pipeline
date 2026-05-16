import numpy as np
import pandas as pd
import networkx as nx
import typing as t

def read_hyperbolic_data(archivo_coords: str, archivo_edges: str) -> t.Tuple[nx.Graph, pd.DataFrame, dict]:
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
    df = pd.read_csv(archivo_coords, sep='\\s+', comment='#', 
                     names=["Vertex", "Inf.Kappa", "Inf.Theta", "Inf.Hyp.Rad."])
    
    # Convertir Vertex a string
    df['Vertex'] = df['Vertex'].astype(str)
    # df = df.set_index('Vertex')
    # Leer parámetros del archivo
    params = {}
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
