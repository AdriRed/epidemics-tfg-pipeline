import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def mercator_disc_ax(ax: plt.Axes, data: pd.DataFrame, mark_nodes: list[str] = [], net: nx.Graph = None, isolines_nodes: list[str] = None, R: float =None, c: float =None, title: str = None, linecolor='#00000045'):
    kappa_vals = []
    positions = {v: (x, y) for _, (v, x, y) in data[['Vertex', 'Disc.X', 'Disc.Y']].iterrows()}
        
    kappa_vals = np.log10(data['Inf.Kappa'])
    
    if (net):
        for a, b in net.edges():
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.add_line(plt.Line2D([xa, xb], [ya, yb], linewidth=0.05, color=linecolor))
    x_orig, y_orig = zip(*positions.values())
    
    scatter = ax.scatter(x_orig, y_orig, c=kappa_vals, cmap='viridis', zorder=10000,
                        s=15, alpha=0.5, edgecolors='black', linewidth=0.3)
    # circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    # ax.add_patch(circle)
    
    for mark_node in mark_nodes:
        mark_data = data[data['Vertex'] == mark_node].iloc[0]
        ax.plot(mark_data['Disc.X'], mark_data['Disc.Y'], 'r*', markersize=15, markeredgecolor='black', zorder=100000)
    

    if isolines_nodes:
        for node in isolines_nodes:
            center = data[data['Vertex'] == node].iloc[0]
            r, theta = center['Disc.Radius'], center['Inf.Theta']
            dibujar_isolineas(ax, r, theta, R=R, c=c, resolucion=3000)

    if (title):
        ax.set_title(title)

    # ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return scatter

def mercator_disc(data: pd.DataFrame, mark_nodes: list[str] = [], net: nx.Graph = None, isolines_nodes: list[str] = None, R=None, c=None, title: str = None):
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi=100)
    max_val_x = np.max(np.abs(data['Disc.X']))*1.1
    max_val_y = np.max(np.abs(data['Disc.Y']))*1.1
    maxval = np.max([max_val_x, max_val_y])

    ax.set_xlim(-maxval, maxval)
    ax.set_ylim(-maxval, maxval)

    mercator_disc_ax(ax, data, mark_nodes, net, isolines_nodes, R, c)
    # plt.colorbar(scatter, ax=ax, label='log10(κ)')
    plt.show()
    plt.close(fig)
    plt.rcParams['text.usetex'] = True
    


def dibujar_isolineas(ax, r_centro, theta_centro, R=1, c=-1, zeta=1.0,
                      niveles=None, resolucion=300, **kwargs_contour):
    """
    Dibuja sobre el eje 'ax' isolíneas de distancia hiperbólica
    respecto al punto (r_centro, theta_centro).

    Parámetros
    ----------
    ax : matplotlib.axes.Axes
        Eje donde se dibujarán las curvas.
    r_centro, theta_centro : float
        Coordenadas nativas del punto de referencia (r_h, theta).
    zeta : float
        Factor de escala de la distancia (por defecto 1.0).
    niveles : array-like o None
        Lista de distancias donde dibujar isolíneas.
        Si es None, se eligen automáticamente.
    resolucion : int
        Número de puntos en cada dirección para la malla.
    kwargs_contour : dict
        Argumentos adicionales para `ax.contour` (color, linewidth, etc.)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from . import hyperbolic as hyp

    # Límites actuales del gráfico
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Crear malla cartesiana
    x = np.linspace(xlim[0], xlim[1], resolucion)
    y = np.linspace(ylim[0], ylim[1], resolucion)
    X, Y = np.meshgrid(x, y)
    # ax.scatter(X, Y)
    # Convertir a coordenadas polares nativas
    R_hyp = np.hypot(X, Y)
    # R = 2.0 * np.arctanh(np.clip(R, 0, 1 - 1e-12))
    Theta = np.arctan2(Y, X)

    # Diferencia angular con el punto de referencia
    delta_theta = Theta - theta_centro

    # Distancia hiperbólica nativa (r_a = r_centro, r_b = R)
    dist = hyp.hyperbolic_distance_og(r_centro, R_hyp, delta_theta)
    # display(dist)
    # dist = link_probability_og(dist, R, c)
    # Niveles automáticos si no se especifican
    if niveles is None:
        max_dist = np.nanmax(dist)
        niveles = np.linspace(0, max_dist, 20)[1:]  # sin el cero
    orig_map=plt.cm.get_cmap('Reds')

    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed()
    # Dibujar curvas
    # cs = ax.scatter(X, Y, c=dist, alpha=0.3, cmap=reversed_map, **kwargs_contour)
    cs = ax.contourf(X, Y, dist, alpha=0.4, cmap=reversed_map, levels=niveles, **kwargs_contour)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%1.2f')

    return cs


def mercator_epidemic_disc(data: pd.DataFrame, susceptible_coords, infected_coords, recovered_coords, filename: str = None, time:str = None):
    import numpy as np
    import matplotlib.pyplot as plt
    import gc
    plt.rcParams['text.usetex'] = False
    fig, ax = None, None
    if (filename):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg 

        fig = Figure(figsize=(14, 12), dpi=100)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    x_orig_white, y_orig_white = [], []
    x_orig_pink, y_orig_pink = [], []
    x_orig_red, y_orig_red = [], [] 
    if len(infected_coords) > 0:
        x_orig_red, y_orig_red = zip(*infected_coords) 
    if (len(recovered_coords)):
        x_orig_pink, y_orig_pink = zip(*recovered_coords)
    if len(susceptible_coords) > 0:
        x_orig_white, y_orig_white = zip(*susceptible_coords)

    max_val = np.max(np.abs([data['Disc.X'], data['Disc.Y']]))*1.1

    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.scatter(x_orig_white, y_orig_white, s=15, alpha=0.5, linewidth=0.3, c='white', edgecolors='black')
    ax.scatter(x_orig_pink, y_orig_pink, alpha=0.1, s=15, c='blue')
    ax.scatter(x_orig_red, y_orig_red, s=15, c='red')
    
    if (time is not None):
        ax.set_title(f"t={time:.03f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if (filename):
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(filename)   # Guarda directamente
    else:
        plt.show()
    plt.rcParams['text.usetex'] = True
    gc.collect()