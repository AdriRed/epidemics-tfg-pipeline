import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def mercator_disc_ax(ax: plt.Axes, data: pd.DataFrame, mark_nodes: list[str] = [], net: nx.Graph = None, isolines_nodes: list[str] = None, R: float =None, c: float =None, title: str = None, linecolor='#00000045'):
    positions = {v: (x, y) for _, (v, x, y) in data[['Vertex', 'Disc.X', 'Disc.Y']].iterrows()}
    kappa_vals = np.log10(data['Inf.Kappa'])
    x_orig, y_orig = zip(*positions.values())
    
    # ARISTAS: zorder muy bajo
    if net:
        for a, b in net.edges():
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.add_line(plt.Line2D([xa, xb], [ya, yb], 
                                  linewidth=0.05, color=linecolor, 
                                  zorder=0))  # <- Debajo de todo
    
    # NODOS: zorder muy alto
    scatter = ax.scatter(x_orig, y_orig, c=kappa_vals, cmap='viridis', 
                        s=15, edgecolors='black', linewidth=0.3,
                        zorder=10)  # <- Encima de las aristas
    
    # NODOS MARCADOS: zorder aún más alto
    for mark_node in mark_nodes:
        mark_data = data[data['Vertex'] == mark_node].iloc[0]
        ax.plot(mark_data['Disc.X'], mark_data['Disc.Y'], 'r*', 
               markersize=15, markeredgecolor='black', 
               zorder=20)  # <- Encima de todo
    
    # Isolíneas
    if isolines_nodes:
        for node in isolines_nodes:
            center = data[data['Vertex'] == node].iloc[0]
            r, theta = center['Disc.Radius'], center['Inf.Theta']
            dibujar_isolineas(ax, r, theta, R=R, c=c, resolucion=3000, zorder=5)
    
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, zorder=0)  # Grid debajo
    
    if title:
        ax.set_title(title)
    
    return scatter

import numpy as np
from matplotlib.patches import Arc
from matplotlib.path import Path
import matplotlib.patches as patches

def get_hyperbolic_edge(x1, y1, x2, y2, num_points=100):
    """
    Calcula la geodésica hiperbólica entre dos puntos en el disco de Poincaré.
    Retorna las coordenadas x, y de la curva.
    """
    # Convertir a números complejos
    z1 = x1 + 1j*y1
    z2 = x2 + 1j*y2
    
    # Para puntos colineales con el centro o muy cercanos, usar línea recta
    if abs(x1*y2 - x2*y1) < 1e-10:  # Alineados con el centro
        return np.linspace(x1, x2, num_points), np.linspace(y1, y2, num_points)
    
    # Encontrar el centro del círculo ortogonal
    # La geodésica es un arco de círculo perpendicular al círculo unitario
    
    # Calcular el círculo que pasa por z1, z2 y es ortogonal al círculo unitario
    # El centro del círculo ortogonal está en la intersección de las mediatrices
    
    # Método: El centro del círculo de la geodésica satisface |c|^2 - R^2 = 1
    # y |c - z1| = |c - z2| = R
    
    # Resolver para el centro c = (cx, cy)
    # De la condición de equidistancia:
    # (cx - x1)^2 + (cy - y1)^2 = (cx - x2)^2 + (cy - y2)^2
    # Esto da: 2cx(x2-x1) + 2cy(y2-y1) = x2^2 - x1^2 + y2^2 - y1^2
    
    A = 2*(x2 - x1)
    B = 2*(y2 - y1)
    C = x2**2 - x1**2 + y2**2 - y1**2
    
    # Ortogonalidad: |c|^2 - R^2 = 1, y R^2 = |c - z1|^2
    # Esto implica: |c|^2 - |c - z1|^2 = 1
    # Expandido: 2cx*x1 + 2cy*y1 = x1^2 + y1^2 - 1
    
    D = 2*x1
    E = 2*y1
    F = x1**2 + y1**2 - 1
    
    # Resolver sistema lineal para cx, cy
    det = A*E - B*D
    if abs(det) < 1e-10:
        # Puntos diametralmente opuestos o degenerados
        return np.linspace(x1, x2, num_points), np.linspace(y1, y2, num_points)
    
    cx = (C*E - B*F) / det
    cy = (A*F - C*D) / det
    c = cx + 1j*cy
    
    # Radio del círculo
    R = abs(c - z1)
    
    # Ángulos de los puntos respecto al centro
    angle1 = np.arctan2(y1 - cy, x1 - cx)
    angle2 = np.arctan2(y2 - cy, x2 - cx)
    
    # Asegurar el arco más corto
    d_angle = angle2 - angle1
    if d_angle > np.pi:
        d_angle -= 2*np.pi
    elif d_angle < -np.pi:
        d_angle += 2*np.pi
    
    angles = np.linspace(angle1, angle1 + d_angle, num_points)
    x_points = cx + R * np.cos(angles)
    y_points = cy + R * np.sin(angles)
    
    return x_points, y_points

def mercator_disc_hyperbolic_net_ax(ax: plt.Axes, data: pd.DataFrame, mark_nodes: list[str] = [], 
                     net: nx.Graph = None, isolines_nodes: list[str] = None, 
                     R: float = None, c: float = None, title: str = None, 
                     linecolor='#00000045', hyperbolic_edges=True):
    
    kappa_vals = []
    positions = {v: (x, y) for _, (v, x, y) in data[['Vertex', 'Disc.X', 'Disc.Y']].iterrows()}
    
    kappa_vals = np.log10(data['Inf.Kappa'])
    
    if net:
        for a, b in net.edges():
            xa, ya = positions[a]
            xb, yb = positions[b]
            
            if hyperbolic_edges:
                # Dibujar geodésica hiperbólica
                x_curve, y_curve = get_hyperbolic_edge(xa, ya, xb, yb)
                ax.plot(x_curve, y_curve, linewidth=0.5, color=linecolor, 
                       solid_capstyle='round', zorder=1)
            else:
                # Línea recta euclidiana (original)
                ax.add_line(plt.Line2D([xa, xb], [ya, yb], 
                                      linewidth=0.5, color=linecolor))
    
    x_orig, y_orig = zip(*positions.values())
    
    scatter = ax.scatter(x_orig, y_orig, c=kappa_vals, cmap='viridis', zorder=10000,
                        s=15, alpha=0.5, edgecolors='black', linewidth=0.3)
    
    for mark_node in mark_nodes:
        mark_data = data[data['Vertex'] == mark_node].iloc[0]
        ax.plot(mark_data['Disc.X'], mark_data['Disc.Y'], 'r*', 
                markersize=15, markeredgecolor='black', zorder=100000)
    
    if isolines_nodes:
        for node in isolines_nodes:
            center = data[data['Vertex'] == node].iloc[0]
            r, theta = center['Disc.Radius'], center['Inf.Theta']
            # Asumiendo que dibujar_isolineas ya usa geodésicas hiperbólicas
            dibujar_isolineas(ax, r, theta, R=R, c=c, resolucion=3000)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    # Importante: mantener la relación de aspecto para geometría hiperbólica
    # ax.set_aspect('equal')
    # # Asegurar que el disco se ve completo
    # ax.set_xlim(-1.05, 1.05)
    # ax.set_ylim(-1.05, 1.05)
    
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