def configure_matplotlib():
    import matplotlib.pyplot as plt
    # Configuración estilo revtex4
    plt.style.use('default')  # Empezar desde cero

    rcparams = {
        # Fuentes
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 10,
        'font.weight': 'normal',
        
        # Tamaño de figura (ancho de columna típico de revtex4: ~3.5 pulgadas)
        'figure.figsize': (3.5, 2.8),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        
        # Ejes
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        
        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        
        # Líneas
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Leyenda
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.loc': 'best',
        
        # Guardado
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'pgf.texsystem': 'xelatex'
    }

    plt.rcParams.update(rcparams)


def ajuste_polinomial_con_estadisticas_v2(x, y, grado):
    """
    Versión consistente usando np.polynomial.polynomial (orden ascendente)
    """
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Coeficientes en orden ASCENDENTE: [a0, a1, a2, ...]
    coefs = np.polynomial.polynomial.polyfit(x, y, grado)
    
    # Predicciones - polyval espera orden ascendente
    y_pred = np.polynomial.polynomial.polyval(x, coefs)
    
    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Matriz de diseño en orden ASCENDENTE
    A = np.vander(x, grado + 1, increasing=True)
    
    n = len(x)
    p = grado + 1
    var_res = ss_res / (n - p)
    
    cov_mat = var_res * np.linalg.pinv(A.T @ A)  # Usar pinv por estabilidad
    errores = np.sqrt(np.diag(cov_mat))
    
    return coefs, errores, r2, cov_mat
