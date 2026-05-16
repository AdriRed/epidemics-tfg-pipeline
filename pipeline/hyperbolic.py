
def kappa_to_hyperbolic(kappa, kappa_min): # ln k/k_0
    """
    Convierte κ a coordenada radial hiperbólica
    r = ln(κ/κ_min)
    """
    import numpy as np
    return np.log(kappa / kappa_min)

def hyperbolic_to_mercator(r_hiperbolico, edge_count, mu, kappa_min):
    """
    Convierte radio hiperbólico a coordenada en disco de Poincaré
    r_poincare = R*-2*r_hyp
    """
    import numpy as np
    R = 2 * np.log(edge_count/(mu*np.pi*kappa_min**2))
    return R - 2* r_hiperbolico

def mercator_to_poincare(r_mercator, R):
    """
    Convierte coordenada radial en proyección Mercator a radio en disco de Poincaré.
    """
    import numpy as np
    r_hip = (R - r_mercator) / 2.0
    # r_hip puede ser negativo si r_mercator > R; pero en teoría nunca ocurre.
    # Aseguramos que r_hip >= 0 (clipping)
    r_hip = max(r_hip, 0.0)
    return np.tanh(r_hip / 2.0)

def poincare_to_mercator(r_poincare, R):
    """
    Convierte radio en disco de Poincaré (0 a 1) a coordenada Mercator.
    """
    import numpy as np
    if r_poincare >= 1.0:
        r_poincare = 1.0 - 1e-12
    r_hip = 2.0 * np.arctanh(r_poincare)
    r_mercator = R - 2.0 * r_hip
    return r_mercator


def hyperbolic_distance_og_thetas(r_a, theta_a, r_b, theta_b, zeta=1.0):   
    import numpy as np
    theta_diff = np.abs(np.deg2rad(theta_a - theta_b))
    return hyperbolic_distance_og(r_a, r_b, np.pi - np.abs(np.pi - theta_diff))


def hyperbolic_distance_og(r_a, r_b, theta_diff, zeta=1.0):   
    import numpy as np
    cosh_val = np.cosh(zeta*r_a) * np.cosh(zeta*r_b) - \
               np.sinh(zeta*r_a) * np.sinh(zeta*r_b) * np.cos(theta_diff)
    cosh_val = np.clip(cosh_val, 1.0, None)
    return np.arccosh(cosh_val) / zeta

def link_probability_og(distance, R, c):
    import numpy as np
    """Probabilidad de enlace entre 0 y 1 (estable numéricamente)"""
    z = c * (distance - R)
    # Evitar overflow
    z = np.clip(z, -700, 700)  # exp(700) es ~10^304, cerca del límite
    return (np.exp(z))
