
def hyperbolic_boost(df, center_node):
    import numpy as np
    df_new = df.copy()
    # R = df['Disc.Radius']/max(df['Disc.Radius'])
    x0, x1, x2 = df['x0'], df['x1'], df['x2']
    disc_center = df[df['Vertex'] == center_node].iloc[0]
    phi, alpha, = np.acosh(disc_center['x0']), disc_center['Inf.Theta']

    x1_rot = x1 * np.cos(alpha) + x2 * np.sin(alpha)
    x2_rot = -x1 * np.sin(alpha) + x2 * np.cos(alpha)

    cosh_phi, sinh_phi = np.cosh(phi), np.sinh(phi)
    x0_new = cosh_phi * x0 + sinh_phi * x1_rot
    x1_new_rot = sinh_phi * x0 + cosh_phi * x1_rot
    x2_new_rot = x2_rot
    
    # Rotar de vuelta
    x1_new = x1_new_rot * np.cos(alpha) - x2_new_rot * np.sin(alpha)
    x2_new = x1_new_rot * np.sin(alpha) + x2_new_rot * np.cos(alpha)
    
    # Volver a disco de Poincaré
    R_new = np.sqrt(x1_new**2 + x2_new**2) / (1 + x0_new)
    theta_new = np.arctan2(x2_new, x1_new)
    df_new['x0'] = x0_new
    df_new['x1'] = x1_new
    df_new['x2'] = x2_new
    df_new['Disc.Radius'] = R_new
    df_new['Inf.Theta'] = theta_new
    df_new['Verifi'] = -df_new['x0']**2+df_new['x1']**2+df_new['x2']**2
    df_new['Disc.X'] = df_new['Disc.Radius']*np.cos(df_new['Inf.Theta'])
    df_new['Disc.Y'] = df_new['Disc.Radius']*np.sin(df_new['Inf.Theta'])
    df_new

    return df_new
        
def centrar_en_origen(r, theta, r_centro, theta_centro, zeta=1.0):
    """
    Aplica una isometría hiperbólica que mueve (r_centro, theta_centro) al origen.
    
    Parámetros
    ----------
    r, theta : ndarray
        Coordenadas polares nativas de los puntos a transformar.
    r_centro, theta_centro : float
        Punto que queremos llevar al origen.
    zeta : float
        Factor de curvatura (curvatura = -zeta²).
    
    Retorna
    -------
    r_nuevo, theta_nuevo : ndarray
        Coordenadas polares tras la transformación.
    """
    import numpy as np
    # --- 1. Punto de referencia en el hiperboloide ---
    rho0 = zeta * r_centro
    t0 = np.cosh(rho0)
    # Vector espacial unitario del punto de referencia
    nx = np.cos(theta_centro)
    ny = np.sin(theta_centro)
    # Las componentes espaciales completas serían sinh(rho0)*nx, sinh(rho0)*ny
    # pero para el boost solo necesitamos la dirección (nx, ny).

    # --- 2. Todos los puntos en el hiperboloide ---
    rho = zeta * r
    t = np.cosh(rho)
    x = np.sinh(rho) * np.cos(theta)
    y = np.sinh(rho) * np.sin(theta)

    # --- 3. Boost de Lorentz que lleva (t0, x0, y0) → (1, 0, 0) ---
    # producto escalar espacial entre P y cada punto
    dot = nx * x + ny * y
    ch0 = np.cosh(rho0)
    sh0 = np.sinh(rho0)

    t_prime = ch0 * t - sh0 * dot
    x_prime = -sh0 * nx * t + x + (ch0 - 1) * nx * dot
    y_prime = -sh0 * ny * t + y + (ch0 - 1) * ny * dot
    # --- 4. De vuelta a coordenadas polares nativas ---
    # t' = cosh(zeta * r_nuevo)
    t_prime_clip = np.clip(t_prime, 1.0, None)
    r_nuevo = np.arccosh(t_prime_clip) / zeta
    theta_nuevo = np.arctan2(y_prime, x_prime)
    return r_nuevo, theta_nuevo