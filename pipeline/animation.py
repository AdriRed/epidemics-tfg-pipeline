import pandas as pd
def mercator_disc_epidemic_anim(
    df,
    events,
    epidemics_fig_output,
    step=0.1,
    t_start=None,
    t_end=None,
):
    """
    Simula una epidemia en el disco hiperbólico y genera frames para un GIF/animación.

    Parámetros
    ----------
    df : DataFrame
        Datos de los nodos con columnas 'Vertex', 'Disc.X', 'Disc.Y'.
    events : DataFrame
        Eventos con columnas 't', 'vertex', 'event' (I o R).
    epidemics_fig_output : str
        Carpeta donde se guardarán las imágenes.
    step : float
        Incremento de tiempo entre frames.
    t_start : float, opcional
        Tiempo inicial de la animación. Si es None, se usa el min(events['t']).
    t_end : float, opcional
        Tiempo final de la animación. Si es None, se usa el max(events['t']).
    parallel : bool
        Si es True, renderiza las imágenes en paralelo. En Jupyter puede dar problemas,
        desactívalo si no puedes interrumpir el proceso.
    max_workers : int, opcional
        Número de procesos paralelos (solo si parallel=True). Por defecto usa los disponibles.
    """
    import os
    import shutil
    import math
    from tqdm import tqdm
    from . import figures as figs
    # ---------- Limpieza y preparación del directorio ----------
    if os.path.exists(epidemics_fig_output):
        shutil.rmtree(epidemics_fig_output)
    os.makedirs(epidemics_fig_output)

    # ---------- Coordenadas de los vértices ----------

    # ---------- Conjuntos de estado (como coordenadas) ----------
    infected = set()
    recovered = set()

    # ---------- Ventana temporal ----------
    if t_start is None:
        t_start = events["t"].min()
    if t_end is None:
        t_end = events["t"].max()

    print(f"Ventana temporal: {t_start:.2f} → {t_end:.2f}")

    # Número de pasos
    n_steps = math.ceil((t_end - t_start) / step)
    print(f"Total de frames: {n_steps}")

    # ---------- Eventos ordenados ----------
    events_sorted = events.sort_values("t").reset_index(drop=True)
    event_idx = 0
    n_events = len(events_sorted)

    # ---------- Fase 1: Simulación secuencial (rápida) ----------
    print("Simulando evolución de estados...")
    snapshots = []  # cada elemento: (t, list(susc), list(inf), list(rec))
    snap = None
    for i in tqdm(range(n_steps), desc="Simulación"):
        t = t_start + (i + 1) * step

        # Procesar eventos hasta t
        snap = get_snapshot(df, events_sorted, t, snap)

        mask_sus = df['Vertex'].isin(snap['susceptibles'])
        mask_inf = df['Vertex'].isin(snap['infected'])
        mask_rec = df['Vertex'].isin(snap['recovered'])

        coords_sus = zip(*df[mask_sus][['Disc.X', 'Disc.Y']])
        coords_inf = zip(*df[mask_inf][['Disc.X', 'Disc.Y']])
        coords_rec = zip(*df[mask_rec][['Disc.X', 'Disc.Y']])

        snapshots.append((snap['time'], coords_sus, coords_inf, coords_rec))

    # ---------- Fase 2: Renderizado ----------

    for i, s in enumerate(tqdm(snapshots)):
        t, susc_list, inf_list, rec_list = s
    # La función mercator_epidemic_disc probablemente espera conjuntos
        figs.mercator_epidemic_disc(
            df,
            set(susc_list),
            set(inf_list),
            set(rec_list),
            f"{epidemics_fig_output}/sim-{i:04d}.png",
            t,
        )

def get_snapshot(df: pd.DataFrame, events_sorted: pd.DataFrame, t: float, last_state: dict = None ):
    susceptible = last_state['susceptible'] if last_state else set(df["Vertex"].to_list())
    event_idx = 0
    infected, recovered = set(), set()
    if (last_state):
        infected = last_state['infected']
        recovered = last_state['recovered']

    scope = events_sorted[events_sorted['t'] < t]
    if (last_state):
        scope = scope[scope['t'] >= last_state['time']]
    for i, ev in scope.iterrows():
        v = ev["vertex"]

        if ev["event"] == "I":
            susceptible.discard(v)
            recovered.discard(v)
            infected.add(v)
        elif ev["event"] == "R":
            infected.discard(v)
            susceptible.discard(v)
            recovered.add(v)


        # Guardar estado como listas (serializable)
    return {
        'time': t,
        'susceptible': susceptible,
        'recovered': recovered,
        'infected': infected
    }


def generate_gif(df, epidemic_folder, events, boosted, start_node=None):
    from . import hyperbolic as hyp
    import os
    df_new = df.copy()
    fig_output = f'{epidemic_folder}/disc'
    if boosted:
        fig_output += 'boosted'
        df_new = hyp.hyperbolic_boost(df, start_node)

    os.makedirs(fig_output, exist_ok=True)
    mercator_disc_epidemic_anim(
        df_new,
        events,
        fig_output,
        step=0.001,
        t_start=min(events[events['t'] > 0]['t']),   
        t_end=min(events[events['t'] > 0]['t'])+3,     
    )
