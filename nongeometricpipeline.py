# %% [markdown]
# ## Net generators

# %% [markdown]
# ## Tipus de xarxes
# 
# ### No geomètriques
# 
# - Erdos-Ranyi (no es scale-free, clustering baix)
# - Barabasi-Albert (scale-free, clustering baix)
# - Configuracional (clustering = 0)
# 
# ### Geomètriques
# 
# - $\mathbb{S}^1\mathbb{H}^2$
# 

# %% [markdown]
# ## Equivalència de paràmetres
# 
# | Característica                                  | ER                      | BA                                   | Configuracional                                   | S1 / H2                 |
# | ----------------------------------------------- | ----------------------- | ------------------------------------ | ------------------------------------------------- | ----------------------- |
# | $N$                                             | $N$                     | $N$                                  | $N$                                               | $N$                     |
# | Grau mig $\langle k \rangle$                    | $p=\frac{⟨k⟩}{N−1}$     | $m=\frac{⟨k⟩}{2}$​ (ha de ser enter) | En funció de la sequencia o distribució de graus. | $\langle k \rangle$     |
# | Exponent de la distribució $\gamma$ (si aplica) | -                       | Fixat, $\gamma = 3$                  | Escollir $P(k) = k^{-\gamma}$                     | $\gamma$                |
# | Clustering $C$                                  | Molt baix, no ajustable | Molt baix, no ajustable              | 0, no ajustable                                   | En funció de la $\beta$ |
# 

# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
import pipeline.data as data
import os
import pipeline.animation as anim
import subprocess
import multiprocessing as mp

# for seed in (12345, 12346, 12347, 12348, 12349):
for seed in (12346, 12347, 12348, 12349):
    
    g = 2.1
    n = 1000
    b = 2.1
    k = 20
    m = int(k/2)
    p = k/(n-1)
    
    working_folder = f'./generated-nets'
    s1h2_working_folder = f'{working_folder}/s1h2-s={seed}'
    er_working_folder = f'{working_folder}/er-s={seed}'
    ba_working_folder = f'{working_folder}/ba-s={seed}'
    conf_working_folder = f'{working_folder}/config-s={seed}'

    er_file = f'{er_working_folder}/er-n={n}-k={k}-s={seed}'
    ba_file = f'{ba_working_folder}/ba-n={n}-k={k}-s={seed}'
    conf_file = f'{conf_working_folder}/conf-n={n}-k={k}-g={g}-s={seed}'
    s1h2_file = f'{s1h2_working_folder}/s1h2-n={n}-k={k}-g={g}-b={b}-s={seed}'

    er_edges_file = f'{er_file}.edge'
    ba_edges_file = f'{ba_file}.edge'
    conf_edges_file = f'{conf_file}.edge'
    s1h2_edges_file = f'{s1h2_file}.edge'


    er_coords_file = f'{er_file}.inf_coord'
    ba_coords_file = f'{ba_file}.inf_coord'
    conf_coords_file = f'{conf_file}.inf_coord'
    s1h2_coords_file = f'{s1h2_file}.gen_coord'


    # %%
    import pipeline.data as data

    Gba, dfba, paramsba = data.read_hyperbolic_data(ba_coords_file, ba_edges_file)
    Ger, dfer, paramser = data.read_hyperbolic_data(er_coords_file, er_edges_file)
    Gconf, dfconf, paramsconf = data.read_hyperbolic_data(conf_coords_file, conf_edges_file)
    Gs1h2 , dfs1h2, paramss1h2 = data.read_hyperbolic_data(s1h2_coords_file, s1h2_edges_file, True)

    # %% [markdown]
    # # Resultats

    # %%
    for model in ('er', 's1h2', 'conf', 'ba'):
    # hecho hasta s1h2 12346 (incluido)
        match model:
            case 'er':
                df = dfer
                G = Ger
                params = paramser
                edges_file = er_edges_file
                working_folder = er_working_folder
            case 's1h2':
                df = dfs1h2
                G = Gs1h2
                params = paramss1h2
                edges_file = s1h2_edges_file
                working_folder = s1h2_working_folder
            case 'conf':
                df = dfconf
                G = Gconf
                params = paramsconf
                edges_file = conf_edges_file
                working_folder = conf_working_folder
            case 'ba':
                df = dfba
                G = Gba
                params = paramsba
                edges_file = ba_edges_file
                working_folder = ba_working_folder
        print(f'Calculando modelo {model} con seed = {seed}')
        # %% [markdown]
        # #### Velocitat máxima en funció del grau del node inicial

        # %%
        import pipeline.hyperbolic as hyp

        coords = (
            df.set_index('Vertex')
            [['Disc.Radius', 'Inf.Theta']]
            .to_dict('index')
        )

        hyp_distances = {}
        Gk = {}
        for u, v in G.edges():
            
            r1 = coords[u]['Disc.Radius']
            t1 = coords[u]['Inf.Theta']

            r2 = coords[v]['Disc.Radius']
            t2 = coords[v]['Inf.Theta']

            d = hyp.hyperbolic_distance_og_thetas(r1, t1, r2, t2)

            G[u][v]['dist'] = d

        # %%
        R = 2 * np.log(params['nb. vertices']/(params['mu']*np.pi*params['kappa_min']**2))


        edges = pd.DataFrame(G.edges, columns=['a', 'b'])
        edges = pd.merge(edges, df[['Vertex', 'Disc.Radius', 'Inf.Theta']], left_on='a', right_on='Vertex', suffixes=('_a', '_b'))
        edges = pd.merge(edges, df[['Vertex', 'Disc.Radius', 'Inf.Theta']], left_on='b', right_on='Vertex', suffixes=('_a', '_b'))
        edges['Theta_Dif'] = np.pi - np.abs(np.pi - np.abs(edges['Inf.Theta_a']-edges['Inf.Theta_b']))

        edges['Distance'] = np.where(edges['Theta_Dif'] == 0, 
                                    np.abs(edges['Disc.Radius_a']- edges['Disc.Radius_b']), 
                                    hyp.hyperbolic_distance_og(edges['Disc.Radius_a'], edges['Disc.Radius_b'], edges['Theta_Dif']))


        n=0
        c = -(n+1)/10
        edges['Epidemic_Func'] = hyp.link_probability_og(edges['Distance'], R, c)

        avg_epidemic_func = np.average(edges['Epidemic_Func'])

        edges['Weight_Multiplier'] = edges['Epidemic_Func']/avg_epidemic_func
        edges.to_csv(f"{edges_file}_weight_-{n+1}x10^-1", sep='\t', header=False, index=False, columns=['a', 'b', 'Weight_Multiplier'])

        # %%
        events_sn = {}
        targeted_epidemic_folder = f'{working_folder}/targeted-epidemics'
        import os
        os.makedirs(targeted_epidemic_folder, exist_ok=True)
        output_batch_file = f"{targeted_epidemic_folder}/batch_targeted_sir.txt"

        seed_base = 42075
        seeds = 15
        i_rate = 1
        r_rate = 1
        limit_time=1E10
        model_type=2 # sir
        c='-1x10^-1'
        weight_file = f'{edges_file}_weight_{c}'
        weighted=True
        n=0

        # %%
        with open(output_batch_file, "w") as f:
            f.write("# infection_rate  recovery_rate  seed  limit_time  model_type start_node\n")
            for v in df['Vertex']:
                for s in range(seeds):
                # Escribir línea con 5 valores (start_node opcional omitido)
                    f.write(f"{i_rate:.6f} {r_rate:.6f} {seed_base+s} {limit_time:.1f} {model_type} {v}\n")
                    n = n+1

        print(f"Archivo batch generado: {output_batch_file} con {n} simulaciones.")

        # %%
        print('Ejecutando simulaciones')
        subprocess.run(["./tools/epidemics","-b",output_batch_file,"-o",targeted_epidemic_folder,"-st","-ss",str(100),"-ev","-w",weight_file], 
                        stdout = subprocess.DEVNULL, check=True)
        print('Ejecutando simulaciones OK')


        # %%
        
        deg = G.degree()
        events_sn = {}
        stats_sn = {}
        seed_base = 42075
        seeds = 15
        for n, k in tqdm(deg, desc="Cargando resultados de las simulaciones"):
            for s in range(seeds):
                i_rate = 1
                r_rate = 1
                model='SIR'
                c='-1x10^-1'
                weight_file = f'{edges_file}_weight_{c}'
                weighted=True
                events_file = f'{targeted_epidemic_folder}/{data.build_events_filename(edges_file, weighted, model, i_rate, r_rate, seed_base+s, int(n))}'
                stats_file = f'{targeted_epidemic_folder}/{data.build_stats_filename(edges_file, weighted, model, i_rate, r_rate, seed_base+s, int(n))}'
                events_sn[(n, s)] = data.read_events_data(events_file)
                stats_sn[(n, s)] = data.read_stats_data(stats_file)

        # %%
        
        # ------------------------------------------------------------
        # 1. Preprocesamiento: posiciones de los nodos
        # ------------------------------------------------------------
        vertices = df['Vertex'].values
        num_nodes = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        radii = df['Disc.Radius'].values
        thetas = df['Inf.Theta'].values

        # ------------------------------------------------------------
        # 3. Precalcular matriz de distancias hiperbólicas (N x N)
        #    Solo si N es razonable (ej. < 5000). Si no, habrá que
        #    calcular distancias sobre la marcha pero evitando recalcular snapshots.
        # ------------------------------------------------------------
        print("Precalculando matriz de distancias hiperbólicas...")
        # Función vectorizada que calcula distancias entre dos conjuntos de puntos
        # (puedes adaptar tu hyp.hyperbolic_distance_og_thetas para que trabaje con broadcasting)
        def hyperbolic_distance_matrix(r1, theta1, r2, theta2):
            # Implementación típica de distancia hiperbólica en el disco de Poincaré
            # d = arcosh(cosh(r1)*cosh(r2) - sinh(r1)*sinh(r2)*cos(theta1 - theta2))
            # Devuelve matriz de tamaño (len(r1), len(r2))
            cosh_r1 = np.cosh(r1)
            cosh_r2 = np.cosh(r2)
            sinh_r1 = np.sinh(r1)
            sinh_r2 = np.sinh(r2)
            delta_theta = np.abs(theta1[:, None] - theta2[None, :])
            cos_delta = np.cos(delta_theta)
            arg = cosh_r1[:, None] * cosh_r2[None, :] - sinh_r1[:, None] * sinh_r2[None, :] * cos_delta
            # Evitar NaN por errores de redondeo
            arg = np.clip(arg, 1.0, None)   # fuerza valores menores a 1 a ser exactamente 1
            return np.arccosh(arg)  
        D = hyperbolic_distance_matrix(radii, thetas, radii, thetas)  # D[i,j] = distancia entre nodo i y nodo j


        # %%
        # 3. Tiempos
        xs = np.linspace(0, 2, 1000)

        # 4. Diccionario para almacenar resultados por nodo
        avg_hyps = {}
        err_hyps = {}
        avg_hyps_maxs = {}

        
        # Construir el mapeo rápido una sola vez (fuera de la función)
        # max_vertex = max(vertex_to_idx.values())
        # vertex_to_idx_array = np.full(max_vertex + 1, -1, dtype=int)
        # for v, i in vertex_to_idx.items():
        #     vertex_to_idx_array[v] = i

        def compute_node_optimized(args):
            n, _ = args
            idx = vertex_to_idx[n]
            row = D[idx]                      # fila fija de distancias
            valid_seeds = [s for s in range(seeds) if stats_sn[(n, s)]['rdens'].iloc[-1] >= 0.9]
            
            if not valid_seeds:
                # Si no hay semillas válidas, devolvemos arrays vacíos
                empty = [[]] * 7  # ajustaremos al número real de métricas
                return n, empty, empty, empty, empty, empty, empty, empty  # o tantas como métricas

            n_times = len(xs)
            # Acumuladores (7 métricas)
            sum_mean   = np.zeros(n_times)
            sum_median = np.zeros(n_times)
            sum_std    = np.zeros(n_times)
            sum_q1     = np.zeros(n_times)
            sum_q3     = np.zeros(n_times)
            sum_inf    = np.zeros(n_times)
            sum_rec    = np.zeros(n_times)
            count = 0

            all_vertices = set(vertices)

            for s in valid_seeds:
                events_df = events_sn.get((n, s))
                if events_df is None or events_df.empty:
                    # Sin eventos: infectados = 0, recuperados = 0 → distancias = 0
                    # No sumamos nada (contribución nula)
                    continue

                # Ordenar eventos (si ya lo están puedes omitir sort_values)
                events_df = events_df.sort_values('t')
                events = events_df[['t', 'vertex', 'event']].values

                S = all_vertices.copy()
                I = set()
                R = set()
                event_idx = 0
                n_events = len(events)

                for t_idx, t in enumerate(xs):
                    while event_idx < n_events and events[event_idx, 0] < t:
                        t_ev, v, ev_type = events[event_idx]
                        if ev_type == 'I':
                            S.discard(v)
                            R.discard(v)
                            I.add(v)
                        elif ev_type == 'R':
                            I.discard(v)
                            S.discard(v)
                            R.add(v)
                        event_idx += 1

                    # Métricas de distancia
                    if I:
                        # mapeo directo (más rápido, sin get ni máscaras)
                        infected_idxs = np.array([vertex_to_idx[v] for v in I], dtype=np.int64)
                        dists = row[infected_idxs]
                        # una sola llamada para los tres percentiles
                        q1, med, q3 = np.percentile(dists, [25, 50, 75])
                        mean_d = np.mean(dists)
                        std_d  = np.std(dists)
                    else:
                        mean_d = std_d = med = q1 = q3 = 0.0

                    sum_mean[t_idx]   += mean_d
                    sum_median[t_idx] += med
                    sum_std[t_idx]    += std_d
                    sum_q1[t_idx]     += q1
                    sum_q3[t_idx]     += q3
                    sum_inf[t_idx]    += len(I)
                    sum_rec[t_idx]    += len(R)

                count += 1

            if count > 0:
                return (n,
                        (sum_mean   / count).tolist(),
                        (sum_median / count).tolist(),
                        (sum_std    / count).tolist(),
                        (sum_q1     / count).tolist(),
                        (sum_q3     / count).tolist(),
                        (sum_inf    / count).tolist(),
                        (sum_rec    / count).tolist())
            else:
                # Sin semillas válidas: todo NaN
                nan_list = [np.nan] * n_times
                avg_mean_d = avg_median_d = avg_std_d = avg_q1_d = avg_q3_d = nan_list
                avg_infected = avg_recovered = nan_list

            return n, avg_mean_d, avg_median_d, avg_std_d, avg_q1_d, avg_q3_d, avg_infected, avg_recovered
        
        # 1. Cargar / definir todos los datos grandes (df, vertex_to_idx, D, xs, seeds, events_sn, anim)
        xs = np.linspace(0, 0.7, 1000)
        # ... (código de inicialización)

        # 2. Forzar uso de fork (por si acaso)
        mp.set_start_method('fork', force=True)
        # 3. Lanzar Pool
        with mp.Pool(processes=8) as pool:
            resultados = list(tqdm(pool.imap_unordered(compute_node_optimized, list(G.degree())),
                                    total=G.number_of_nodes(),
                                    desc="Procesando nodos"))

        # Acumular resultados del pool (desempaquetar todas las listas)
        avg_hyps = {}
        avg_medians = {}
        err_hyps = {}
        q1_hyps = {}
        q3_hyps = {}
        avg_infected = {}
        avg_recovered = {}

        for n, mean_d, median_d, std_d, q1_d, q3_d, inf, rec in resultados:
            avg_hyps[n] = mean_d
            avg_medians[n] = median_d
            err_hyps[n] = std_d
            q1_hyps[n] = q1_d
            q3_hyps[n] = q3_d
            avg_infected[n] = inf
            avg_recovered[n] = rec

        # %%
        results_folder = f'{targeted_epidemic_folder}/results'
        
        os.makedirs(results_folder, exist_ok=True)

        # Guardar cada nodo en su archivo .dat
        for v, k in tqdm(G.degree(), desc='Guardando post-procesado de distancias en archivos'):
            filename = f'{results_folder}/n={v}.dat'
            # Transponer para tener filas: t, media, mediana, std, Q1, Q3, infectados, recuperados
            node_data = np.column_stack([
                xs,
                avg_hyps[v],
                avg_medians[v],
                err_hyps[v],
                q1_hyps[v],
                q3_hyps[v],
                avg_infected[v],
                avg_recovered[v]
            ])
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# DATA\n")
                f.write(f"# sn={v}\n")
                f.write(f"# k={k}\n")
                f.write("#\n")
                f.write("# t, <d_hyp>, median_d, std_d, Q1_d, Q3_d, infected_count, recovered_count\n")
                for row in node_data:
                    # Escribir con formato flotante adecuado
                    f.write("\t".join(f"{val:.6g}" for val in row) + "\n")

        # %% [markdown]
        # ## Outbreak size final

        # %%
        
        deg = G.degree()
        outbreak_size_per_degree = {}
        seed_base = 42075
        seeds = 15
        targeted_epidemic_folder = f'{working_folder}/targeted-epidemics'

        for n, k in tqdm(deg, desc='Guardando post-procesado de outbreak size en archivos'):
            for s in range(seeds):
                i_rate = 1
                r_rate = 1
                model='SIR'
                c='-1x10^-1'
                weight_file = f'{edges_file}_weight_{c}'
                weighted=True
                file = f'{targeted_epidemic_folder}/{data.build_stats_filename(edges_file, weighted, model, i_rate, r_rate, seed_base+s, int(n))}'
                stats = pd.read_csv(file, names=['time', 'infected_density', 'recovered_density', 'actual_infection_rate', 'actual_recovery_rate'],
                                    sep=r'\s+', comment='#')
                if (k not in outbreak_size_per_degree):
                    outbreak_size_per_degree[k] = []
                if (stats['recovered_density'].values[-1] > 0.9):
                    outbreak_size_per_degree[k].append(stats['recovered_density'].values[-1])

        with open(f'{working_folder}/outbreak_per_degree.dat', 'w') as f:
            f.writelines([
                '# network parameters\n',
                f'# model: {model}\n',
                f'# n={n}\n',
                f'# beta={b}\n',
                f'# gamma={g}\n',
                f'# <k>={k}\n',
                f'# seed={s}\n',
                '# \n',
                '# k_star, avg_final_recovered_density, err_final_recovered_density\n',
            ])
            for k, v in outbreak_size_per_degree.items():
                f.write(f'{k}\t{np.mean(v)}\t{np.std(v)}\n')


