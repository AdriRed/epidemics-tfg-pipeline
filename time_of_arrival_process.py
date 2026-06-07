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
for seed in (12345, 12346, 12347, 12348, 12349):
    
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
    conf_working_folder = f'{working_folder}/conf-s={seed}'

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

        # # %%
        events_sn = {}
        targeted_epidemic_folder = f'{working_folder}/targeted-epidemics'
        # import os
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

        # # %%
        # with open(output_batch_file, "w") as f:
        #     f.write("# infection_rate  recovery_rate  seed  limit_time  model_type start_node\n")
        #     for v in df['Vertex']:
        #         for s in range(seeds):
        #         # Escribir línea con 5 valores (start_node opcional omitido)
        #             f.write(f"{i_rate:.6f} {r_rate:.6f} {seed_base+s} {limit_time:.1f} {model_type} {v}\n")
        #             n = n+1

        # print(f"Archivo batch generado: {output_batch_file} con {n} simulaciones.")

        # # %%
        # print('Ejecutando simulaciones')
        # subprocess.run(["./tools/epidemics","-b",output_batch_file,"-o",targeted_epidemic_folder,"-st","-ss",str(100),"-ev","-w",weight_file], 
        #                 stdout = subprocess.DEVNULL, check=True)
        # print('Ejecutando simulaciones OK')


        # %%
        
        deg = G.degree()
        events_sn = {}
        seed_base = 42075
        seeds = 15
        for n, k in tqdm(deg, desc=f"Cargando resultados de las simulaciones ({model}, s={seed})"):
            for s in range(seeds):
                i_rate = 1
                r_rate = 1
                model='SIR'
                c='-1x10^-1'
                weight_file = f'{edges_file}_weight_{c}'
                weighted=True
                events_file = f'{targeted_epidemic_folder}/{data.build_events_filename(edges_file, weighted, model, i_rate, r_rate, seed_base+s, int(n), True)}'
                events_sn[(n, s)] = data.read_events_data(events_file)

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
        # 1. Precalcular matriz de distancias hiperbólicas (N x N)
        # ------------------------------------------------------------
        print("Precalculando matriz de distancias hiperbólicas...")
        vertices = df['Vertex'].values
        num_nodes = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        radii = df['Disc.Radius'].values
        thetas = df['Inf.Theta'].values

        def hyperbolic_distance_matrix(r1, theta1, r2, theta2):
            cosh_r1 = np.cosh(r1)
            cosh_r2 = np.cosh(r2)
            sinh_r1 = np.sinh(r1)
            sinh_r2 = np.sinh(r2)
            delta_theta = np.abs(theta1[:, None] - theta2[None, :])
            cos_delta = np.cos(delta_theta)
            arg = cosh_r1[:, None] * cosh_r2[None, :] - sinh_r1[:, None] * sinh_r2[None, :] * cos_delta
            arg = np.clip(arg, 1.0, None)
            return np.arccosh(arg)

        D = hyperbolic_distance_matrix(radii, thetas, radii, thetas)

        # ------------------------------------------------------------
        # 2. Carpeta para guardar los archivos por nodo origen
        # ------------------------------------------------------------
        arrivals_folder = os.path.join(working_folder, "infection_arrivals_by_source")
        os.makedirs(arrivals_folder, exist_ok=True)

        # ------------------------------------------------------------
        # 3. Extraer eventos de infección (distancia, tiempo) para cada nodo origen
        # ------------------------------------------------------------
        print("Extrayendo eventos de infección...")
        # Recorrer todos los nodos como posibles orígenes
        for n, k in tqdm(G.degree(), desc=f"Procesando nodos origen ({model}, s={seed})"):
            idx_source = vertex_to_idx[n]
            # Archivo de salida para este nodo origen (modo append)
            out_file = os.path.join(arrivals_folder, f"source_{n}.dat")
            # Escribir cabecera si el archivo no existe (solo la primera vez)
            if not os.path.exists(out_file):
                with open(out_file, "w") as f:
                    f.write("# distance\ttime\n")
            # Recorrer todas las semillas de epidemia
            for s in range(seeds):
                events_df = events_sn.get((n, s))
                if events_df is None or events_df.empty:
                    continue
                events_df = events_df.sort_values('t')
                # Escribir cada evento de infección (excepto el origen en t=0)
                with open(out_file, "a") as f:
                    for t_inf, v_inf in events_df[events_df['event'] == 'I'][['t', 'vertex']].to_numpy():
                        if v_inf == n and t_inf == 0.0:
                            continue   # omitir la semilla inicial
                        idx_inf = vertex_to_idx[v_inf]
                        dist = D[idx_source, idx_inf]
                        f.write(f"{dist:.8f}\t{t_inf:.8f}\n")
        print("Extracción completada para este modelo y seed de red.")


