
def process_rate(rate, working_folder, weight=None, limit=None):
    """Procesa todas las repeticiones para una tasa específica"""
    import numpy as np
    import pandas as pd
    integrals = []
    outbreak_size = []
    
    for j in range(1, 61):
        try:
            filename = f"{working_folder}/weight_-{weight}x10^-1/stats-n10000-g=2.2-b=3_GC-wSIR-I=   {rate:.5f}-R=   0.10000-S={42069+j}.dat"

            df = pd.read_csv(filename, sep=r'\s+', header=None, 
                            names=['t', 'idens', 'rdens', 'irate', 'rrate'], 
                            engine='python', comment='#')
            df['sdens'] = 1 - df['rdens'] - df['idens']
            if limit is not None:
                if (df['idens'].max() > limit):
                    integrals.append(np.trapezoid(df['idens'], df['t']))
                if (df['rdens'].iloc[-1] > limit):
                    outbreak_size.append(df['rdens'].iloc[-1])
        except:
            print(filename)
    
    if (len(integrals) == 0):
        integrals = [0]
    if (len(outbreak_size) == 0):
        outbreak_size = [0]
    
    return {
        "rate": rate*10,
        "integral": np.mean(integrals),
        # "err_integral": np.std(integrals),
        "outbreak_size": np.mean(outbreak_size),
        "err_outbreak_size": np.mean(outbreak_size)
    }