# Epidemics TFG Pipeline

Pipeline to simulate and analyze epidemic spreading in different types of networks, comparing the behavior in geometric (hyperbolic) networks versus non-geometric networks.

**Author:** Adrià Rojo  
**Year:** 2026

## 🎯 What are we trying to find?

### Main Analysis

1. **Hyperbolic distance vs propagation**: How hyperbolic distance between nodes influences the speed and propagation pattern of the epidemic.

2. **Outbreak size by node degree**: Relationship between the degree of the initial node and the final size of the epidemic outbreak.

3. **2D normalized histograms**: Distribution of hyperbolic distances vs infection arrival time, to characterize propagation patterns in each network model.

4. **Model comparison**: Comparative analysis of propagation in:
   - Non-geometric networks: Erdős-Rényi (ER), Barabási-Albert (BA), Configuration model
   - Geometric networks: S¹/H² (hyperbolic space)

## 🛠️ What technologies are used?

### Python Environment

```bash
conda create -n epidemics-tfg python=3.13 \
  matplotlib ipykernel numpy pandas scipy sympy networkx tqdm \
  scikit-learn pillow jupyter

pip install uncertainties
```

**Main Python dependencies:**
- **NumPy**: Numerical computations and matrices
- **Pandas**: Data manipulation (DataFrames)
- **Matplotlib**: Visualization and plotting
- **NetworkX**: Network/graph manipulation
- **SciPy**: Scientific functions (2D histograms, statistics)
- **SymPy**: Symbolic computations (propagation equations)
- **tqdm**: Progress bars
- **uncertainties**: Error propagation
- **scikit-learn**: Data analysis (optional)

### Compiled Tools (Git submodules)

The project depends on **three Git submodules** with C++/Fortran tools that must be compiled:

#### 1. **[SD-model](https://github.com/networkgeometry/SD-model)** (Hyperbolic network generator)
   - **Language**: C++17 with Boost
   - **Dependencies**: `libboost-system`, `libboost-math`
   - **Compilation**: 
     ```bash
     g++ -O3 -std=c++17 -lboost_system -lboost_math_c99 \
         SD-model/src/generatingSD_unix.cpp -o tools/genSD
     ```
   - **Purpose**: Generates networks with power-law degree distribution in hyperbolic space
   - **Parameters**: N (nodes), γ (exponent), β (clustering), k (average degree), seed

#### 2. **[Mercator](https://github.com/networkgeometry/mercator)** (Hyperbolic embedding)
   - **Language**: C++11 with Eigen
   - **Compilation**:
     ```bash
     g++ -O3 -std=c++11 -fpermissive -I./mercator/include/ \
         ./mercator/src/embeddingS1_unix.cpp -o ./tools/mercator
     ```
   - **Purpose**: Maps networks to hyperbolic coordinates (radius, angle)
   - **Input**: `.edge` file (edge list)
   - **Output**: `.inf_coord` file (hyperbolic coordinates)

#### 3. **[Epidemics Simulator](https://github.com/AdriRed/epidemics-tfg)** (Epidemic spreading simulator)
   - **Language**: Fortran 90 with OpenMP
   - **Compilation**:
     ```bash
     gfortran -O3 -march=native -funroll-loops -fopenmp \
         ./epidemics-tfg/include/{mt19937.f90,mt19937_par.f90,fhash.f90,net_loader.f90,reversed_skiplist.f90,epidemic.f90} \
         ./epidemics-tfg/main2.f90 \
         -o ./tools/epidemics
     ```
   - **Purpose**: Simulates epidemic propagation (SIR/SIS models) on networks
   - **Main parameters**:
     - `-i`: Infection rate
     - `-r`: Recovery rate
     - `-m`: Model (SIR or SIS)
     - `-w`: Weighted network
     - `-b`: Batch file with simulations
     - `-st`: Save statistics
     - `-ev`: Save events

## 📁 Project Structure

```
.
├── README.md                          # This file
├── pipeline.ipynb                     # Main pipeline (geometric networks)
├── non-geometric-pipeline.ipynb       # Pipeline (non-geometric networks)
├── outbreak-size-pipeline.ipynb       # Outbreak size analysis
├── histogram_pipeline.ipynb           # 2D histogram generation
├── net-generation.ipynb               # Network generation and analysis
├── calc.ipynb                         # Symbolic calculations with SymPy
│
├── pipeline/                          # Main Python module
│   ├── __init__.py
│   ├── data.py                        # Read coordinates and statistics
│   ├── hyperbolic.py                  # Hyperbolic distance calculations
│   ├── figures.py                     # Figure generation
│   ├── boost.py                       # Boost integration
│   ├── animation.py                   # Animations (optional)
│
├── tools/                             # Compiled executables
│   ├── genSD                          # SD-model generator
│   ├── mercator                       # Hyperbolic embedding
│   └── epidemics                      # Epidemic simulator
│
├── SD-model/                          # [Git submodule]
├── mercator/                          # [Git submodule]
├── epidemics-tfg/                     # [Git submodule]
│
├── pipeline-output/                   # Output files (generated)
└── generated-nets/                    # Generated networks (generated)
```

## 🚀 Installation and Setup

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/AdriRed/epidemics-tfg-pipeline.git
cd epidemics-tfg-pipeline
```

### 2. Create conda environment

```bash
conda create -n epidemics-tfg python=3.13 \
  matplotlib ipykernel numpy pandas scipy sympy networkx tqdm \
  scikit-learn pillow jupyter

conda activate epidemics-tfg

pip install uncertainties
```

### 3. Compile the tools

```bash
mkdir -p tools

# SD-model
g++ -O3 -std=c++17 -lboost_system -lboost_math_c99 \
    SD-model/src/generatingSD_unix.cpp -o tools/genSD

# Mercator
g++ -O3 -std=c++11 -fpermissive -I./mercator/include/ \
    ./mercator/src/embeddingS1_unix.cpp -o ./tools/mercator

# Epidemics
gfortran -O3 -march=native -funroll-loops -fopenmp \
    ./epidemics-tfg/include/mt19937.f90 \
    ./epidemics-tfg/include/mt19937_par.f90 \
    ./epidemics-tfg/include/fhash.f90 \
    ./epidemics-tfg/include/net_loader.f90 \
    ./epidemics-tfg/include/reversed_skiplist.f90 \
    ./epidemics-tfg/include/epidemic.f90 \
    ./epidemics-tfg/main2.f90 \
    -o ./tools/epidemics
```

## 📊 Workflow

### Option A: Geometric Networks (S¹/H²)

1. **`pipeline.ipynb`**: Generates hyperbolic networks and simulates epidemics
   - Parameters: N, γ, β, ⟨k⟩, seed
   - Output: Statistics and infection events

2. **`histogram_pipeline.ipynb`**: Processes distance-time histograms
   - Input: Event files
   - Output: Normalized 2D histograms (`.npy`)

### Option B: Non-Geometric Networks (ER, BA, Configuration)

1. **`non-geometric-pipeline.ipynb`**: Generates non-geometric networks and simulates epidemics
   - Implements parameter equivalence
   - Extracts events by source node

2. **`time_of_arrival_process.py`**: Arrival time analysis
   - Calculates induced hyperbolic distances
   - Groups events by distance and time

### Cross-sectional Analysis

- **`outbreak-size-pipeline.ipynb`**: Outbreak size vs node degree
- **`statistics-nets.ipynb`**: Comparative statistics across models

## 📝 Input/Output Files

### Input (Network)

- `.edge`: Edge list file (text)
  ```
  node1 node2
  node3 node4
  ...
  ```

### Output (Simulation)

- `.inf_coord`: Hyperbolic coordinates (from Mercator)
  ```
  Vertex  Inf.Kappa  Inf.Theta  Inf.Hyp.Rad.  ...
  0       26.276     3.032      15.218        ...
  ```

- `events-*.dat`: Infection/recovery events
  ```
  # t vertex event
  0.001 5 I
  0.042 7 I
  ...
  ```

- `stats-*.dat`: Statistics per timestep
  ```
  # time infected_density recovered_density ...
  0.0   0.001 0.0 ...
  0.1   0.045 0.001 ...
  ```

- `outbreak_per_degree.dat`: Final outbreak size by degree
  ```
  # k_star avg_final_recovered_density err
  10 0.456 0.032
  20 0.523 0.041
  ```

## 🔧 System Requirements

### Compilers

- **GCC/G++**: >= 9 (for C++17)
- **Gfortran**: >= 9
- **Boost**: >= 1.70 (headers)

### Example installation on Debian/Ubuntu

```bash
sudo apt-get install g++ gfortran libboost-all-dev
```

