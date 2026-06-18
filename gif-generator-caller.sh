#!/bin/bash
# =============================================================================
# Script generador de frames de epidemia hiperbólica
# Cambia las variables según tu configuración y ejecuta:
#   bash generate_epidemic_gif.sh
# =============================================================================
export LC_NUMERIC=C

SCRIPT="gif-generator-events.py"

# ---------- Directorios y nombres base ----------
BASE_DIR="generated-nets-2"               # carpeta raíz de los datos
MODEL="s1h2"                            # tipo de modelo (s1h2, etc.)
SEED_MODEL=12345                        # semilla del grafo
N=1000                                  # número de nodos

# ---------- Parámetros del modelo (opcionales, déjalos vacíos si no existen) ----------
K=20                                    # grado medio (k)
G=2.1                                   # exponente de grado (gamma)
B=2.1                                   # exponente de clustering (beta)

# ---------- Tipo de coordenadas: "gen_coord" o "inf_coord" ----------
COORD_TYPE="gen_coord"

# ---------- Parámetros de la epidemia ----------
EPIDEMIC_DIR="epidemic_-1x10^-1"       # subcarpeta de eventos
WEIGHTED=true                           # epidemia sobre grafo pesado (true/false)
MODEL_TYPE="SIR"                        # modelo epidémico: SIR, SIS...
I_RATE=1.0                              # tasa de infección
R_RATE=1.0                              # tasa de recuperación
SEED_EPIDEMIC=42089                     # semilla de la epidemia
START_NODE=510                          # nodo inicial (sin ceros a la izquierda)

# ---------- Carpeta de salida de frames ----------
OUTPUT_DIR="./frames-events-sn=510"

# ---------- Parámetros de la animación ----------
T_START=0
T_END=3
STEP=0.005
PARALLEL="--parallel"                   # pon "" si no quieres paralelo
MAX_WORKERS=8                           # solo si usas --parallel
BOOST_NODE=

# ---------- Parámetros de salida de vídeo / GIF ----------
FRAMERATE=15
GIF_OUTPUT="${OUTPUT_DIR}/epidemia.gif"
VIDEO_OUTPUT="${OUTPUT_DIR}/epidemia.mp4"
PALETTE_FILE="${OUTPUT_DIR}/palette.png"







# ---------- Construcción automática de rutas ----------
MODEL_FOLDER="${MODEL}-s=${SEED_MODEL}"
BASE_NAME="${MODEL}-n=${N}"

# Parámetros del modelo dinámicos
MODEL_PARAMS=""
[ -n "$K" ] && MODEL_PARAMS="${MODEL_PARAMS}-k=${K}"
[ -n "$G" ] && MODEL_PARAMS="${MODEL_PARAMS}-g=${G}"
[ -n "$B" ] && MODEL_PARAMS="${MODEL_PARAMS}-b=${B}"

FULL_MODEL_NAME="${BASE_NAME}${MODEL_PARAMS}-s=${SEED_MODEL}"

# Archivo de coordenadas
COORDS_FILE="${BASE_DIR}/${MODEL_FOLDER}/${FULL_MODEL_NAME}.${COORD_TYPE}"
# Archivo de aristas (edge)
EDGES_FILE="${BASE_DIR}/${MODEL_FOLDER}/${FULL_MODEL_NAME}.edge"

# Archivo de eventos
# Formato: events-<nombre_modelo>-w<SI/NO><MODEL_TYPE>-I=<I> -R=<R> -S=<SEED_EPID> -SN=<START_NODE>.dat
WEIGHT_FLAG=""
$WEIGHTED && WEIGHT_FLAG="w" || WEIGHT_FLAG=""
# Ajustar formato de números: 1.00000 1.00000, semilla con 5 dígitos...
I_FMT=$(printf "%10.5f" $I_RATE)
R_FMT=$(printf "%10.5f" $R_RATE)
SEED_FMT=$(printf "%10d" $SEED_EPIDEMIC)
SN_FMT=$(printf "%05d" $START_NODE)

EVENTS_NAME="events-${FULL_MODEL_NAME}-${WEIGHT_FLAG}${MODEL_TYPE}-I=${I_FMT}-R=${R_FMT}-S=${SEED_FMT}-SN=${SN_FMT}.dat"
EVENTS_FILE="${BASE_DIR}/${MODEL_FOLDER}/${EPIDEMIC_DIR}/${EVENTS_NAME}"

# ---------- Ejecución ----------
echo "Coords: $COORDS_FILE"
echo "Edges:  $EDGES_FILE"
echo "Events: $EVENTS_FILE"
echo "Output: $OUTPUT_DIR"

BOOST_ARG=""
if [ -n "$BOOST_NODE" ]; then
    BOOST_ARG="--boost $BOOST_NODE"
fi

python "$SCRIPT" \
    "$COORDS_FILE" \
    "$EDGES_FILE" \
    "$EVENTS_FILE" \
    "$OUTPUT_DIR" \
    --t-start $T_START --t-end $T_END --step $STEP \
    $PARALLEL --max-workers $MAX_WORKERS \
    $BOOST_ARG
    


# Verificar si Python terminó correctamente
if [ $? -ne 0 ]; then
    echo "Error en la generación de frames."
    exit 1
fi

# ---------- Generación de GIF y vídeo lossless ----------
echo "Generando GIF (con paleta optimizada)..."
ffmpeg -framerate $FRAMERATE -i "${OUTPUT_DIR}/sim-%04d.png" \
       -vf "palettegen" "$PALETTE_FILE" -y

ffmpeg -framerate $FRAMERATE -i "${OUTPUT_DIR}/sim-%04d.png" -i "$PALETTE_FILE" \
       -filter_complex "paletteuse" "$GIF_OUTPUT" -y

echo "Generando vídeo lossless (H.264 qp=0)..."
ffmpeg -framerate $FRAMERATE -i "${OUTPUT_DIR}/sim-%04d.png" \
       -c:v libx264 -qp 0 -preset veryslow -pix_fmt yuv420p "$VIDEO_OUTPUT" -y

# Eliminar paleta temporal
rm -f "$PALETTE_FILE"

echo "Proceso completado:"
echo "  GIF:    $GIF_OUTPUT"
echo "  Vídeo:  $VIDEO_OUTPUT"