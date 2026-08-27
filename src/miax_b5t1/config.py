"""Parametros globales del proyecto.

Centralizar la configuracion evita que cada notebook use valores distintos y
garantiza que todas las versiones del clasificador se entrenen y evaluen en
condiciones identicas, tal y como exige el enunciado del taller.
"""

import os

# --- Rutas -----------------------------------------------------------------

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIR_DATOS_RAW = os.path.join(RAIZ, "data", "raw")
DIR_DATOS_PROC = os.path.join(RAIZ, "data", "processed")
DIR_MODELOS = os.path.join(RAIZ, "models")
DIR_FIGURAS = os.path.join(RAIZ, "results", "figuras")
DIR_TABLAS = os.path.join(RAIZ, "results", "tablas")

RUTA_PRECIOS_RAW = os.path.join(DIR_DATOS_RAW, "precios_close_sp500.csv")

# Cache reducido con el universo ya filtrado. Ocupa pocos megabytes, se versiona
# junto al codigo y permite ejecutar el proyecto completo tras clonar el
# repositorio, sin depender de la descarga ni del servicio externo.
RUTA_PRECIOS_UNIVERSO = os.path.join(DIR_DATOS_PROC, "precios_universo.csv")
RUTA_DATASET = os.path.join(DIR_DATOS_PROC, "dataset.npz")

# --- Construccion del dataset ----------------------------------------------

# Fecha de inicio de la descarga. Cuanto mas atras, menos tickers sobreviven al
# filtrado por historia completa, pero mas ciclos de mercado quedan cubiertos.
FECHA_INICIO = "1945-01-01"

# Numero de activos del universo final. El valor por defecto reproduce el
# tamano del universo empleado en el material del taller. Se puede reducir para
# abaratar los entrenamientos sin tocar el resto del codigo.
N_TICKERS = 23

# Dias de historia que observa el modelo.
VENTANA_X = 60

# Horizonte futuro sobre el que se mide el drawdown.
VENTANA_Y = 30

# Proporcion objetivo de la clase positiva. El umbral de drawdown que la produce
# se calibra sobre el bloque de entrenamiento y se aplica sin cambios al resto.
PROPORCION_POSITIVOS = 0.10

# --- Particion temporal ----------------------------------------------------

FRAC_TRAIN = 0.70
FRAC_VAL = 0.15

# Dias descartados entre bloques. Una ventana usa 60 dias de pasado y 30 de
# futuro, de modo que un hueco de 90 dias impide que una misma observacion de
# mercado aparezca a ambos lados de la frontera.
EMBARGO = VENTANA_X + VENTANA_Y

# --- Experimento -----------------------------------------------------------

# Presupuesto de muestras reales con el que se entrenan tanto los generadores
# como el clasificador. Limitar los reales es lo que crea la situacion de
# escasez que el dato sintetico pretende compensar.
N_REALES = 3000

# Multiplicadores de datos sinteticos sobre el presupuesto real.
RATIOS_SINTETICOS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

# Repeticiones con semilla distinta de cada configuracion.
N_SEMILLAS = 5

SEMILLA = 42

# --- Entrenamiento del clasificador ----------------------------------------

EPOCHS_CLASIFICADOR = 80
BATCH_CLASIFICADOR = 64
PACIENCIA = 12

# --- Estetica --------------------------------------------------------------

ESTILO_MPL = "ggplot"

COLORES_MODELO = {
    "baseline": "#4c4c4c",
    "ruido": "#8c8c8c",
    "cgan": "#c0392b",
    "cvae": "#2874a6",
    "diffusion": "#1e8449",
}

NOMBRES_MODELO = {
    "baseline": "Solo reales",
    "ruido": "Ruido gaussiano",
    "cgan": "cGAN",
    "cvae": "CVAE",
    "diffusion": "Diffusion",
}


def asegurar_directorios():
    """Crea la estructura de carpetas de salida si aun no existe."""
    for ruta in (DIR_DATOS_RAW, DIR_DATOS_PROC, DIR_MODELOS, DIR_FIGURAS, DIR_TABLAS):
        os.makedirs(ruta, exist_ok=True)
