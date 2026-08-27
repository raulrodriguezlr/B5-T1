"""Construccion del dataset supervisado a partir de los precios historicos.

El flujo completo es: precios de cierre ajustados -> rentabilidades logaritmicas
-> ventanas deslizantes de 60 dias (X) -> etiqueta binaria de drawdown severo a
30 dias (y) -> particion cronologica con embargo -> escalado.
"""

import os

import numpy as np
import pandas as pd

from . import config


# --- Carga y limpieza ------------------------------------------------------


def cargar_precios(ruta=None, n_tickers=None):
    """Lee el cache de precios y devuelve las series con historia completa.

    Se descartan las columnas con cualquier hueco, de modo que el universo
    resultante esta libre de valores ausentes y no requiere imputacion. El
    criterio introduce sesgo de supervivencia, circunstancia que se documenta
    en el analisis exploratorio.

    Se prefiere el cache reducido, que contiene el universo ya filtrado y se
    distribuye con el repositorio. Si no existe se recurre a la descarga
    completa, mucho mas pesada.
    """
    if ruta is None:
        if os.path.exists(config.RUTA_PRECIOS_UNIVERSO):
            ruta = config.RUTA_PRECIOS_UNIVERSO
        else:
            ruta = config.RUTA_PRECIOS_RAW

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encuentra ningun cache de precios. "
            "Ejecutar antes 'python scripts/descargar_precios.py'."
        )

    precios = pd.read_csv(ruta, index_col=0, parse_dates=True)
    precios = precios.dropna(axis=1)

    n_tickers = n_tickers or config.N_TICKERS
    if precios.shape[1] > n_tickers:
        # Ante un universo mayor que el solicitado se conservan los activos de
        # mayor recorrido temporal efectivo, manteniendo el orden alfabetico
        # para que la seleccion sea reproducible.
        precios = precios.iloc[:, :n_tickers]

    return precios


def calcular_rentabilidades(precios):
    """Convierte precios en rentabilidades logaritmicas diarias."""
    return np.log(precios).diff().dropna()


def limpiar_rentabilidades(returns, n_sigmas=12.0):
    """Neutraliza registros imposibles procedentes de errores de cotizacion.

    Un salto superior a `n_sigmas` desviaciones tipicas en un unico dia rara vez
    corresponde a un movimiento real de mercado y suele deberse a splits o
    dividendos mal ajustados. Esos valores se sustituyen por el limite, lo que
    preserva el signo del movimiento sin permitir que un unico registro domine
    la escala de la serie.
    """
    limites = n_sigmas * returns.std()
    n_recortes = int((returns.abs() > limites).sum().sum())
    returns_limpio = returns.clip(lower=-limites, upper=limites, axis=1)
    return returns_limpio, n_recortes


# --- Construccion de X e y -------------------------------------------------


def serie_mercado(returns):
    """Rentabilidad diaria de la cartera equiponderada del universo."""
    return returns.mean(axis=1)


def drawdown_futuro(ret_mercado, ventana_y=None):
    """Drawdown maximo de la cartera en los `ventana_y` dias siguientes.

    Para cada fecha se reconstruye la trayectoria acumulada del horizonte futuro
    y se mide la caida maxima desde su propio maximo movil. El resultado es
    negativo o cero y expresa la peor perdida encadenada del periodo.
    """
    ventana_y = ventana_y or config.VENTANA_Y
    valores = ret_mercado.values
    n = len(valores)
    dd = np.full(n, np.nan)

    for i in range(n - ventana_y):
        tramo = np.exp(np.cumsum(valores[i : i + ventana_y]))
        maximo_movil = np.maximum.accumulate(tramo)
        dd[i] = float((tramo / maximo_movil - 1.0).min())

    return pd.Series(dd, index=ret_mercado.index, name="drawdown_futuro")


def construir_ventanas(returns, dd_futuro, ventana_x=None, ventana_y=None):
    """Genera las ventanas de entrada y el drawdown asociado a cada una.

    La ventana i-esima recoge los `ventana_x` dias anteriores a la fecha de
    corte, y su etiqueta describe lo que ocurre en los `ventana_y` dias
    posteriores. Ninguna observacion futura entra en la entrada del modelo.
    """
    ventana_x = ventana_x or config.VENTANA_X
    ventana_y = ventana_y or config.VENTANA_Y

    matriz = returns.values
    lista_x, lista_dd, fechas = [], [], []

    for i in range(ventana_x, len(returns) - ventana_y):
        lista_x.append(matriz[i - ventana_x : i])
        lista_dd.append(dd_futuro.iloc[i])
        fechas.append(returns.index[i])

    X = np.array(lista_x, dtype="float32")
    dd = np.array(lista_dd, dtype="float32")
    return X, dd, pd.DatetimeIndex(fechas)


def calibrar_umbral(dd_train, proporcion=None):
    """Determina el umbral de drawdown que produce la proporcion deseada.

    El umbral se fija exclusivamente con el bloque de entrenamiento. Calibrarlo
    sobre la muestra completa filtraria informacion del periodo de test hacia la
    definicion del problema.
    """
    proporcion = proporcion or config.PROPORCION_POSITIVOS
    return float(np.quantile(dd_train, proporcion))


def etiquetar(dd, umbral):
    """Marca como positivas las ventanas cuyo drawdown futuro cae bajo el umbral."""
    return (dd <= umbral).astype("float32")


# --- Particion cronologica -------------------------------------------------


def particion_temporal(n, frac_train=None, frac_val=None, embargo=None):
    """Devuelve los indices de train, validacion y test por bloques temporales.

    Entre bloques se descarta un tramo de longitud `embargo` para que ninguna
    ventana de un conjunto comparta observaciones de mercado con otro. Sin ese
    hueco, el solapamiento de ventanas contiguas transferiria informacion del
    test al entrenamiento y las metricas resultarian optimistas.
    """
    frac_train = frac_train if frac_train is not None else config.FRAC_TRAIN
    frac_val = frac_val if frac_val is not None else config.FRAC_VAL
    embargo = embargo if embargo is not None else config.EMBARGO

    corte_train = int(n * frac_train)
    corte_val = int(n * (frac_train + frac_val))

    idx_train = np.arange(0, corte_train)
    idx_val = np.arange(corte_train + embargo, corte_val)
    idx_test = np.arange(corte_val + embargo, n)

    return idx_train, idx_val, idx_test


# --- Escalado --------------------------------------------------------------


class EscaladorTanh:
    """Estandariza por activo y comprime el resultado al intervalo [-1, 1].

    Los generadores empleados en el taller cierran con activacion `tanh`, cuyo
    recorrido es exactamente ese intervalo. Una estandarizacion simple dejaria
    parte de la masa de probabilidad fuera del alcance del generador, mientras
    que un escalado por minimo y maximo concentraria casi toda la distribucion
    en torno a cero por efecto de las colas gruesas caracteristicas de las
    series financieras. El recorte a `n_sigmas` desviaciones tipicas resuelve
    ambos problemas a costa de saturar una fraccion muy pequena de valores.
    """

    def __init__(self, n_sigmas=4.0):
        self.n_sigmas = n_sigmas
        self.media_ = None
        self.escala_ = None

    def ajustar(self, X):
        self.media_ = X.mean(axis=(0, 1), keepdims=True)
        self.escala_ = X.std(axis=(0, 1), keepdims=True) * self.n_sigmas
        self.escala_ = np.where(self.escala_ == 0, 1.0, self.escala_)
        return self

    def transformar(self, X):
        return np.clip((X - self.media_) / self.escala_, -1.0, 1.0).astype("float32")

    def invertir(self, X):
        return (X * self.escala_ + self.media_).astype("float32")

    def ajustar_transformar(self, X):
        return self.ajustar(X).transformar(X)

    def parametros(self):
        return {"media": self.media_, "escala": self.escala_, "n_sigmas": self.n_sigmas}

    @classmethod
    def desde_parametros(cls, media, escala, n_sigmas):
        esc = cls(n_sigmas=float(n_sigmas))
        esc.media_ = media
        esc.escala_ = escala
        return esc


# --- Persistencia ----------------------------------------------------------


def guardar_dataset(ruta, **arrays):
    """Escribe el dataset procesado en formato npz comprimido."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.savez_compressed(ruta, **arrays)


def cargar_dataset(ruta=None):
    """Recupera el dataset procesado generado por el notebook 00.

    Devuelve un diccionario con las particiones ya escaladas, la etiqueta, las
    fechas de cada ventana y los parametros del escalador.
    """
    ruta = ruta or config.RUTA_DATASET
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encuentra {ruta}. Ejecutar antes el notebook "
            "'00_datos_y_preparacion.ipynb'."
        )

    with np.load(ruta, allow_pickle=True) as f:
        datos = {k: f[k] for k in f.files}

    # Las fechas se almacenan como cadenas ISO. Guardarlas como enteros seria
    # fragil, porque la unidad interna de los tipos de fecha depende de la
    # version de pandas y una lectura con la unidad equivocada desplaza toda la
    # serie varias decadas.
    for clave in ("fechas_train", "fechas_val", "fechas_test"):
        if clave in datos:
            datos[clave] = pd.DatetimeIndex(datos[clave].astype(str))

    datos["escalador"] = EscaladorTanh.desde_parametros(
        datos["escalador_media"], datos["escalador_escala"], datos["escalador_n_sigmas"]
    )
    return datos


def submuestra_real(X, y, n_reales, semilla=None):
    """Extrae un subconjunto de entrenamiento conservando la tasa de positivos.

    El presupuesto de datos reales es la variable que crea la escasez que el
    dato sintetico debe compensar. El muestreo es estratificado para que la
    proporcion de episodios de estres no dependa del azar de la extraccion.
    """
    semilla = config.SEMILLA if semilla is None else semilla
    rng = np.random.default_rng(semilla)

    if n_reales >= len(X):
        return X, y

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = int(round(n_reales * len(idx_pos) / len(y)))
    n_pos = max(1, min(n_pos, len(idx_pos)))
    n_neg = n_reales - n_pos

    sel = np.concatenate(
        [
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(sel)
    return X[sel], y[sel]
