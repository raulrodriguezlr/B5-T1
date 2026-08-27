"""Protocolo de comparacion entre datasets con distinta proporcion de sinteticos.

Todos los notebooks de generadores terminan invocando `barrido_ratios`, lo que
garantiza que cada modelo generativo se evalue exactamente bajo las mismas
condiciones: mismo presupuesto de datos reales, mismos multiplicadores, misma
arquitectura, mismas semillas y mismo conjunto de test.
"""

import os

import numpy as np
import pandas as pd

from . import config
from .modelo import entrenar_clasificador, evaluar, umbral_optimo


def mezclar(X_real, y_real, X_synth, y_synth, ratio, semilla=None):
    """Combina el presupuesto real con una cantidad proporcional de sinteticos.

    El numero de muestras sinteticas es `ratio` veces el de reales. Se conserva
    la totalidad de los datos reales en todas las configuraciones, de modo que
    el sintetico actua como ampliacion y nunca como sustituto.
    """
    if ratio <= 0 or X_synth is None or len(X_synth) == 0:
        return X_real, y_real

    rng = np.random.default_rng(config.SEMILLA if semilla is None else semilla)
    n_synth = int(round(ratio * len(X_real)))
    reemplazo = n_synth > len(X_synth)
    idx = rng.choice(len(X_synth), size=n_synth, replace=reemplazo)

    X = np.concatenate([X_real, X_synth[idx]], axis=0)
    y = np.concatenate([y_real, y_synth[idx]], axis=0)

    orden = rng.permutation(len(X))
    return X[orden], y[orden]


def barrido_ratios(
    nombre_modelo,
    X_real,
    y_real,
    X_synth,
    y_synth,
    X_val,
    y_val,
    X_test,
    y_test,
    ratios=None,
    n_semillas=None,
    verbose=True,
):
    """Entrena y evalua el clasificador para cada ratio y semilla.

    Devuelve un DataFrame con una fila por combinacion de ratio y semilla, y un
    diccionario con las curvas de perdida de cada entrenamiento, necesarias para
    documentar la convergencia tal y como pide el enunciado.
    """
    ratios = ratios if ratios is not None else config.RATIOS_SINTETICOS
    n_semillas = n_semillas if n_semillas is not None else config.N_SEMILLAS

    filas = []
    historias = {}

    for ratio in ratios:
        for k in range(n_semillas):
            semilla = config.SEMILLA + k

            X_mix, y_mix = mezclar(X_real, y_real, X_synth, y_synth, ratio, semilla)
            modelo, historia = entrenar_clasificador(
                X_mix, y_mix, X_val, y_val, semilla=semilla
            )

            umbral = umbral_optimo(modelo, X_val, y_val)
            metricas = evaluar(modelo, X_test, y_test, umbral=umbral)

            # El conjunto de test cubre un unico regimen de mercado y sus
            # metricas resultan ruidosas. Se registra tambien el resultado en
            # validacion para poder distinguir el efecto del dato sintetico del
            # efecto del cambio de regimen entre ambos periodos.
            metricas_val = evaluar(modelo, X_val, y_val, umbral=umbral)
            metricas["pr_auc_val"] = metricas_val["pr_auc"]
            metricas["f1_val"] = metricas_val["f1"]

            metricas.update(
                {
                    "modelo": nombre_modelo,
                    "ratio": float(ratio),
                    "semilla": semilla,
                    "n_reales": int(len(X_real)),
                    "n_sinteticos": int(len(X_mix) - len(X_real)),
                    "epochs": len(historia.history["loss"]),
                }
            )
            filas.append(metricas)
            historias[(float(ratio), semilla)] = historia.history

            if verbose:
                print(
                    f"  {nombre_modelo:<10s} ratio={ratio:<5.2f} semilla={semilla} "
                    f"PR-AUC(test)={metricas['pr_auc']:.4f} "
                    f"PR-AUC(val)={metricas['pr_auc_val']:.4f} "
                    f"F1={metricas['f1']:.4f}"
                )

    columnas = [
        "modelo", "ratio", "semilla", "n_reales", "n_sinteticos",
        "pr_auc", "pr_auc_val", "roc_auc", "f1", "f1_val", "precision", "recall",
        "umbral", "vn", "fp", "fn", "vp", "epochs",
    ]
    return pd.DataFrame(filas)[columnas], historias


def resumir(df):
    """Agrega los resultados por modelo y ratio promediando sobre semillas."""
    return (
        df.groupby(["modelo", "ratio"])
        .agg(
            pr_auc_media=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_val_media=("pr_auc_val", "mean"),
            pr_auc_val_std=("pr_auc_val", "std"),
            f1_media=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_media=("recall", "mean"),
            precision_media=("precision", "mean"),
            n_sinteticos=("n_sinteticos", "max"),
        )
        .reset_index()
    )


def guardar_resultados(df, nombre_modelo):
    """Persiste los resultados de un generador en results/tablas."""
    config.asegurar_directorios()
    ruta = os.path.join(config.DIR_TABLAS, f"resultados_{nombre_modelo}.csv")
    df.to_csv(ruta, index=False)
    return ruta


def cargar_resultados(nombres):
    """Reune las tablas de todos los generadores disponibles."""
    trozos = []
    for nombre in nombres:
        ruta = os.path.join(config.DIR_TABLAS, f"resultados_{nombre}.csv")
        if os.path.exists(ruta):
            trozos.append(pd.read_csv(ruta))
        else:
            print(f"Aviso: no se encuentra {ruta}; el modelo '{nombre}' queda fuera.")
    if not trozos:
        raise FileNotFoundError("No hay ninguna tabla de resultados disponible.")
    return pd.concat(trozos, ignore_index=True)


def guardar_sinteticos(nombre_modelo, X_synth, y_synth, perdidas=None):
    """Escribe el dataset sintetico de un generador en el formato acordado.

    Este es el unico punto de contacto entre los notebooks de generadores y el
    resto del proyecto. Cualquier generador que respete esta firma encaja en la
    comparativa sin tocar el codigo comun.
    """
    config.asegurar_directorios()
    ruta = os.path.join(config.DIR_MODELOS, f"sinteticos_{nombre_modelo}.npz")

    arrays = {
        "X_synth": np.asarray(X_synth, dtype="float32"),
        "y_synth": np.asarray(y_synth, dtype="float32"),
    }
    if perdidas is not None:
        for clave, valores in perdidas.items():
            arrays[f"loss_{clave}"] = np.asarray(valores, dtype="float32")

    np.savez_compressed(ruta, **arrays)
    return ruta


def cargar_sinteticos(nombre_modelo):
    """Recupera el dataset sintetico producido por un generador."""
    ruta = os.path.join(config.DIR_MODELOS, f"sinteticos_{nombre_modelo}.npz")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encuentra {ruta}. Ejecutar antes el notebook del generador "
            f"'{nombre_modelo}'."
        )
    with np.load(ruta) as f:
        return {k: f[k] for k in f.files}
