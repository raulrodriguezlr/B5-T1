"""Clasificador de referencia y protocolo unico de entrenamiento y evaluacion.

El enunciado exige que todas las versiones del modelo compartan arquitectura.
Definirla en un solo lugar asegura que las diferencias observadas entre
configuraciones procedan de los datos de entrenamiento y no de cambios
accidentales en la red.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import keras
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from keras.models import Sequential

from . import config


# Configuracion de referencia. Es la que emplean todas las versiones del
# clasificador en la comparativa entre generadores, y la que la busqueda de
# arquitectura del notebook 02 toma como punto de partida.
ARQUITECTURA_REFERENCIA = {
    "filtros": (64, 128, 128),
    "kernel": 3,
    "densa": 100,
    "dropout": 0.3,
}


def construir_clasificador(n_pasos, n_activos, semilla=None, filtros=None,
                           kernel=None, densa=None, dropout=None):
    """Red convolucional unidimensional para clasificacion binaria.

    La estructura reproduce la del clasificador empleado en el taller: bloques
    de convolucion y submuestreo que extraen patrones locales de la ventana
    temporal, seguidos de una capa densa. La unica adaptacion es la salida, que
    pasa a ser una unidad con activacion sigmoide por tratarse de un problema
    de clasificacion y no de regresion.

    Los hiperparametros son opcionales y, omitidos, reproducen exactamente la
    configuracion de referencia: la busqueda del notebook 02 los recorre, pero
    ninguna otra parte del proyecto los toca, de modo que la comparativa entre
    generadores sigue empleando una unica arquitectura.
    """
    if semilla is not None:
        keras.utils.set_random_seed(semilla)

    filtros = ARQUITECTURA_REFERENCIA["filtros"] if filtros is None else filtros
    kernel = ARQUITECTURA_REFERENCIA["kernel"] if kernel is None else kernel
    densa = ARQUITECTURA_REFERENCIA["densa"] if densa is None else densa
    dropout = ARQUITECTURA_REFERENCIA["dropout"] if dropout is None else dropout

    capas = [Input(shape=(n_pasos, n_activos))]
    for n_filtros in filtros:
        capas.append(Conv1D(filters=n_filtros, kernel_size=kernel, activation="relu"))
        capas.append(MaxPooling1D(pool_size=2))
    capas += [
        Flatten(),
        Dense(densa, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ]

    modelo = Sequential(capas, name="clasificador_drawdown")

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return modelo


# --- Busqueda de arquitectura ----------------------------------------------

# El enunciado pide, ademas del entrenamiento y la evaluacion, que se busque una
# arquitectura valida usando los datos reales. La rejilla recorre las cuatro
# decisiones con mas peso sobre la capacidad de la red —profundidad, anchura de
# los bloques convolucionales, tamano del nucleo y regularizacion— alrededor de
# la configuracion de referencia, que aparece marcada en los resultados.
REJILLA_ARQUITECTURA = [
    {"filtros": (32, 64, 64), "kernel": 3, "densa": 100, "dropout": 0.3},
    {"filtros": (64, 128, 128), "kernel": 3, "densa": 100, "dropout": 0.3},
    {"filtros": (64, 128, 256), "kernel": 3, "densa": 100, "dropout": 0.3},
    {"filtros": (64, 128, 128), "kernel": 5, "densa": 100, "dropout": 0.3},
    {"filtros": (64, 128, 128), "kernel": 3, "densa": 50, "dropout": 0.3},
    {"filtros": (64, 128, 128), "kernel": 3, "densa": 200, "dropout": 0.3},
    {"filtros": (64, 128, 128), "kernel": 3, "densa": 100, "dropout": 0.1},
    {"filtros": (64, 128, 128), "kernel": 3, "densa": 100, "dropout": 0.5},
    {"filtros": (32, 64), "kernel": 3, "densa": 100, "dropout": 0.3},
    {"filtros": (64, 128), "kernel": 5, "densa": 50, "dropout": 0.5},
]


def buscar_arquitectura(X_train, y_train, X_val, y_val, rejilla=None,
                        semillas=3, verbose=True):
    """Recorre la rejilla y ordena las configuraciones por su PR-AUC en validacion.

    La busqueda emplea unicamente el presupuesto de datos reales y la particion
    de validacion. **El conjunto de test no interviene en ningun momento**, de
    modo que la arquitectura elegida no queda contaminada por el periodo sobre
    el que despues se informa.

    Conviene leer las cifras sabiendo que la misma particion de validacion fija
    tambien la parada temprana y el punto de corte, de modo que son optimistas
    en terminos absolutos. Lo que la tabla permite es ordenar configuraciones
    entre si, que es para lo que se usa.
    """
    rejilla = REJILLA_ARQUITECTURA if rejilla is None else rejilla
    filas = []

    for hiper in rejilla:
        completo = dict(ARQUITECTURA_REFERENCIA)
        completo.update(hiper)

        pr_auc, f1, epochs = [], [], []
        for k in range(semillas):
            semilla = config.SEMILLA + k
            red, historia = entrenar_clasificador(
                X_train, y_train, X_val, y_val, semilla=semilla, hiper=completo)
            metricas = evaluar(red, X_val, y_val,
                               umbral=umbral_optimo(red, X_val, y_val))
            pr_auc.append(metricas["pr_auc"])
            f1.append(metricas["f1"])
            epochs.append(len(historia.history["loss"]))

        n_params = construir_clasificador(
            X_train.shape[1], X_train.shape[2], semilla=config.SEMILLA,
            **completo).count_params()

        filas.append({
            "filtros": "-".join(str(f) for f in completo["filtros"]),
            "kernel": completo["kernel"],
            "densa": completo["densa"],
            "dropout": completo["dropout"],
            "parametros": int(n_params),
            "pr_auc_val_media": float(np.mean(pr_auc)),
            "pr_auc_val_std": float(np.std(pr_auc, ddof=1)) if semillas > 1 else 0.0,
            "f1_val_media": float(np.mean(f1)),
            "epochs_medios": float(np.mean(epochs)),
            "referencia": completo == ARQUITECTURA_REFERENCIA,
        })

        if verbose:
            print(f"  filtros={filas[-1]['filtros']:<12s} kernel={completo['kernel']} "
                  f"densa={completo['densa']:<4d} dropout={completo['dropout']:.1f}  "
                  f"params={n_params:>8,}  "
                  f"PR-AUC(val)={filas[-1]['pr_auc_val_media']:.4f}"
                  f" +- {filas[-1]['pr_auc_val_std']:.4f}"
                  f"{'   <- referencia' if filas[-1]['referencia'] else ''}")

    return (pd.DataFrame(filas)
            .sort_values("pr_auc_val_media", ascending=False)
            .reset_index(drop=True))


def pesos_de_clase(y):
    """Pesos inversamente proporcionales a la frecuencia de cada clase.

    Con una tasa de positivos cercana al diez por ciento, la entropia cruzada
    sin ponderar tiende a la solucion trivial de predecir siempre la clase
    mayoritaria. La ponderacion se aplica de forma identica en todas las
    configuraciones, de modo que no favorece a ninguna en la comparativa.
    """
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    total = n_pos + n_neg
    return {0: total / (2.0 * n_neg), 1: total / (2.0 * n_pos)}


def entrenar_clasificador(X_train, y_train, X_val, y_val, semilla=None, verbose=0,
                          hiper=None):
    """Entrena el clasificador con parada temprana sobre la precision-recall.

    La parada temprana vigila el area bajo la curva de precision y exhaustividad
    en validacion, metrica adecuada cuando la clase de interes es minoritaria,
    y restaura los pesos del mejor epoch para que el resultado no dependa del
    momento exacto en que se detiene el ajuste.
    """
    modelo = construir_clasificador(X_train.shape[1], X_train.shape[2], semilla=semilla,
                                    **(hiper or {}))

    parada = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=config.PACIENCIA,
        restore_best_weights=True,
    )

    historia = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS_CLASIFICADOR,
        batch_size=config.BATCH_CLASIFICADOR,
        class_weight=pesos_de_clase(y_train),
        callbacks=[parada],
        verbose=verbose,
    )
    return modelo, historia


def umbral_optimo(modelo, X_val, y_val):
    """Busca el punto de corte que maximiza F1 sobre el conjunto de validacion.

    El corte por defecto en 0.5 no es adecuado cuando las clases estan
    desequilibradas y las probabilidades han sido ponderadas. El umbral se
    ajusta en validacion y se aplica despues al test sin volver a mirarlo.
    """
    p_val = modelo.predict(X_val, verbose=0).ravel()
    candidatos = np.linspace(0.05, 0.95, 91)
    puntuaciones = [f1_score(y_val, (p_val >= u).astype(int), zero_division=0) for u in candidatos]
    return float(candidatos[int(np.argmax(puntuaciones))])


def evaluar(modelo, X_test, y_test, umbral=0.5):
    """Calcula las metricas de test para un clasificador ya entrenado."""
    p_test = modelo.predict(X_test, verbose=0).ravel()
    y_pred = (p_test >= umbral).astype(int)

    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    return {
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "umbral": float(umbral),
        "vn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(tp),
    }
