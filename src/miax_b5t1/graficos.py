"""Funciones de representacion compartidas por todos los notebooks.

Concentrar aqui los graficos recurrentes mantiene una estetica homogenea en el
informe y evita repetir el mismo bloque de matplotlib en cada notebook.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from . import config


def aplicar_estilo():
    """Fija el estilo visual comun a todas las figuras del proyecto."""
    plt.style.use(config.ESTILO_MPL)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def guardar(fig, nombre):
    """Almacena una figura en results/figuras y devuelve la ruta."""
    config.asegurar_directorios()
    ruta = os.path.join(config.DIR_FIGURAS, f"{nombre}.png")
    fig.savefig(ruta)
    return ruta


def curva_perdida(historia, titulo, nombre_fichero=None, claves=("loss", "val_loss")):
    """Representa la evolucion de la perdida durante el entrenamiento.

    El enunciado exige mostrar, para cada entrenamiento, evidencia de que el
    modelo ha convergido. Esta figura es la que documenta ese punto.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    etiquetas = {"loss": "entrenamiento", "val_loss": "validacion"}

    for clave in claves:
        if clave in historia:
            ax.plot(historia[clave], label=etiquetas.get(clave, clave), linewidth=1.6)

    ax.set_title(titulo)
    ax.set_xlabel("epoch")
    ax.set_ylabel("perdida")
    ax.legend()
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)
    return fig, ax


def curva_perdida_adversaria(perdida_d, perdida_g, titulo, nombre_fichero=None):
    """Representa la dinamica entre generador y discriminador.

    En un esquema adversario la perdida no desciende de forma monotona: lo que
    se busca es un equilibrio en el que ninguna de las dos redes domine a la
    otra de manera sostenida.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(perdida_d, label="discriminador", linewidth=1.4)
    ax.plot(perdida_g, label="generador", linewidth=1.4)
    ax.set_title(titulo)
    ax.set_xlabel("iteracion")
    ax.set_ylabel("perdida")
    ax.legend()
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)
    return fig, ax


def comparar_trayectorias(X_real, X_synth, titulo, nombre_fichero=None, n=6, activo=0):
    """Enfrenta ventanas reales y sinteticas de un mismo activo.

    Es la primera comprobacion cualitativa de cualquier generador: las series
    producidas deben presentar el aspecto de ruido con agrupamiento de
    volatilidad propio de una rentabilidad diaria, y no trayectorias suaves ni
    saturadas en los extremos del rango.
    """
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2), sharey=True)

    for j in range(n):
        axes[0, j].plot(X_real[j, :, activo], linewidth=0.9, color="#2874a6")
        axes[1, j].plot(X_synth[j, :, activo], linewidth=0.9, color="#c0392b")
        for i in range(2):
            axes[i, j].set_xticks([])

    axes[0, 0].set_ylabel("reales")
    axes[1, 0].set_ylabel("sinteticas")
    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)
    return fig, axes


def comparar_distribuciones(X_real, X_synth, titulo, nombre_fichero=None):
    """Compara marginal, autocorrelacion y estructura de covarianza.

    Un generador puede reproducir el histograma de rentabilidades y aun asi
    fallar en la dependencia entre activos, que es justamente lo que determina
    el comportamiento de una cartera en un episodio de estres. Por eso se
    contrasta tambien la matriz de correlacion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].hist(X_real.ravel(), bins=120, density=True, alpha=0.55,
                 label="reales", color="#2874a6")
    axes[0].hist(X_synth.ravel(), bins=120, density=True, alpha=0.55,
                 label="sinteticas", color="#c0392b")
    axes[0].set_yscale("log")
    axes[0].set_title("Distribucion marginal")
    axes[0].legend()

    def volatilidad_por_paso(X):
        return X.std(axis=(0, 2))

    axes[1].plot(volatilidad_por_paso(X_real), label="reales", color="#2874a6")
    axes[1].plot(volatilidad_por_paso(X_synth), label="sinteticas", color="#c0392b")
    axes[1].set_title("Volatilidad por posicion de la ventana")
    axes[1].set_xlabel("dia dentro de la ventana")
    axes[1].legend()

    corr_real = np.corrcoef(X_real.reshape(-1, X_real.shape[2]).T)
    corr_synth = np.corrcoef(X_synth.reshape(-1, X_synth.shape[2]).T)
    triangulo = np.triu_indices_from(corr_real, k=1)

    axes[2].scatter(corr_real[triangulo], corr_synth[triangulo], s=8, alpha=0.6,
                    color="#1e8449")
    lim = [-0.2, 1.0]
    axes[2].plot(lim, lim, color="#4c4c4c", linestyle="--", linewidth=1)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel("correlacion real")
    axes[2].set_ylabel("correlacion sintetica")
    axes[2].set_title("Correlacion entre activos")

    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)
    return fig, axes


def curva_ratios(resumen, metrica="pr_auc_media", titulo=None, nombre_fichero=None,
                 desviacion="pr_auc_std"):
    """Representa el efecto del volumen de datos sinteticos sobre el test.

    Es la figura central del taller: en el eje horizontal la cantidad de datos
    sinteticos anadidos y en el vertical la calidad alcanzada por el
    clasificador, con una curva por modelo generativo.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for nombre, grupo in resumen.groupby("modelo"):
        grupo = grupo.sort_values("ratio")
        color = config.COLORES_MODELO.get(nombre, None)
        etiqueta = config.NOMBRES_MODELO.get(nombre, nombre)

        ax.plot(grupo["ratio"], grupo[metrica], marker="o", label=etiqueta,
                color=color, linewidth=1.8)

        if desviacion in grupo:
            inferior = grupo[metrica] - grupo[desviacion].fillna(0)
            superior = grupo[metrica] + grupo[desviacion].fillna(0)
            ax.fill_between(grupo["ratio"], inferior, superior, alpha=0.15, color=color)

    ax.set_xlabel("sinteticos anadidos (multiplos del presupuesto real)")
    ax.set_ylabel(metrica.replace("_media", "").replace("_", " ").upper())
    ax.set_title(titulo or "Efecto del dato sintetico sobre el test")
    ax.legend()
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)
    return fig, ax


def novedad_vecino_mas_cercano(X_real, X_synth, titulo, nombre_fichero=None,
                               n_muestra=800, semilla=42):
    """Compara cuanta novedad aporta un generador frente a copiar los datos.

    Para cada muestra sintetica se mide la distancia a la ventana real mas
    proxima. Como referencia se calcula la misma distancia entre muestras reales
    distintas. Un generador que se limite a reproducir el conjunto de
    entrenamiento producira distancias mucho menores que esa referencia, senal
    de memorizacion; uno que las supere ampliamente estara generando ruido sin
    relacion con los datos.
    """
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(semilla)
    plano_real = X_real.reshape(len(X_real), -1)
    plano_synth = X_synth.reshape(len(X_synth), -1)

    idx_s = rng.choice(len(plano_synth), size=min(n_muestra, len(plano_synth)), replace=False)
    idx_r = rng.choice(len(plano_real), size=min(n_muestra, len(plano_real)), replace=False)

    d_synth = pairwise_distances(plano_synth[idx_s], plano_real).min(axis=1)

    # La referencia mide la distancia entre ventanas reales distintas. Se anula
    # la comparacion de cada ventana consigo misma, que valdria cero.
    d_real = pairwise_distances(plano_real[idx_r], plano_real)
    d_real[np.arange(len(idx_r)), idx_r] = np.inf
    d_real = d_real.min(axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(d_real, bins=60, density=True, alpha=0.6, color="#2874a6",
            label="entre ventanas reales")
    ax.hist(d_synth, bins=60, density=True, alpha=0.6, color="#c0392b",
            label="de sintetica a real mas proxima")
    ax.set_xlabel("distancia euclidea")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()

    if nombre_fichero:
        guardar(fig, nombre_fichero)

    resumen = {
        "distancia_media_sintetica": float(d_synth.mean()),
        "distancia_media_real": float(d_real.mean()),
        "ratio": float(d_synth.mean() / d_real.mean()),
    }
    return fig, ax, resumen
