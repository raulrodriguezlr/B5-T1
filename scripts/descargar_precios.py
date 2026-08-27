"""Descarga y cachea los precios de cierre ajustados del universo S&P 500.

La descarga desde Yahoo Finance es lenta y depende de la disponibilidad del
servicio, por lo que se ejecuta una sola vez y el resultado se guarda en disco.
Los notebooks leen el cache y nunca vuelven a descargar.

Uso:
    python scripts/descargar_precios.py
"""

import os
import warnings

import pandas as pd
import yfinance as yf

warnings.simplefilter(action="ignore", category=FutureWarning)

URL_TICKERS = (
    "https://raw.githubusercontent.com/alfonso-santos/"
    "microcredencial-carteras-python-2023/main/Tema_5_APT/data/sp500_tickers.csv"
)

FECHA_INICIO = "1945-01-01"
RUTA_RAW = os.path.join("data", "raw", "precios_close_sp500.csv")
RUTA_UNIVERSO = os.path.join("data", "processed", "precios_universo.csv")


def main():
    tickers_sp500 = list(pd.read_csv(URL_TICKERS))
    print(f"Tickers en el universo de partida: {len(tickers_sp500)}")

    precios_close = yf.download(
        tickers_sp500,
        start=FECHA_INICIO,
        auto_adjust=True,
        progress=True,
    )["Close"]

    print(f"Matriz de precios descargada: {precios_close.shape}")

    os.makedirs(os.path.dirname(RUTA_RAW), exist_ok=True)
    precios_close.to_csv(RUTA_RAW)
    print(f"Cache escrito en {RUTA_RAW}")

    supervivientes = precios_close.dropna(axis=1)
    print(f"Series con historia completa desde {FECHA_INICIO}: {supervivientes.shape}")

    # Cache reducido, ligero y versionable: permite ejecutar el proyecto tras
    # clonar el repositorio sin repetir la descarga.
    os.makedirs(os.path.dirname(RUTA_UNIVERSO), exist_ok=True)
    supervivientes.to_csv(RUTA_UNIVERSO)
    tamano = os.path.getsize(RUTA_UNIVERSO) / 1e6
    print(f"Cache del universo escrito en {RUTA_UNIVERSO} ({tamano:.1f} MB)")


if __name__ == "__main__":
    main()
