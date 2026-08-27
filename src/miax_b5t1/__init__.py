"""Utilidades compartidas del taller B5-T1.

El paquete concentra la construccion del dataset, la arquitectura del
clasificador y el protocolo de evaluacion, de modo que todos los notebooks
trabajen sobre exactamente las mismas definiciones.
"""

from . import config, datos, experimento, graficos, modelo

__all__ = ["config", "datos", "experimento", "graficos", "modelo"]
