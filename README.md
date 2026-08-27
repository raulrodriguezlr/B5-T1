# Generación de datos financieros sintéticos

**Taller B5-T1 · MIAX**
Raúl Rodríguez, Pietro y Alonso

Este repositorio estudia si los datos sintéticos generados por modelos
neuronales mejoran un clasificador que trata de anticipar caídas severas del
mercado. Se comparan tres modelos generativos de familias distintas y un cuarto
modelo elemental que sirve de referencia mínima.

---

## 1. El problema

**Anticipar episodios de caída severa del S&P 500.**

Dada la ventana de rentabilidades de los últimos 60 días, se predice si la
cartera sufrirá un retroceso pronunciado durante los 30 días siguientes.

| Elemento | Definición |
|---|---|
| Universo | 23 valores del S&P 500 con historia completa desde 1962 |
| Entrada `X` | ventana de 60 sesiones × 23 activos de rentabilidades logarítmicas |
| Etiqueta `y` | 1 si el drawdown máximo de la cartera equiponderada en los 30 días siguientes es peor que −7,91 % |
| Clasificador | CNN 1D, idéntica en todas las configuraciones |
| Métrica | PR-AUC sobre la clase positiva |
| Muestras | 16.180 ventanas, 10 % positivas |

El umbral de −7,91 % no se eligió a mano: es el cuantil del 10 % de la
distribución de drawdown calculada **solo sobre el bloque de entrenamiento**.
Calibrarlo sobre la muestra completa habría filtrado información del test hacia
la propia definición del problema.

### Por qué este problema necesita datos sintéticos

El conjunto de entrenamiento contiene 1.137 ventanas positivas, cifra que en
apariencia no indica escasez. Pero las ventanas se solapan: dos ventanas
separadas por un día comparten 59 de sus 60 sesiones. El recuento relevante no
es el de ventanas sino el de **episodios independientes**:

| Partición | Ventanas positivas | Episodios distintos |
|---|---|---|
| Entrenamiento | 1.137 | **51** |
| Validación | 299 | 10 |
| Test | 243 | **8** |

Cincuenta y un episodios en 45 años. Y si en lugar de muestrear sobre toda la
historia se toman las sesiones más recientes del entrenamiento, como haría quien
solo dispone de datos actuales, el resultado es aún más elocuente:

```
últimas   500 ventanas ->   0 positivas (0.0%)
últimas  1000 ventanas ->   0 positivas (0.0%)
últimas  2000 ventanas -> 257 positivas (12.8%)
```

Los tramos recientes cortos no contienen ningún episodio de estrés. No es que
falten observaciones: falta el fenómeno entero. Ésa es la carencia concreta que
un modelo generativo condicionado por clase podría compensar.

---

## 2. Decisiones metodológicas

Tres decisiones condicionan todos los resultados y conviene tenerlas presentes
al leerlos.

**Partición cronológica con embargo.** Un reparto aleatorio de ventanas
solapadas situaría observaciones casi idénticas a ambos lados de la frontera y
produciría métricas infladas. Se usan bloques temporales consecutivos
(train 1962-2007, validación 2007-2016, test 2017-2026) separados por un hueco de
90 sesiones, igual a la suma de la ventana de entrada y el horizonte de la
etiqueta.

**Los generadores solo ven el presupuesto real.** Todos se entrenan con las
mismas 3.000 ventanas de entrenamiento, nunca con el histórico completo ni con
validación o test. Si un generador viera datos vedados al clasificador, la
comparación no significaría nada.

**Arquitectura y protocolo únicos.** La CNN y el procedimiento de evaluación
viven en `src/miax_b5t1/` y son los mismos para las cuatro configuraciones, de
modo que las diferencias observadas sean atribuibles a los datos.

El protocolo es además **reproducible bit a bit**: la referencia obtenida en el
notebook 02 y la configuración sin sintéticos de cada generador coinciden en
todos los decimales (0,2443 en test, 0,3303 en validación), y re-ejecutar un
notebook completo devuelve exactamente las mismas cifras.

### Un hallazgo que reorienta el trabajo

La curva de aprendizaje del clasificador reserva una sorpresa:

| Muestras reales | PR-AUC validación | PR-AUC test |
|---|---|---|
| 500 | 0,333 | 0,304 |
| 1.000 | 0,297 | 0,276 |
| 3.000 | 0,328 | 0,273 |
| 8.000 | 0,306 | 0,226 |
| 11.326 | 0,324 | 0,191 |

*(Cifras de la curva de aprendizaje, con submuestreo aleatorio distinto por
semilla. La referencia definitiva del experimento, con el subconjunto fijo, es
la de la sección 4.)*

En validación el rendimiento es **plano**: multiplicar por veinte los datos no lo
mejora. En test incluso desciende, porque entrenar con más historia significa
entrenar con más décadas antiguas, y el test corresponde a un régimen de mercado
distinto.

La conclusión conviene enunciarla sin rodeos: **este problema no está limitado
por el volumen de datos**. Lo que escasea es la variedad de episodios de estrés,
no el número de observaciones. Eso delimita lo que cabe esperar de un generador:
su utilidad no está en fabricar más observaciones, que sobran, sino en producir
configuraciones de estrés plausibles y distintas entre sí.

---

## 3. Los cuatro generadores

| Modelo | Familia | Notebook | Responsable |
|---|---|---|---|
| Ruido gaussiano | perturbación de los datos originales | `03` | Raúl |
| cGAN | adversaria condicional | `04` | Raúl |
| CVAE | latente variacional condicional | `05` | *por asignar* |
| Diffusion | difusión condicional | `06` | *por asignar* |

Los cuatro son **condicionales**: se les puede pedir explícitamente muestras de
la clase positiva, que es el punto del ejercicio dada la escasez de episodios de
estrés.

Cada uno genera 12.000 muestras y se evalúa con el mismo barrido de proporciones
`{0, 0,25, 0,5, 1, 2, 4}` sobre el presupuesto real, repitiendo cada
configuración con 5 semillas.

---

## 4. Resultados

> Faltan el CVAE y el modelo de difusión, aún sin asignar. Las cifras de esta
> sección proceden de `results/tablas/` y las genera el código de los notebooks.

### Referencia sin datos sintéticos

| Métrica | Test | Validación |
|---|---|---|
| PR-AUC | 0,244 ± 0,097 | 0,330 ± 0,021 |
| F1 | 0,238 | — |
| Clasificador sin información | 0,104 | 0,128 |

El modelo extrae señal real: más que duplica la tasa base. En términos absolutos
el resultado es modesto, lo cual era esperable, ya que las clases se solapan de
forma acusada y anticipar caídas de mercado es un problema genuinamente difícil.

Conviene fijarse en la dispersión. En test el coeficiente de variación entre
semillas es de 0,40; en validación, de 0,06. El test cubre un único régimen con
solo 8 episodios de estrés, de modo que sus métricas son muy ruidosas y la
comparación se apoya en ambas particiones.

### Efecto de los datos sintéticos

PR-AUC en test según la proporción de sintéticos añadida sobre el presupuesto
real:

| Proporción | Ruido | cGAN |
|---|---|---|
| 0 (referencia) | 0,244 ± 0,097 | 0,244 ± 0,097 |
| 0,25 | 0,259 | 0,255 |
| 0,5 | 0,202 | 0,272 |
| 1 | 0,234 | 0,281 |
| **2** | 0,180 | **0,289 ± 0,038** |
| 4 | 0,138 | 0,270 |

Los dos generadores se comportan de forma opuesta. El **ruido degrada** el
clasificador de forma monótona: en validación la caída a proporción 4 es de
−0,054 con p = 0,009, estadísticamente significativa. El **cGAN mejora** hasta
proporción 2 y decae después.

Sobre la magnitud de la mejora del cGAN conviene ser preciso:

- La mejora de **F1 en proporción 2 es de +0,072 con p = 0,040**, significativa.
- La de PR-AUC en test es de +0,045 pero **p = 0,380**: no alcanza significación.
  La variabilidad del test es demasiado alta para detectar un efecto de ese
  tamaño con cinco semillas.
- En validación no se aprecia mejora.

La lectura honesta es que existe un efecto favorable de tamaño pequeño, no una
mejora rotunda. Presentarlo de otro modo no resistiría el contraste con la
variabilidad entre semillas.

**El efecto más claro no es el de la media sino el de la varianza.** La
desviación típica del PR-AUC entre semillas cae de 0,097 a 0,038 al añadir datos
sintéticos del cGAN. Con independencia de cuánto suba la media, el clasificador
se vuelve mucho menos sensible a la inicialización, lo que en un problema con tan
pocos episodios de estrés tiene valor por sí mismo.

### Calidad de las muestras

| Propiedad | Reales | cGAN |
|---|---|---|
| Desviación típica | 0,241 | 0,273 |
| Curtosis | 2,20 | 2,90 |
| Correlación media entre activos | 0,266 | 0,182 |

El cGAN conserva las colas pesadas —incluso las acentúa— pero **subestima
sistemáticamente la correlación entre activos**. Como el drawdown se mide sobre
la cartera agregada y es la sincronía entre activos la que determina su magnitud,
ésa es la limitación más relevante del generador en este problema concreto.

La prueba de novedad separa nítidamente a los dos generadores. Las muestras de
ruido se sitúan a distancia 2,2 de su ventana real más próxima, cuando dos
ventanas reales distintas distan unas 9 unidades: son variantes locales de los
datos, no configuraciones nuevas. Las del cGAN sí exploran regiones no ocupadas
por el conjunto de entrenamiento.

Esto explica la diferencia de comportamiento entre ambos y justifica el coste de
entrenar un modelo generativo frente a limitarse a perturbar los datos.

---

## 5. Reproducir el trabajo

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m ipykernel install --user --name miax-b5t1 --display-name "Python (MIAX B5-T1)"
```

Después se ejecutan los notebooks en orden, seleccionando el kernel
**Python (MIAX B5-T1)**. Los notebooks 03 a 06 son independientes entre sí una
vez ejecutado el 00.

No hace falta descargar datos: el repositorio incluye
`data/processed/precios_universo.csv` con los 23 activos ya filtrados. La
descarga completa (`scripts/descargar_precios.py`) solo es necesaria para
cambiar el universo de partida.

**Sobre el entorno:** Python 3.14 no dispone de TensorFlow, de modo que Keras 3
se ejecuta sobre PyTorch mediante `KERAS_BACKEND=torch`. La API de capas y
modelos es idéntica. Todo corre en CPU: un entrenamiento del clasificador tarda
unos 7 segundos y el cGAN unos 7 minutos. Los notebooks funcionan también en
Colab sin cambios.

---

## 6. Estructura

```
├── scripts/descargar_precios.py    descarga y cachea los precios
├── src/miax_b5t1/
│   ├── config.py                   parámetros del proyecto
│   ├── datos.py                    dataset, etiqueta, particiones, escalado
│   ├── modelo.py                   CNN clasificadora y evaluación
│   ├── experimento.py              barrido de proporciones y persistencia
│   └── graficos.py                 figuras compartidas
├── notebooks/
│   ├── 00_datos_y_dataset.ipynb        datos, limpieza, dataset
│   ├── 01_analisis_exploratorio.ipynb  EDA y diagnóstico de la escasez
│   ├── 02_clasificador_base.ipynb      arquitectura y referencia
│   ├── 03_generador_ruido.ipynb        modelo simple
│   ├── 04_generador_cgan.ipynb         adversario condicional
│   ├── 05_generador_cvae.ipynb         (esqueleto)
│   ├── 06_generador_diffusion.ipynb    (esqueleto)
│   └── 07_comparativa_final.ipynb      (esqueleto)
├── data/  models/  results/
```

El contrato entre notebooks es mínimo: cada generador lee
`data/processed/dataset.npz` y escribe `models/sinteticos_<modelo>.npz` y
`results/tablas/resultados_<modelo>.csv`. El notebook de comparativa consume esas
tablas.

---

## 7. Cumplimiento del enunciado

| Requisito | Dónde |
|---|---|
| Problema financiero justificado | `README` §1, notebooks 00 y 01 |
| Tres modelos generativos de tipos distintos | notebooks 04, 05, 06 |
| Cuarto modelo simple con ruido | notebook 03 |
| Datasets con distinta proporción real/sintético | `experimento.barrido_ratios`, proporciones `{0 … 4}` |
| Misma arquitectura en todas las versiones | `modelo.construir_clasificador` |
| Curvas de loss de cada entrenamiento | figuras `*_perdidas_clasificador` y `*_convergencia_*` |
| Gráficos de análisis de resultados | `results/figuras/` |
| Código que genera todas las tablas y gráficas | notebooks y `src/miax_b5t1/graficos.py` |
