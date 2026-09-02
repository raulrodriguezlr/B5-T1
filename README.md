# Generación de datos financieros sintéticos

**Taller B5-T1 · MIAX**
Raúl, Pietro y Alonso

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

El protocolo es **determinista dentro de una misma máquina**: la referencia
obtenida en el notebook 02 y la configuración sin sintéticos de los cuatro
generadores coinciden en todos los decimales (0,2253 en test, 0,3314 en
validación), y re-ejecutar un notebook completo devuelve exactamente las mismas
cifras.

Esa coincidencia no sobrevive, en cambio, a un cambio de máquina, y conviene
decirlo porque condiciona la lectura de todo lo demás. Ejecutando el mismo
código, con las mismas semillas y el mismo `dataset.npz` byte a byte, tres
equipos del grupo obtuvieron referencias distintas en test: **0,2253, 0,2409 y
0,2443**. Las versiones de Keras y PyTorch eran las mismas; la diferencia está
por debajo, en el orden de las operaciones en punto flotante. La magnitud de esa
discrepancia es del mismo orden que todos los efectos que este trabajo mide, y
es la razón por la que la comparación se apoya en la dispersión entre semillas y
no en el valor puntual.

**Todas las cifras de este README proceden de una única máquina**, y las cinco
tablas de `results/tablas/` se generaron en la misma ejecución. Que la fila
`ratio = 0` sea idéntica en las cinco es la comprobación de que así es.

### Un hallazgo que reorienta el trabajo

La curva de aprendizaje del clasificador reserva una sorpresa:

| Muestras reales | PR-AUC validación | PR-AUC test |
|---|---|---|
| 500 | 0,333 | 0,304 |
| 1.000 | 0,297 | 0,275 |
| 2.000 | 0,331 | 0,261 |
| 3.000 | 0,330 | 0,292 |
| 5.000 | 0,302 | 0,198 |
| 8.000 | 0,320 | 0,238 |
| 11.326 | 0,304 | 0,192 |

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

| Modelo | Familia | Notebook | Coste |
|---|---|---|---|
| Ruido gaussiano | perturbación de los datos originales | `03` | segundos |
| cGAN | adversaria condicional | `04` | 8 min |
| CVAE | latente variacional condicional | `05` | 4 min |
| Difusión | difusión condicional | `06` | 20 min |

Los cuatro son **condicionales**: se les puede pedir explícitamente muestras de
la clase positiva, que es el punto del ejercicio dada la escasez de episodios de
estrés. Cada uno genera 12.000 muestras y se evalúa con el mismo barrido de
proporciones `{0, 0,25, 0,5, 1, 2, 4}` sobre el presupuesto real, repitiendo
cada configuración con 5 semillas.

El modelo de difusión se aparta de la formulación clásica en un punto que merece
mención. La receta habitual entrena la red para estimar el ruido añadido, pero
aquí no funciona: las rentabilidades diarias son casi ruido blanco en la
dirección temporal —autocorrelación de 0,019— y en ese régimen el ruido no es
recuperable. El error se estancaba en 0,80 sobre un máximo de 1,00 y el muestreo
divergía. Estimando la ventana limpia en lugar del ruido, y recortándola en cada
paso, el error baja a 0,338 y el proceso se estabiliza.

---

## 4. Resultados

### Referencia sin datos sintéticos

| Métrica | Test | Validación |
|---|---|---|
| PR-AUC | 0,225 ± 0,094 | 0,331 ± 0,021 |
| F1 | 0,223 ± 0,051 | — |
| Clasificador sin información | 0,104 | 0,128 |
| Regresión logística sobre descriptores | **0,350** | — |

La red extrae señal: más que duplica la tasa base. Pero la última fila obliga a
matizar. Una regresión logística sobre seis descriptores agregados de la ventana
—volatilidad, retorno acumulado, drawdown corriente, mínimo diario, dispersión
entre activos y rentabilidad absoluta media— alcanza 0,350 en test, por encima
de la red convolucional. La comparación no es simétrica, porque la lineal se
entrena con el histórico completo y la red con el presupuesto de 3.000 ventanas,
pero acota lo que se está midiendo: **la convolución no aporta nada sobre un
resumen estadístico sencillo de la ventana**.

La búsqueda de arquitectura del notebook 02 apunta en la misma dirección. De las
diez configuraciones probadas —variando profundidad, anchura, tamaño del núcleo
y regularización— la mejor supera a la de referencia en +0,017, frente a una
dispersión entre semillas de 0,066. Ninguna se distingue de las demás, y se
mantiene la configuración del material del taller.

### Efecto de los datos sintéticos

PR-AUC en test según la proporción de sintéticos añadida sobre el presupuesto
real:

| Proporción | Ruido | cGAN | CVAE | Difusión |
|---|---|---|---|---|
| 0 (referencia) | 0,225 | 0,225 | 0,225 | 0,225 |
| 0,25 | **0,267** | 0,215 | **0,257** | **0,254** |
| 0,5 | 0,191 | 0,153 | 0,234 | 0,220 |
| 1 | 0,217 | 0,187 | 0,222 | 0,188 |
| 2 | 0,167 | 0,212 | 0,219 | 0,164 |
| 4 | 0,164 | 0,207 | 0,207 | 0,224 |

De las veinte configuraciones contrastadas frente a su propia referencia
—contraste de medias pareado por semilla— **ninguna mejora de forma
significativa y seis degradan de forma significativa**.

| Generador | mejor variación | proporción | p |
|---|---|---|---|
| Ruido gaussiano | +0,042 | 0,25 | 0,186 |
| CVAE | +0,032 | 0,25 | 0,385 |
| Difusión | +0,029 | 0,25 | 0,215 |
| cGAN | 0,000 | — | — |

Las degradaciones, en cambio, sí se miden. El CVAE empeora la validación de
forma monótona a partir de la proporción 1 (−0,030, −0,043 y −0,064 con p de
0,014, 0,009 y 0,025), la difusión a proporción 0,5 (−0,037, p = 0,033) y el
ruido a proporción 2 en test (−0,058, p = 0,025).

**La respuesta a la pregunta del taller es, en este problema y con estos
generadores, que no.**

### Por qué no

La razón estaba anunciada en la sección 2: la curva de aprendizaje es plana en
validación y descendente en test, de modo que el problema no está limitado por
el volumen de datos. Añadir observaciones —reales o sintéticas— no puede ayudar.
Un generador sólo serviría si produjese episodios de estrés plausibles y
distintos de los observados, y ninguno lo hace.

### Calidad de las muestras

| | Desviación | Curtosis | Correlación entre activos | Novedad |
|---|---|---|---|---|
| **Reales** | **0,241** | **2,20** | **0,266** | 1,00 |
| Ruido gaussiano | 0,248 | 1,83 | 0,249 | 0,22 |
| cGAN | 0,281 | 2,88 | 0,156 | 1,04 |
| CVAE | 0,047 | 1,28 | 0,768 | 0,56 |
| Difusión | 0,392 | 0,03 | 0,435 | 1,45 |

La columna de novedad mide la distancia media de cada muestra sintética a la
ventana real más próxima, dividida por la distancia típica entre dos ventanas
reales distintas.

Los tres generadores neuronales fallan en la **correlación entre activos**, que
es precisamente la propiedad que determina la etiqueta: el drawdown se mide
sobre la cartera equiponderada, y es la sincronía entre activos la que decide su
magnitud. El cGAN la subestima; el CVAE y la difusión la sobrestiman.

El CVAE merece mención aparte. Su desviación típica es **cinco veces menor** que
la real y su correlación casi triple: genera ventanas casi planas en las que los
veintitrés activos se mueven al unísono. Es el suavizado característico de un
autocodificador variacional llevado al extremo, y explica por qué es el
generador que más daño hace.

### Realismo y utilidad no coinciden

El cGAN es, de los tres neuronales, el que mejor reproduce la distribución
marginal —dispersión y curtosis casi exactas— y el único cuya novedad se sitúa
justo en la distancia que separa a dos ventanas reales. Y es también el único
cuya mejor variación es exactamente cero. La difusión, en el otro extremo,
produce las muestras más alejadas del conjunto de entrenamiento y tampoco aporta
nada: esa distancia mide alejamiento de la distribución real, no exploración de
regiones plausibles.

El generador simple del notebook 03 es el que menos daño hace en proporciones
pequeñas, por la razón trivial de que sus muestras son datos reales perturbados.
Ordenados por coste computacional —segundos, 4, 8 y 20 minutos— el orden resulta
ser exactamente el inverso al de utilidad. **Ninguno de los tres modelos
generativos justifica su coste frente a añadir ruido gaussiano.**

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

Los notebooks 01 a 07 no necesitan ninguna descarga: el repositorio incluye
`data/processed/precios_universo.csv` con los 23 activos ya filtrados, y de ahí
sale todo el dataset.

**El notebook 00 es la excepción.** Su primera sección describe el universo de
partida y para ello lee `data/raw/precios_close_sp500.csv`, que no se versiona
por tamaño. Quien parta de un clon limpio debe ejecutar antes
`scripts/descargar_precios.py`. Conviene saber que ese script reescribe también
`data/processed/precios_universo.csv` con la descarga del día: para reproducir
las cifras de la sección 4 hay que restaurar el fichero versionado
(`git checkout data/processed/precios_universo.csv`) antes de ejecutar el
notebook.

**Sobre el entorno:** Python 3.14 no dispone de TensorFlow, de modo que Keras 3
se ejecuta sobre PyTorch mediante `KERAS_BACKEND=torch`. La API de capas y
modelos es idéntica. Todo corre en CPU. Tiempos medidos en un portátil
convencional: notebook 00 unos 2 minutos, 01 y 07 menos de uno, 02 unos 22
—incluida la búsqueda de arquitectura—, 03 unos 15, 04 unos 25, 05 unos 15 y 06
unos 36. Los notebooks funcionan también en Colab sin cambios.

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
│   ├── 05_generador_cvae.ipynb         latente variacional condicional
│   ├── 06_generador_diffusion.ipynb    difusión condicional
│   └── 07_comparativa_final.ipynb      contrastes y comparativa
├── data/  models/
└── results/
    ├── figuras/                    todas las figuras del informe
    └── tablas/
        ├── resultados_<modelo>.csv     una fila por ratio y semilla
        ├── busqueda_arquitectura.csv   rejilla del notebook 02
        ├── contrastes.csv              las 20 comparaciones con su p
        ├── comparativa_final.csv       tabla resumen del informe
        └── calidad_frente_utilidad.csv realismo frente a efecto
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
| Tres modelos generativos de tipos distintos | notebooks 04 (cGAN), 05 (CVAE), 06 (difusión) |
| Cuarto modelo simple con ruido | notebook 03 |
| Datasets con distinta proporción real/sintético | `experimento.barrido_ratios`, proporciones `{0 … 4}` |
| Arquitectura buscada sobre los datos reales | notebook 02 §7, `modelo.buscar_arquitectura` |
| Misma arquitectura en todas las versiones | `modelo.construir_clasificador` |
| Curvas de loss de cada entrenamiento | figuras `*_perdidas_clasificador` y `*_convergencia_*` |
| Gráficos de análisis de resultados | `results/figuras/` |
| Código que genera todas las tablas y gráficas | notebooks y `src/miax_b5t1/graficos.py` |
| Análisis del efecto de la proporción de sintéticos | `README` §4, notebook 07 |
| Comparación entre los cuatro generadores | notebook 07, `comparativa_final.csv` |
