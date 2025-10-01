# Práctica 2 - VC

Este documento recoge el desarrollo de la segunda práctica de la asignatura **Visión por Computador**. En esta práctica se realizan **4 tareas**, cada una enfocada en la detección de bordes, análisis de píxeles y procesamiento de imágenes en tiempo real con Python.


## Autores

- Jesus Santacruz Martín-Delgado
- Roberto Tejero Martín

## Librerías utilizadas

```python
import cv2  
import numpy as np
import matplotlib.pyplot as plt
```

- **OpenCV (cv2):** Se encarga de la captura, procesamiento y visualización de imágenes y vídeo en tiempo real, así como la detección de bordes con Canny y Sobel.

- **NumPy (np):** Permite manipular de forma eficiente los datos de las imágenes y realizar operaciones matemáticas sobre ellos.

- **Matplotlib (plt):** Se usa para mostrar imágenes, gráficos y visualizar el análisis de píxeles por filas y columnas.

---

## Tarea 1 - Análisis de píxeles por filas con Canny

En esta primera tarea se realizó el **conteo de píxeles blancos por filas** utilizando el detector de bordes Canny sobre la imagen del mandril.

El procedimiento fue:

1. Aplicar el detector de bordes Canny a la imagen en escala de grises.

2. Utilizar `cv2.reduce()` para contar píxeles blancos por filas y columnas.

3. Normalizar los valores y encontrar el máximo.

4. Identificar las filas con un número de píxeles blancos ≥ 0.90 * máximo.

5. Visualizar los resultados con gráficos y líneas de referencia.

### Fragmento de código

```python
# Aplica Canny (detector de bordes)
canny = cv2.Canny(gris, 100, 200)

# Suma los valores de los pixeles blancos por columna y por fila
col_counts = cv2.reduce(canny, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
row_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)

# Normaliza los valores (0-1)
cols = col_counts[0] / (255 * canny.shape[0])
rows = row_counts[:,0] / (255 * canny.shape[1])
```

---

## Tarea 2 - Comparación entre Sobel y Canny

El objetivo de esta tarea era **comparar los resultados del detector Sobel con Canny** realizando el mismo análisis de conteo de píxeles por filas y columnas.

El proceso consistió en aplicar el filtro Sobel en ambas direcciones (x e y), convertir a 8 bits, y realizar el conteo similar al de Canny. Posteriormente se marcaron las filas y columnas significativas sobre la imagen original del mandril.

### Fragmento de código

```python
# Aplica el filtro Sobel en ambas direcciones para detectar bordes
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)  # x
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)  # y
sobel = cv2.add(sobelx, sobely)

# Conversión a byte con openCV
sobel8 = cv2.convertScaleAbs(sobel)

# Aplica un umbral para binarizar la imagen
valorUmbral = 130
_, imagenUmbralizada = cv2.threshold(gris, valorUmbral, 255, cv2.THRESH_BINARY)
```

---

## Tarea 3 - Demostrador de procesamiento en tiempo real

En esta tarea se desarrolló un **demostrador interactivo con la cámara** que permite alternar entre diferentes modos de procesamiento de imagen.

Los modos implementados fueron:

- **Modo 1**: Filtro Sobel aplicado directamente sobre BGR para detección de bordes.

- **Modo 2**: Umbralización con valor alto (200) para binarizar la imagen.

- **Modo 3**: Detección de bordes con Canny.

- **Modo 4**: Colormap térmico JET que convierte la imagen a escala de grises y aplica una paleta de colores tipo "mapa de calor".

- **Modo 5**: Efecto de negativo mediante una inversión de colores.

- **Modo 6**: Efecto pixelado que reduce la resolución y la vuelve a escalar para crear un aspecto de "píxeles grandes".

El usuario puede cambiar entre modos presionando la **tecla espaciadora** y salir con ESC.

### Fragmento de código

```python
match nodo:
    case 1:
        # Aplica el filtro Sobel en ambas direcciones para detectar bordes sobre BGR
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)  # x
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)  # y
        output = cv2.add(sobelx, sobely)
        pass
    case 2:
        # Aplica un umbral alto para binarizar la imagen sobre BGR
        _, output = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
        pass
    case 3:
        # Aplica Canny para detección de bordes
        output = cv2.Canny(frame, 100, 200)
        pass
    case 4:
        # Aplica un colormap de tipo JET para un efecto térmico
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        pass
    case 5:
        # Aplica un efecto de negativo usando numpy
        output = 255 - frame
        pass
    case _:
        # Aplica un efecto pixelado
        h, w = frame.shape[:2]
        temp = cv2.resize(frame, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        pass
```

---

## Tarea 4 - Pincel por detección de color

En la última tarea se implementó un **sistema de pincel virtual por detección de color rojo** inspirado en el pincel de disney chanel.

El sistema detecta objetos rojos en tiempo real y permite dibujar sobre un lienzo virtual. Las funcionalidades incluyen:

- Detección de color rojo en espacio BGR.
- Cálculo del centro del contorno detectado.
- Dibujo persistente en un lienzo virtual.
- Opción de limpiar el lienzo presionando 'c'.

### Fragmento de código

```python
# Detección de rojo
mask = cv2.GaussianBlur(frame, (9,9), 0)
lower_red = np.array([0, 0, 100])    # B, G, R mínimos
upper_red = np.array([100, 100, 255]) # B, G, R máximos
mask = cv2.inRange(mask, lower_red, upper_red)

# Calcula el centroide usando el rectángulo delimitador
x, y, w, h = cv2.boundingRect(c)
cx = x + w // 2
cy = y + h // 2

# Dibuja en el lienzo y en el frame
cv2.circle(canvas, (cx, cy), 15, (0,0,255), -1)
cv2.circle(frame, (cx, cy), 15, (255,255,255), 2)
```

---

## Fuentes

- Documentación de OpenCV (https://docs.opencv.org/)
- Documentación de NumPy (https://numpy.org/doc/2.2/index.html)
- GitHub Copilot
- Vídeos de referencia para la Tarea 4:
  - [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy)
  - [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared)
  - [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared)
