# Lab--2
En esta práctica de laboratorio se estudiaron tres herramientas fundamentales del procesamiento digital de señales: la convolución, la correlación y la transformada de Fourier. Inicialmente, se realizó el cálculo de la convolución entre dos secuencias discretas definidas a partir de los dígitos del código y la cédula de los integrantes, desarrollándolo tanto de forma manual como mediante Python para analizar la respuesta de un sistema ante una señal de entrada. Posteriormente, se analizó la correlación cruzada entre dos señales senoidales con el fin de evaluar su grado de similitud. Finalmente, se generó una señal biológica, la cual fue digitalizada considerando el criterio de Nyquist y posteriormente caracterizada mediante parámetros estadísticos. A esta señal se le aplicó la transformada de Fourier para estudiar su comportamiento en el dominio de la frecuencia y observar su contenido espectral.
## Parte A
En la primera parte de la práctica se trabajó la operación de convolución entre un sistema h[n] y una señal de entrada  x[n], los cuales fueron definidos a partir de la unión de los dígitos del código y la cédula de los integrantes. A partir de estas dos secuencias se obtuvo la señal de salida y[n], que corresponde al resultado de la convolución entre x[n] y h[n]. Inicialmente, el cálculo se realizó de forma manual con el objetivo de comprender el procedimiento paso a paso y determinar la secuencia resultante. Posteriormente, el mismo proceso se implementó en Python, lo que permitió verificar los resultados obtenidos y generar la representación gráfica de las señales.

**-Convolución  y grafica manual**

**-Convolución y gráfica en python**

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Señales de entrada
# -------------------------------
h = [5, 6, 0, 0, 8, 7, 0, 5, 6, 0, 0, 6, 0, 5]  # Código
x = [1,0,7,2,6,4,3,3,6,5,1,0,7,6,2,4,1,5,9,8]  # Cédula

# Convolución
y = np.convolve(x, h)

# Mostrar valores secuenciales

print("Valores de y[n]:")
for i, val in enumerate(y):
    print(f"y[{i}] = {val}")

# -------------------------------
# Graficar señal
# -------------------------------
n = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
plt.stem(n,y)
# Índices de tiempo
plt.title("Señal y[n] = x[n] * h[n]")
markerline, stemlines, baseline = plt.stem(n, y)
plt.setp(stemlines, color='pink')
plt.setp(markerline, color='pink')
plt.xlabel("n (índice)")
plt.ylabel("y[n]")
plt.grid(True)
plt.tight_layout()
plt.show()
```
<img width="158" height="599" alt="image" src="https://github.com/user-attachments/assets/88f91450-a417-48e7-9c98-a37e45f14303" />
<br>

<img width="705" height="474" alt="image" src="https://github.com/user-attachments/assets/20c7e629-9331-49b2-a763-8c115ed1d374" />
<br>
<img width="217" height="513" alt="image" src="https://github.com/user-attachments/assets/c1bd2083-9b68-458c-bd2a-3db0c3d20e2a" />
 Imagen 3.  Diagrama de flujo parte A

## Parte B 
## Parte C 
