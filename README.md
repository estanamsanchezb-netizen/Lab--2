# Lab--2
En esta práctica de laboratorio se estudiaron tres herramientas fundamentales del procesamiento digital de señales: la convolución, la correlación y la transformada de Fourier. Inicialmente, se realizó el cálculo de la convolución entre dos secuencias discretas definidas a partir de los dígitos del código y la cédula de los integrantes, desarrollándolo tanto de forma manual como mediante Python para analizar la respuesta de un sistema ante una señal de entrada. Posteriormente, se analizó la correlación cruzada entre dos señales senoidales con el fin de evaluar su grado de similitud. Finalmente, se generó una señal biológica, la cual fue digitalizada considerando el criterio de Nyquist y posteriormente caracterizada mediante parámetros estadísticos. A esta señal se le aplicó la transformada de Fourier para estudiar su comportamiento en el dominio de la frecuencia y observar su contenido espectral.
## Parte A
En la primera parte de la práctica se trabajó la operación de convolución entre un sistema h[n] y una señal de entrada  x[n], los cuales fueron definidos a partir de la unión de los dígitos del código y la cédula de los integrantes. A partir de estas dos secuencias se obtuvo la señal de salida y[n], que corresponde al resultado de la convolución entre x[n] y h[n]. Inicialmente, el cálculo se realizó de forma manual con el objetivo de comprender el procedimiento paso a paso y determinar la secuencia resultante. Posteriormente, el mismo proceso se implementó en Python, lo que permitió verificar los resultados obtenidos y generar la representación gráfica de las señales.

**-Convolución  y grafica manual**

<img width="517" height="727" alt="image" src="https://github.com/user-attachments/assets/71a154f4-303f-489b-a994-e12b7ddb9443" />

<br>
<img width="481" height="130" alt="image" src="https://github.com/user-attachments/assets/b90bfdbb-dc02-460d-a23c-df8bdd76222c" />

<br>
<img width="535" height="821" alt="image" src="https://github.com/user-attachments/assets/e4d2da6a-933e-492b-bfda-ab05b180f504" />

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

En esta parte de la práctica se definieron dos señales discretas y se calculó su correlación cruzada con el objetivo de analizar la similitud entre ellas. Posteriormente, se procedió a graficar la señal de correlación cruzada tanto normalizada como no normalizada, con el fin de observar su comportamiento.

<img width="446" height="84" alt="image" src="https://github.com/user-attachments/assets/c859626e-e873-4110-b429-75a5a84d563c" />

1. Correlación cruzada entre ambas señales
```python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Datos del enunciado
# -----------------------------
Ts = 1.25e-3          # 1.25 ms
f  = 100.0            # 100 Hz
N  = 9
n  = np.arange(0, N)  # 0 <= n < 9  ->  n = 0,1,...,8
t  = n * Ts

# Señales
x1 = np.cos(2*np.pi*f*t)
x2 = np.sin(2*np.pi*f*t)

# -----------------------------
# 1) Correlación cruzada
# Definición usada:
# r12[k] = sum_n x1[n] * x2[n-k]
# (np.correlate(x1,x2,'full') entrega exactamente esa correlación para señales reales)
# -----------------------------
r12  = np.correlate(x1, x2, mode="full")
lags = np.arange(-(N-1), (N-1)+1)   # k = -8,...,0,...,+8

# normalizada, [-1,1]
r12_norm = r12 / (np.linalg.norm(x1)*np.linalg.norm(x2))

# Mostrar valores
print("x1[n] =", np.round(x1, 6))
print("x2[n] =", np.round(x2, 6))

print("\n--- r12[k] (no normalizada) ---")
for k, val in zip(lags, r12):
    print(f"k={k:>2d}  r12={val: .6f}")

print("\n--- r12[k] (normalizada) ---")
for k, val in zip(lags, r12_norm):
    print(f"k={k:>2d}  r12_norm={val: .6f}")


# -----------------------------
# 2) Representación gráfica 
# -----------------------------

# Colores rosados
pink_line  = "#FF4D8D"
pink_fill  = "#FF8FB8"

# 1) No normalizada
plt.figure(figsize=(10,4))
m1, s1, b1 = plt.stem(lags, r12, basefmt=" ")
plt.setp(m1, color=pink_line, markersize=6)
plt.setp(s1, color=pink_fill, linewidth=2)
plt.xlabel("Lag k (muestras)")
plt.ylabel(r"$r_{x_1x_2}[k]$")
plt.title("Correlación cruzada (no normalizada)")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# 2) Normalizada
plt.figure(figsize=(10,4))
m2, s2, b2 = plt.stem(lags, r12_norm, basefmt=" ")
plt.setp(m2, color=pink_line, markersize=6)
plt.setp(s2, color=pink_fill, linewidth=2)
plt.xlabel("Lag k (muestras)")
plt.ylabel(r"$r_{x_1x_2}[k]$ (normalizada)")
plt.title("Correlación cruzada (normalizada)")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()
```
<img width="892" height="414" alt="image" src="https://github.com/user-attachments/assets/c3edec83-28a7-48a4-9ed9-cc18e9b7ceaa" />
<br>
<img width="325" height="327" alt="image" src="https://github.com/user-attachments/assets/ba16b27d-e89f-4588-9bf7-7552abc8ce37" />
<br>
2. Grafica de señales normalizada y no normalizada

<img width="878" height="651" alt="image" src="https://github.com/user-attachments/assets/609f4079-52c7-4535-afd3-a53b730a72a0" />
<br>
3. ¿En qué situaciones resulta útil aplicar la correlación cruzada en el procesamiento digital de señales?

## Parte C 
