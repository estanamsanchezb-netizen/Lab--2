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
 En esta parte se realizo la captura de una señal EOG (electrooculograma) del generador de señales con ayuda del DAQ. se uso el siguiente codigo para su captura 
 ```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nidaqmx
from nidaqmx.constants import AcquisitionType
from threading import Thread, Event
from collections import deque
import datetime
import time
from scipy import signal  

fs = 400        
canal = "Dev6/ai0"    
tamano_bloque = int(fs * 0.05)   
ventana_tiempo = 5.0             

lowcut = 0.1
highcut = 40
orden = 4

sos = signal.butter(orden, [lowcut/(fs/2), highcut/(fs/2)], 
                    btype='bandpass', output='sos')
zi = signal.sosfilt_zi(sos)  

buffer_graf = deque(maxlen=int(fs * ventana_tiempo))
datos_guardados = []

adquiriendo = Event()
detener_hilo = Event()
thread_lectura = None

def hilo_lectura():
    global datos_guardados, buffer_graf, zi
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(canal)
    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print(f"\n▶️ Adquisición iniciada en {canal} ({fs} Hz).")

    while not detener_hilo.is_set():
        if adquiriendo.is_set():
            try:
                datos = np.array(task.read(number_of_samples_per_channel=tamano_bloque))
            
                datos_filtrados, zi = signal.sosfilt(sos, datos, zi=zi)
                
                buffer_graf.extend(datos_filtrados)
                datos_guardados.extend(datos_filtrados)

            except Exception as e:
                print("⚠ Error de lectura:", e)
                break
        else:
            time.sleep(0.05)

    task.stop()
    task.close()
    print("Adquisición detenida")

def iniciar(event):
    global thread_lectura
    if not adquiriendo.is_set():
        if thread_lectura is None or not thread_lectura.is_alive():
            detener_hilo.clear()
            thread_lectura = Thread(target=hilo_lectura, daemon=True)
            thread_lectura.start()
        adquiriendo.set()
        print("▶️ Grabando...")

def detener(event):
    """Detiene y guarda los datos."""
    adquiriendo.clear()
    detener_hilo.set()
    time.sleep(0.3)

    if datos_guardados:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"ECG_filtrado__ahorasi_ultimaaaaa{timestamp}.txt"
        tiempos = np.arange(len(datos_guardados)) / fs
        data = np.column_stack((tiempos, datos_guardados))
        np.savetxt(nombre_archivo, data, fmt="%.6f", header="Tiempo(s)\tVoltaje(V)")
        print(f"Señal guardada en {nombre_archivo} ({len(datos_guardados)} muestras)")
    else:
        print("No se capturaron datos")

fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
linea, = ax.plot([], [], lw=1.2, color='royalblue')
ax.set_xlim(0, ventana_tiempo)
ax.set_ylim(-3, 3)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Voltaje [V]")
ax.set_title("ECG filtrado (IIR tiempo real)")
ax.grid(True, linestyle="--", alpha=0.6)

x = np.linspace(0, ventana_tiempo, int(fs * ventana_tiempo))
y = np.zeros_like(x)

def actualizar(frame):
    if len(buffer_graf) > 0:
        y = np.array(buffer_graf)
        if len(y) < len(x):
            y = np.pad(y, (len(x)-len(y), 0), constant_values=0)
        linea.set_data(x, y)
    return linea,

ax_iniciar = plt.axes([0.3, 0.1, 0.15, 0.075])
ax_detener = plt.axes([0.55, 0.1, 0.2, 0.075])
btn_iniciar = Button(ax_iniciar, 'Iniciar', color='purple', hovercolor='purple')
btn_detener = Button(ax_detener, 'Detener y Guardar', color='pink', hovercolor='pink')
btn_iniciar.on_clicked(iniciar)
btn_detener.on_clicked(detener)

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, actualizar, interval=50, blit=True)
plt.tight_layout()
plt.show()
```
Para esta señal, se determinó la frecuencia de Nyquist. Se utilizó la frecuencia máxima de la señal EOG que es de aproximadamente de 50 Hz, esta se multiplicó por 2 para obtener la frecuencia de Nyquist (50 Hz* 2 = 100 Hz).  La frecuencia de muestreo es 4 veces la de Nyquist (400Hz).
<img width="992" height="359" alt="image" src="https://github.com/user-attachments/assets/3d2b3e65-99db-4022-bff9-3514ef209f61" />

 Siguiendo, se hallo Media, mediana, desviación estándar, máximo, mínimo, la  Transformada de Fourier, densidad espectral de potencia.
```python
import numpy as np
import matplotlib.pyplot as plt


filepath = "/content/ECG_filtrado__ahorasi_ultimaaaaa20260227_144114.txt"

# El archivo tiene encabezado tipo: "# Tiempo(s)\tVoltaje(V)"
data = np.loadtxt(filepath, comments="#")  # ignora líneas que empiezan con "#"
t = data[:, 0]
x = data[:, 1]

# Frecuencia de muestreo (se infiere del paso temporal)
dt = np.median(np.diff(t))
fs = 1.0 / dt



# =========================
# 1) Caracterización (a)
# =========================
media   = np.mean(x)
mediana = np.median(x)
std     = np.std(x, ddof=1)   # desviación estándar muestral
xmin    = np.min(x)
xmax    = np.max(x)

print("\n--- Caracterización estadística ---")
print(f"Media:               {media:.6f} V")
print(f"Mediana:             {mediana:.6f} V")
print(f"Desviación estándar: {std:.6f} V")
print(f"Mínimo:              {xmin:.6f} V")
print(f"Máximo:              {xmax:.6f} V")

# Señal en el tiempo (CYAN)
plt.figure(figsize=(10,4))
plt.plot(t, x, color="#00C2D1", linewidth=2)   # cyan
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal en el dominio del tiempo")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# =========================
# 2) FFT y espectro (a)
# =========================
N = len(x)

# Quitar DC para que el pico en 0 Hz no domine
x0 = x - np.mean(x)

# Ventana (reduce leakage). Si no la quieres, comenta estas 2 líneas.
w = np.hanning(N)
xw = x0 * w

# FFT de un solo lado (rfft) y su eje de frecuencias
X = np.fft.rfft(xw)
f = np.fft.rfftfreq(N, d=dt)

# Magnitud (normalización simple para visualizar)
mag = np.abs(X) / N

# ===== 1) FFT magnitud (MORADO) =====
plt.figure(figsize=(10,4))
plt.plot(f, mag, color="#7B2CBF", linewidth=2.2)   # morado
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|X(f)| (magnitud)")
plt.title("Transformada de Fourier (magnitud, 1 lado)")
plt.grid(True, alpha=0.25)
plt.xlim(0, fs/2)
plt.tight_layout()
plt.show()

# ===== 2) PSD (VERDE) =====
plt.figure(figsize=(10,4))
plt.plot(f, PSD, color="#2A9D8F", linewidth=2.2)   # verde
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.title("Densidad espectral de potencia (PSD)")
plt.grid(True, alpha=0.25)
plt.xlim(0, fs/2)
plt.tight_layout()
plt.show()

```
<img width="291" height="98" alt="image" src="https://github.com/user-attachments/assets/daafe437-0220-4884-b84e-c84a748febca" />
<br>
La señal EOG puede clasificarse como aleatoria, ya que depende de los movimientos oculares y de diferentes fuentes de ruido fisiológico. Aunque algunos movimientos oculares repetidos pueden generar patrones similares, la señal es de origen biológico y puede variar debido a factores como la respuesta individual del paciente o condiciones fisiológicas del momento. Además, se considera aperiódica, porque los movimientos oculares no ocurren en intervalos regulares ni siguen un ciclo repetitivo constante. Finalmente, la señal es originalmente analógica, debido a que proviene de un fenómeno fisiológico continuo en el tiempo; sin embargo, al ser adquirida mediante un sistema de adquisición de datos (DAQ) y posteriormente muestreada para su procesamiento, pasa a representarse en forma digital para su análisis.

<img width="873" height="651" alt="image" src="https://github.com/user-attachments/assets/fc043f37-ffb9-4f0c-b5d2-f505e66e2640" />

