# signal-fourier-analysis
Análisis de señales en dominio del tiempo y frecuencia usando Transformada de Fourier con Python.
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Definición del tiempo
# ==============================
fs = 1000  # frecuencia de muestreo
t = np.linspace(-1, 1, fs)

# ==============================
# 2. Señales en el dominio del tiempo
# ==============================

# Señal senoidal
f = 5
senal_seno = np.sin(2 * np.pi * f * t)

# Pulso rectangular
pulso = np.where(np.abs(t) < 0.2, 1, 0)

# Señal escalón
escalon = np.where(t >= 0, 1, 0)

# ==============================
# 3. Transformada de Fourier
# ==============================

def calcular_fft(senal):
    N = len(senal)
    fft = np.fft.fft(senal)
    fft_shift = np.fft.fftshift(fft)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    return freq, fft_shift

freq_seno, fft_seno = calcular_fft(senal_seno)
freq_pulso, fft_pulso = calcular_fft(pulso)
freq_escalon, fft_escalon = calcular_fft(escalon)

# ==============================
# 4. Gráficas
# ==============================

def graficar(t, senal, freq, fft_senal, titulo):
    plt.figure(figsize=(12,5))

    # Dominio del tiempo
    plt.subplot(1,2,1)
    plt.plot(t, senal)
    plt.title(f"{titulo} - Tiempo")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")

    # Dominio de frecuencia
    plt.subplot(1,2,2)
    plt.plot(freq, np.abs(fft_senal))
    plt.title(f"{titulo} - Frecuencia")
    plt.xlabel("Frecuencia")
    plt.ylabel("Magnitud")

    plt.tight_layout()
    plt.show()

graficar(t, senal_seno, freq_seno, fft_seno, "Señal Senoidal")
graficar(t, pulso, freq_pulso, fft_pulso, "Pulso Rectangular")
graficar(t, escalon, freq_escalon, fft_escalon, "Escalón")

# ==============================
# 5. Verificación de Linealidad
# ==============================

senal_combinada = senal_seno + pulso
freq_comb, fft_comb = calcular_fft(senal_combinada)

plt.figure()
plt.plot(freq_comb, np.abs(fft_comb), label="FFT señal combinada")
plt.plot(freq_seno, np.abs(fft_seno) + np.abs(fft_pulso),
         linestyle='--', label="Suma de FFT individuales")
plt.legend()
plt.title("Verificación de Linealidad")
plt.show()

# ==============================
# 6. Desplazamiento en el tiempo
# ==============================

senal_desplazada = np.sin(2 * np.pi * f * (t - 0.2))
freq_des, fft_des = calcular_fft(senal_desplazada)

plt.figure()
plt.plot(freq_seno, np.abs(fft_seno), label="Original")
plt.plot(freq_des, np.abs(fft_des), linestyle='--', label="Desplazada")
plt.legend()
plt.title("Desplazamiento en el Tiempo")
plt.show()
