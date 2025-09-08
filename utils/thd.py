import numpy as np


def compute_thd(signal):
    """
    Calcula el THD (Total Harmonic Distortion) de una señal.

    Parámetros:
      - signal: array 1D con valores de la señal en el dominio tiempo (amplitudes).

    Retorna:
      - thd_f: el valor de THD_F (fracción, no porcentaje).
      - thd_percent: THD expresado en porcentaje.
    """
    N = len(signal)
    # Transformada de Fourier
    Y = np.fft.rfft(signal)
    # Magnitudes normalizadas (convertimos a amplitud real)
    Y_mag = np.abs(Y) / N * 2  # factor 2 por considerar mitad del espectro

    # Índice de la frecuencia fundamental (asumimos que el primer pico es la fundamental)
    fundamental_idx = np.argmax(Y_mag[1:]) + 1  # excluimos DC

    V1 = Y_mag[fundamental_idx]

    # Harmónicos: todos los otros picos distintos del fundamental y DC
    harmonics = np.delete(Y_mag, [0, fundamental_idx])

    # Cálculo de THD_F
    thd_f = np.sqrt(np.sum(harmonics**2)) / V1

    return thd_f, fundamental_idx


def compute_thd_npdata(signal2d):
    """
    Calcula el THD (Total Harmonic Distortion) de una señal.

    Parámetros:
      - signal: array 1D con valores de la señal en el dominio tiempo (amplitudes).

    Retorna:
      - thd_f: el valor de THD_F (fracción, no porcentaje).
      - thd_percent: THD expresado en porcentaje.
    """
    N = signal2d.shape[1]
    Y = np.fft.rfft(signal2d, axis=1)
    Y_mag = np.abs(Y) / N * 2
    funadmental_idx = np.argmax(Y_mag[:,1:], axis=1) + 1
    V1 = np.max(Y_mag, axis=1)
    thd_f = np.sqrt(np.sum(np.square(Y_mag), axis=1) - np.square(V1)) / V1
    return thd_f, funadmental_idx


# Ejemplo de uso:
if __name__ == "__main__":
    fs = 1000  # Hz
    t = np.linspace(0, 1, fs, endpoint=False)
    # señal con armonicas
    signal = np.sin(2*np.pi*50*t) + 0.1*np.sin(2*np.pi*150*t) + 0.05*np.sin(2*np.pi*250*t)
    thd_f, idx = compute_thd(signal)
    print(f"THD_F = {thd_f:.4f}, (frecuencia fundamental índice {idx})")


    signal_2d = np.empty((3, len(signal)))
    signal_2d[0, :] = signal
    signal_2d[1, :] = signal
    signal_2d[2, :] = signal

    thd_f, idx = compute_thd_npdata(signal_2d)
    print(thd_f)
    print(idx)