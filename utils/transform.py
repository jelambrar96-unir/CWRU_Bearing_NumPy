import random

import numpy as np


def extract_sequences(
        time_series: np.array,
        number_rows : int,
        number_columns : int,
        random_seed : int = None
        ) -> np.array:
    """
    Extrae subsecuencias aleatorias de una serie temporal y las organiza en una matriz.

    Esta función toma un arreglo unidimensional (serie temporal) y genera un número
    especificado de subsecuencias contiguas de longitud fija. Las subsecuencias se
    seleccionan de manera aleatoria y se devuelven en una matriz 2D.

    Parámetros
    ----------
    time_series : np.array
        Serie temporal de entrada en forma de arreglo unidimensional.
    number_rows : int
        Número de subsecuencias a extraer (filas de la matriz de salida).
    number_columns : int
        Longitud de cada subsecuencia (columnas de la matriz de salida).
    random_seed : int, opcional
        Semilla para el generador de números aleatorios, lo que permite reproducibilidad.

    Retorna
    -------
    np.array
        Arreglo bidimensional de forma `(number_rows, number_columns)` que contiene
        las subsecuencias extraídas.

    Excepciones
    -----------
    ValueError
        Si `number_rows` es mayor que el número máximo de subsecuencias posibles
        dado el tamaño de la serie temporal.

    Ejemplos
    --------
    >>> import numpy as np
    >>> serie = np.arange(10)  # Serie: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> extract_sequences(serie, number_rows=3, number_columns=4, random_seed=42)
    array([[2, 3, 4, 5],
           [1, 2, 3, 4],
           [4, 5, 6, 7]])
    """

    time_series_length = time_series.shape[0]
    max_number_of_rows = time_series_length - number_columns + 1

    if number_rows > max_number_of_rows:
        raise ValueError("Input series too short to extract that many subsequences.")

    random.seed(random_seed)
    choices = random.choices(range(max_number_of_rows), k=number_rows)
    random.shuffle(choices)

    output = np.empty((number_rows,number_columns), dtype=time_series.dtype)

    for i, item in enumerate(choices):
        output[i,:] = time_series[item : item + number_columns].reshape((1, number_columns))
    
    return output



