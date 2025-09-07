import numpy as np


def open_npz(file_path):
    """
    Open an NPZ file and return the loaded data.
    
    Args:
        file_path (str): The path to the NPZ file.
    
    Returns:
        dict: A dictionary containing the data stored in the NPZ file.
    """
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")
        return None


def open_npz_key(file_path, key):
    """
    Open an NPZ file and return the loaded data.
    
    Args:
        file_path (str): The path to the NPZ file.
    
    Returns:
        dict: A dictionary containing the data stored in the NPZ file.
    """
    try:
        data = np.load(file_path)[key]
        return data
    except KeyError:
        print(f"Error: The file '{file_path}' does not contains key '{key}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")
        return None


def save_npz(output_path: str, **arrays):
    """
    Guarda múltiples arrays en un archivo .npz.

    Parámetros
    ----------
    output_path : str
        Ruta donde se guardará el archivo .npz.
    **arrays : dict
        Diccionario de arrays que se guardarán en el archivo.
        Cada clave será el nombre con el que se almacenará el array.

    Ejemplo
    -------
    >>> import numpy as np
    >>> a = np.arange(5)
    >>> b = np.random.randn(3, 3)
    >>> save_npz("mis_datos.npz", serie=a, matriz=b)
    """
    np.savez(output_path, **arrays)
    # print(f"Datos guardados en: {output_path}")

