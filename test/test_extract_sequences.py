import unittest
import numpy as np
from utils.transform import extract_sequences  # cámbialo al nombre real de tu archivo .py


class TestExtractSequences(unittest.TestCase):

    def setUp(self):
        """Configura datos comunes para las pruebas."""
        self.serie = np.arange(10)  # Serie de ejemplo: [0,1,2,3,4,5,6,7,8,9]

    def test_shape_output(self):
        """Verifica que el tamaño de salida sea correcto."""
        result = extract_sequences(self.serie, number_rows=3, number_columns=4, random_seed=0)
        self.assertEqual(result.shape, (3, 4))

    def test_reproducibility_with_seed(self):
        """Verifica que el resultado sea reproducible con la misma semilla."""
        result1 = extract_sequences(self.serie, number_rows=2, number_columns=3, random_seed=42)
        result2 = extract_sequences(self.serie, number_rows=2, number_columns=3, random_seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds(self):
        """Verifica que con semillas diferentes, la salida no siempre sea igual."""
        result1 = extract_sequences(self.serie, number_rows=2, number_columns=3, random_seed=1)
        result2 = extract_sequences(self.serie, number_rows=2, number_columns=3, random_seed=99)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(result1, result2)

    def test_value_error_if_too_many_rows(self):
        """Verifica que se lance ValueError si se piden demasiadas subsecuencias."""
        with self.assertRaises(ValueError):
            extract_sequences(self.serie, number_rows=20, number_columns=5)

    def test_dtype_preservation(self):
        """Verifica que el tipo de dato del array de salida se preserve."""
        serie_float = np.linspace(0, 1, 10, dtype=np.float32)
        result = extract_sequences(serie_float, number_rows=3, number_columns=4, random_seed=0)
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
