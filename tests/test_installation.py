import unittest

class TestPackageInstallation(unittest.TestCase):
    def test_numpy_import(self):
        import numpy
        self.assertIsNotNone(numpy)

    def test_pandas_import(self):
        import pandas
        self.assertIsNotNone(pandas)

    def test_numpy_version(self):
        import numpy
        self.assertEqual(numpy.__version__, '1.21.0')

    def test_pandas_version(self):
        import pandas
        self.assertEqual(pandas.__version__, '1.3.0')

    def test_pydantic_import(self):
        import pydantic
        self.assertIsNotNone(pydantic)

    def test_typer_import(self):
        import typer
        self.assertIsNotNone(typer)

    def test_jinja2_import(self):
        import jinja2
        self.assertIsNotNone(jinja2)

    def test_halo_import(self):
        import halo
        self.assertIsNotNone(halo)

if __name__ == '__main__':
    unittest.main()