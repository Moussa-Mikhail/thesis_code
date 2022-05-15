# pylint: disable=missing-docstring
from setuptools import Extension, setup  # type: ignore
from Cython.Build import cythonize  # type: ignore

ext_modules = [
    Extension(
        "integrate_cy",
        ["integrate_cy.pyx"],
    ),
    Extension(
        "transform_cy",
        ["transform_cy.pyx"],
    ),
]

setup(
    ext_modules=cythonize(ext_modules, language_level=3),
)
