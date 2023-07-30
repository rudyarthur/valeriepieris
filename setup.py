from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("valeriepieris", ["src/valeriepieris.pyx"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
]
setup(
   ext_modules = cythonize(extensions, annotate=True, 
   compiler_directives={'language_level' : "3", "cdivision":True, "boundscheck":False, "wraparound":False, "nonecheck":False}
   ) 
)
