from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='MCores',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("MCores",
                 sources=["kernelModesCluster.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='Heinrich Jiang',
    author_email='heinrich.jiang@gmail.com'

)
