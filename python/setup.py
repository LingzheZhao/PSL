from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path

__version__ = '0.0.1'
script_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules=[
    CUDAExtension(
        name='pypsl_lib',
        sources=[
            'pypsl_lib/cps.cpp',
            'pypsl_lib/python_bind.cpp',
        ],
        extra_compile_args={
            'cxx': ['-O2', '-std=c++20', '-DCOMPILE_WITH_CUDA', '-fPIC'],
            'nvcc': ['-O2', '-std=c++20', '-w', '-Xcompiler', '-fPIC'],
        },
        extra_link_args=[
            '-L/usr/local/lib',
            '-lpsl',
            '-lboost_system',
            '-lboost_program_options',
        ],
        include_dirs=[
            "/usr/include/eigen3/",
            "/usr/include/opencv4/",
            "/usr/local/include/opencv4/",
        ],
    ),
]

INSTALL_REQUIREMENTS = ['torch', 'torchvision', 'eigen']

if __name__ == "__main__":
    setup(
        description='Python bindings of PlaneSweepingLib',
        author='Lingzhe Zhao',
        author_email='zhaolingzhe@westlake.edu.cn',
        license='GPLv3',
        version=__version__,
        name='pypsl',
        packages=['pypsl', 'pypsl_lib'],
        install_requires=INSTALL_REQUIREMENTS,
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
    )
