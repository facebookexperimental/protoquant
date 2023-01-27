import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.0.1"

ext_modules = []
cutlass_path = f"{rootdir}/third_party/cutlass/include"

sources = glob.glob("protoquant/src/**/*.cu", recursive=True)
sources += glob.glob("protoquant/src/**/*.cpp", recursive=True)

ext = CUDAExtension(
    "protoquant._C",
    sorted(sources),
    extra_compile_args=["-I", rootdir, "-I", cutlass_path],
)
ext_modules = [ext]
packages = find_packages(exclude=("test",))
package_name = "protoquant"

setup(
    name=package_name,
    version=version,
    description="",
    long_description="",
    classifiers=[],
    packages=packages,
    package_data={package_name: ["*.so"]},
    install_requires=["triton"],
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
