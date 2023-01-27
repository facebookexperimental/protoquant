import importlib.machinery
import os

import torch


def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin


def _load_library():
    if os.getenv("TORCHQUANT_IS_FBCODE", "1") == "0":
        path = _get_extension_path("_C")
        torch.ops.load_library(path)
    else:
        torch.ops.load_library("//protoquant:ops")
