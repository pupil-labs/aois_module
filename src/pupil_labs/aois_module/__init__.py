"""Top-level entry-point for the aois_module package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

import struct

if struct.calcsize("P") * 8 != 64:
    raise Exception("Sorry, this script only works on 64 bit systems!")

try:
    __version__ = version("pupil_labs.aois_module")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
