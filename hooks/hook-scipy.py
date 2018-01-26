import os
import scipy
import platform

binaries = []

if "windows" in platform.platform().lower():
    dll_path = os.path.join(os.path.dirname(scipy.__file__), 'extra-dll', "*.dll")
    binaries = [(dll_path, "dlls")]

hiddenimports = ['scipy._lib.%s' % m for m in ['messagestream', "_ccallback_c", "_fpumode"]]
