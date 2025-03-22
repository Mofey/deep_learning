import os
import sys
import subprocess

REQUIRED_LIBS_DL = ['tensorflow', 'numpy']
for lib in REQUIRED_LIBS_DL:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])