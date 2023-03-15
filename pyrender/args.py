#TODO: use argparse
import argparse
import os
import sys
ndim=3
scenes=[sys.argv[1]]
if("PYSDL2_DLL_PATH" not in os.environ):
    raise Exception("\"PYSDL2_DLL_PATH\" is unset, SDL2 will not work.")
example_interval=2
train=True




