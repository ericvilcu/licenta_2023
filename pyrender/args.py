import argparse
import os
#TODO: use argparse (properly)

parser=argparse.ArgumentParser(prog="NeuralRendererPython")
parser.add_argument('--example_interval',default='-1',required=False)
parser.add_argument('--notrain',action='store_true',default=False,required=False)
parser.add_argument('-s','--scene',required=False, action='append', nargs='+')


raw_args=parser.parse_args()

ndim=3
scenes=[s[0] for s in raw_args.scene]
example_interval=float(raw_args.example_interval)
train=not raw_args.notrain




if("PYSDL2_DLL_PATH" not in os.environ):
    raise Exception("\"PYSDL2_DLL_PATH\" is unset, SDL2 will not work. Please set \"PYSDL2_DLL_PATH\" either as a global  ")
