import argparse
import os
#TODO: use argparse (properly)

parser=argparse.ArgumentParser(prog="NeuralRendererPython")
parser.add_argument('--workspace',required=True,default="",help="Specifies the workspace's location. This is the only mandatory argument.")

parser.add_argument('--make_workspace',required=False,action="store_true",default=False,help="Specifies to create a workspace.")
parser.add_argument('--example_interval',default='-1',required=False)
parser.add_argument('--notrain',action='store_true',default=False,required=False,help="Specifies to not train the neural network")
parser.add_argument('--norender',action='store_true',default=False,required=False,help="Specifies to not show the interactive window network")
parser.add_argument('-s','--scene',required=False, action='append', nargs=1,help="specifies a scene (ignored if without --make_workspace)")
parser.add_argument('--batch_size',default='',required=False,help="Overrides default batch size(default is the dataset size) (ignored if without --make_workspace)")
parser.add_argument('--padding',default='',required=False,help="Overrides default padding that is applied to not mess up edges (default is the dataset 2**subplots) (ignored if without --make_workspace)")
parser.add_argument('--subplots',default='',required=False,help="Overrides default number of subplots (default is 4) (ignored if without --make_workspace)")
parser.add_argument('--use_gates',default='False',required=False,help="Specifies wether to use gates or not (ignored if without --make_workspace)")

raw_args=parser.parse_args()

ndim=3
scenes=[s[0] for s in raw_args.scene]
example_interval=float(raw_args.example_interval)
train=not raw_args.notrain
live_render=raw_args.norender
make_workspace = raw_args.make_workspace
workspace = raw_args.workspace

nn_args={}

if(raw_args.batch_size!=""):
    nn_args["batch_size"]=int(raw_args.batch_size)
if(raw_args.padding!=""):
    nn_args["start_padding"]=int(raw_args.padding)
if(raw_args.subplots!=""):
    nn_args["subplots"]=int(raw_args.subplots)
if(raw_args.use_gates!=""):
    nn_args["use_gates"]= raw_args.use_gates.lower() == "true" or raw_args.use_gates=='0'

if("PYSDL2_DLL_PATH" not in os.environ):
    raise Exception("\"PYSDL2_DLL_PATH\" is unset, SDL2 will not work. Please set \"PYSDL2_DLL_PATH\" either as a global  ")
