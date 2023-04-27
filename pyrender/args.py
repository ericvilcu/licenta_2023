import argparse
import os
#TODO: organize arguments so finding/modifying them is easier
#os.environ["CUDA_LAUNCH_BLOCKING"]='1'
parser=argparse.ArgumentParser(prog="NeuralRendererPython")
#Required arg
parser.add_argument('--workspace',required=True,default="",help="Specifies the workspace's location. This is the only mandatory argument.")
#i don't know where to put this
parser.add_argument('--autosaves',required=False,default="600",help="Specifies what interval to autosave the nn in so that I stop losing time to shit crashing.")
parser.add_argument('--report_freq',required=False,default="60",help="Specifies what minimum interval of time to wait before displaying batch loss data.")
parser.add_argument('--reorder_points',required=False,default='no',help="Specifies how to reorder point data. See 'reorder.py' for available types")

#initialization,; ignored if w/o --make_workspace
parser.add_argument('--make_workspace',required=False,action="store_true",default=False,help="Specifies to create a workspace.")
parser.add_argument('-s','--scene',required=False, action='append', nargs=1,help="specifies a scene (ignored if without --make_workspace)")
parser.add_argument('--padding',default='',required=False,help="Overrides default padding that is applied to not mess up edges (default is the dataset 2**subplots) (ignored if without --make_workspace)")
parser.add_argument('--nn_type',default='1',required=False,help="Specifies what rendering network to use. see documentation (default=1)")
#nn type==1
parser.add_argument('--subplots',default='4',required=False,help="Overrides default number of subplots (default is 4) (ignored if without --make_workspace)")
parser.add_argument('--use_gates',default='False',required=False,help="Specifies wether to use gates or not (ignored if without --make_workspace)")
parser.add_argument('--extra_channels',default='0',required=False,help="Puts extra channels initially filled with gaussian noise onto points. (ignored if without --make_workspace)")
parser.add_argument('--base_nn',default='',required=False,help="Where to load a pre-initialized nn from. (ignored if without --make_workspace, all other nn args may be ignored, amd --extra_channels should be specified to be this nn's value)")
#parser.add_argument('--base_optim',default='',required=False,help="Where to load a pre-initialized optimizer from. (ignored if without --make_workspace, all other nn args may be ignored, amd --extra_channels should be specified to be this nn's value)")
parser.add_argument('--inv_depth',default='0',required=False,help="Whether to use 1/depth or depth itself")
parser.add_argument('--expand_environment',default='norm',required=False,help="What to use to pad environment if extra dimensions are needed (norm,zeros,ones,-1)")
parser.add_argument('--expand_points',default='norm',required=False,help="What to use to pad points if extra dimensions are needed (norm,zeros,ones,-1)")
#training
parser.add_argument('--notrain',action='store_true',default=False,required=False,help="Specifies to not train the neural network")
parser.add_argument('--batch_size',default='',required=False,help="Overrides default batch size(default is the dataset size)")
#not sure if I even want to train w/ those two.
parser.add_argument('--no_camera_refinement',default=False,required=False,action='store_true',help="Specifies to not improve the camera positions.")
parser.add_argument('--no_nn_refinement',default=False,required=False,action='store_true',help="Specifies to not improve the nn")
parser.add_argument('--no_point_refinement',default=False,required=False,action='store_true',help="Specifies to not improve point colors/positions")
parser.add_argument('--structural_refinement',default=False,required=False,action='store_true',help="Specifies to use structural refinement")
parser.add_argument('--loss_type',default='',help="Specifies which loss backward should be called for.")

#visualization
parser.add_argument('--norender',action='store_true',default=False,required=False,help="Specifies to ditch the main window entirely. note: currently breaks if you ctrl+c the app to stop, so make sure to specify timeout or max_batches")
parser.add_argument('-W','--width',default='',required=False,help="Specifies window width")
parser.add_argument('-H','--height',default='',required=False,help="Specifies window height")
parser.add_argument('--example_interval',default='-1',required=False,help="Specifies the interval to wait before showing a new example image. (default is 4.0 if training, 0.5 otherwise)")
#samples
parser.add_argument('--sample_prefix',default='',required=False,help="Specifies a string to put before file names.")
parser.add_argument('--sample_folder',default='.vscode/',required=False,help="Specifies where to save samples")
parser.add_argument('--samples_every',default='-1',required=False,help="Specifies the # of batches to save an image at. requires sample_folder")
parser.add_argument('--full_samples_final',default=False,required=False,action="store_true",help="Specifies the # of batches to save an image at. requires sample_folder")
parser.add_argument('--screencap_folder',default='.vscode/',required=False,help="Specifies where to save any screencap taken with 'p'")

#timeout/shutdown
#TODO: do something with lambdas to make timeout simpler.
parser.add_argument('--timeout',required=False,default='-1.0',help="Specifies how much time the program should automatically close in.")
parser.add_argument('--max_batches',required=False,default='-1',help="Specifies many batches the nn should train for before the application closes itself. Note: may sometimes train slightly more than the specified amount.")
parser.add_argument('--max_total_batches',required=False,default='-1',help="Specifies many batches the nn should train for (including batches from previous runs) before the application closes itself. Note: may sometimes train slightly more than the specified amount.")
parser.add_argument('--stagnation',required=False,default=('-1','-1'),nargs=2,help=r"Specifies the number of batches to look back on and average, as well as the % of change that is considered insignificant. For example, '--stagnation 10 0.01' means, naming the last 10 batches' average c and the average of the 10 batches before l, to stop when c*(1+0.01)>l")

raw_args=parser.parse_args()

nn_type=int(raw_args.nn_type)

base_nn:str = raw_args.base_nn
ndim=3+int(raw_args.extra_channels)
inv_depth=raw_args.inv_depth
scenes=[s[0] for s in raw_args.scene]
width=raw_args.width
height=raw_args.height
train=not raw_args.notrain
example_interval=float(raw_args.example_interval)
if(example_interval<0):example_interval=4.0 if train else 0.5
live_render=not raw_args.norender
make_workspace:bool = raw_args.make_workspace
workspace:str = raw_args.workspace
improve_cameras=True
main_loss:str=raw_args.loss_type
refine_points = not raw_args.no_point_refinement
nn_refinement = not raw_args.no_nn_refinement
camera_refinement = not raw_args.no_camera_refinement

report_freq=float(raw_args.report_freq)
autosave_s=float(raw_args.autosaves)
timeout=(float(raw_args.timeout)>0)
timeout_s=float(raw_args.timeout)
max_batches      =int(raw_args.max_batches      )
max_total_batches=int(raw_args.max_total_batches)
stagnation_batches,stagnation_p=int(raw_args.stagnation[0]),float(raw_args.stagnation[1])
reorder_points=raw_args.reorder_points

expand_environment=raw_args.expand_environment
expand_points=raw_args.expand_points
STRUCTURAL_REFINEMENT=raw_args.structural_refinement

nn_args={"ndim":ndim}

if(raw_args.batch_size!=""):
    nn_args["batch_size"]=int(raw_args.batch_size)
if(raw_args.padding!=""):
    nn_args["start_padding"]=int(raw_args.padding)
if(raw_args.subplots!=""):
    nn_args["subplots"]=int(raw_args.subplots)
if(raw_args.use_gates!=""):
    nn_args["use_gates"]= raw_args.use_gates.lower() == "true" or raw_args.use_gates=='1'

sample_folder = raw_args.sample_folder
samples_every = int(raw_args.samples_every)
full_samples_final = bool(raw_args.full_samples_final)
sample_prefix = raw_args.sample_prefix
screencap_folder=str(raw_args.screencap_folder)

if("PYSDL2_DLL_PATH" not in os.environ):
    raise Exception("\"PYSDL2_DLL_PATH\" is unset, SDL2 will not work. Please set \"PYSDL2_DLL_PATH\" either as a global variable or set it for this script")
