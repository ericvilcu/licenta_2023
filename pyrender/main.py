from typing import Callable
import args



print("importing torch")
import torch
print("torch imported")
import trainer

from GLRenderer import Renderer
from time import time,sleep
from dataSet import DataSet
import kernelItf
import cameraController
import sdl2

def make_workspace():
    print(f"loading dataset from scenes={set(args.scenes)}")
    ds = DataSet(scenes=args.scenes,force_ndim=args.ndim)
    if(args.base_nn==''):
        t=trainer.trainer(ds,**args.nn_args)
    else:
        t=trainer.trainer(ds,nn=torch.load(args.base_nn))
    print(f"saving workspace to {args.workspace}")
    t.save(args.workspace)
    


if(args.make_workspace):
    make_workspace()

print("loading workspace...")
t=trainer.trainer.load(args.workspace,args=args.nn_args)
print("workspace loaded...")
ds = t.data

if(args.live_render):
    r = Renderer(**{'w':int(args.width),'h':int(args.height)})
    r.removeView('main')
    r.createView('interactive',0.,0.,.5,.5,0,True)
    r.createView('render'     ,.5,0.,1.,.5,0,True)
    r.createView('points'     ,0.,.5,.5,1.,0,True)
    r.createView('target'     ,.5,.5,1.,1.,0,True)

#ds = DataSet(scenes=args.scenes)
s=e=time()

last_test_result=time()-args.example_interval
controller=cameraController.CameraController()
ev=[]

def stack_fun(fun:list[Callable[[],bool]]) -> Callable[[],bool]:
    if len(fun)==0:
        return lambda:False
    if len(fun)==1:
        return fun[0]
    nxt:Callable[[],bool]=stack_fun(fun[1:])
    f=fun[0]
    return lambda:f() or nxt()
    

should_close=[]
if(args.timeout):
    should_close.append(lambda:e-s>=args.timeout_s)
if(args.max_batches>=0):
    should_close.append(lambda:trainer.TOTAL_BATCHES_THIS_RUN>=args.max_batches)
if(args.stagnation_batches!=-1):
    should_close.append(lambda:trainer.is_stagnant())
should_close=stack_fun(should_close)

kernelItf.initialize()

if(args.train):
    t.start_trainer_thread()
try:
    if(args.live_render):
        while(not should_close()):#(not args.timeout or e-s<args.timeout_s) and (args.max_batches<0 or trainer.TOTAL_BATCHES_THIS_RUN<=args.max_batches)):
            if(r.is_window_minimized()):
                r.sleep_until_not_minimized(60)
                ev=r.update()
            else:
                ne=time()
                delta=ne-e
                e=ne
                
                controller.process_sdl_events(ev,delta)
                
                if(controller.take_screencap):
                    r.screencapViews(args.screencap_folder)
                    controller.take_screencap=False
                    
                
                if(e-last_test_result>args.example_interval):
                    last_test_result=e
                    t.display_results_to_renderer(r,'points','render','target')
                #interactive view.
                
                cam=controller.get_camera()
                if(controller.use_neural):
                    view = t.size_safe_forward_nograd(controller.camera_type(),cam,0)
                else:
                    view = controller.only_shown_dimensions(*t.draw_subplots(0,controller.camera_type(),cam,1))
                r.upload_tensor('interactive',view)
                ev=r.update()
                
                if(len([e1 for e1 in ev if(e1.type == sdl2.events.SDL_QUIT)])>0):
                    print("Quitting manually...")
                    break
    else:
        while(not should_close()):#(not args.timeout or e-s<args.timeout_s) and (args.max_batches<0 or trainer.TOTAL_BATCHES_THIS_RUN<=args.max_batches)):
            sleep(60)#bad but i'm not sure how to fix it yet.
except KeyboardInterrupt as ex:
    print(ex)
    print("KeyboardInterrupt detected, stopping and saving...")
except Exception as ex:
    print(repr(ex),ex.__traceback__)
    args.full_samples_final=False#If it crashed, we probably shouldn't
    raise ex
finally:
    
    if(args.live_render):
        print("Closing window...")
        r.shutdown()
    
    if(args.train):
        print("Waiting for last batch to be trained...")
        t.stop_trainer_thread()

    print("Saving...")
    t.save(args.workspace)

    if(args.full_samples_final):
        print("Saving all samples (this may take a while)")
        t.save_all_samples()
    print("Cleanup...")
    kernelItf.cleanup()