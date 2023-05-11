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
import os

def make_workspace():
    print(f"loading dataset from scenes={set(args.scenes)}")
    
    if(os.path.exists(args.workspace)):
        print("WARN: WORKSPACE MAY BE OVERRIDDEN; THIS MAY CAUSE ERRORS. YOU CAN DELETE IT MANUALLY TO FIX MOST OF THEM.")
    
    ds = DataSet(scenes=args.scenes,force_ndim=args.ndim)
    if(args.base_nn==''):
        t=trainer.trainer(ds,**args.nn_args)
    else:
        t=trainer.trainer(ds,nn=torch.load(args.base_nn))
    print(f"saving workspace to {args.workspace}")
    t.save(args.workspace)
    


if(args.make_workspace):
    make_workspace()
elif(not os.path.exists(args.workspace)):
    print("WARNING: WORKSPACE DOES NOT EXIST AND WILL BE CREATED")
    make_workspace()

print("loading workspace...")
t=trainer.trainer.load(args.workspace,args=args.nn_args)
print("workspace loaded...")
ds = t.data

if(args.live_render):
    r = Renderer(**{'w':int(args.width),'h':int(args.height)},window_name=('Visualization' if not args.train else 'Visualization (training in background)'))
    r.removeView('main')
    r.createView('interactive',0.,0.,.5,.5,0,True)
    r.createView('render'     ,.5,0.,1.,.5,0)
    r.createView('points'     ,0.,.5,.5,1.,0)
    r.createView('target'     ,.5,.5,1.,1.,0)

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
if(args.max_total_batches>=0):
    should_close.append(lambda:trainer.metaData.batches>=args.max_total_batches)
if(args.stagnation_batches!=-1):
    should_close.append(lambda:trainer.is_stagnant())
should_close=stack_fun(should_close)

kernelItf.initialize()

if(args.train):
    t.start_trainer_thread()
try:
    if(args.live_render):
        samples_enabled=True
        while(not should_close()):#(not args.timeout or e-s<args.timeout_s) and (args.max_batches<0 or trainer.TOTAL_BATCHES_THIS_RUN<=args.max_batches)):
            if(r.is_window_minimized()):
                r.sleep_until_not_minimized(60)
                ev=r.update()
            else:
                ne=time()
                delta=ne-e
                e=ne
                
                controller.process_sdl_events(ev,delta)
                
                if(controller.samples_enabled != samples_enabled):
                    if(samples_enabled):
                        #DISABLE SAMPLES
                        r.createView('interactive',0.,0.,1,1,0,True)
                        pass
                    else:
                        #ENABLE SAMPLES
                        r.createView('interactive',0.,0.,.5,.5,0,True)
                        r.createView('render'     ,.5,0.,1.,.5,0)
                        r.createView('points'     ,0.,.5,.5,1.,0)
                        r.createView('target'     ,.5,.5,1.,1.,0)
                    samples_enabled=controller.samples_enabled
                
                if(controller.take_screencap):
                    r.screencapViews(args.screencap_folder)
                    controller.take_screencap=False
                    
                
                if(samples_enabled and e-last_test_result>args.example_interval):
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
    args.time_render_speed=args.full_samples_final=False#If it crashed, we probably shouldn't sample/test things
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
        start_save=time()
        t.save_all_samples()
        end_save=time()
        print(f"Saving all samples took {end_save-start_save}")
    
    
    if(args.time_render_speed):
        print("Timing render speed all samples (this may take a while)")
        start_test=time()
        times,backward_times,times_nn,backward_times_nn=t.time_speed_for_all()
        end_test=time()
        print(f"Total time (includes loading) = {end_test-start_test}")
        rf=torch.tensor(times)
        rb=torch.tensor(backward_times)
        nf=torch.tensor(times_nn)
        nb=torch.tensor(backward_times_nn)
        #Note: deviation and such is not actual deviation because i do not know how to get pytorch's timers to give me times for individual runs
        print("Note: nn times were measured 5 times less")
        print(f"Average time (forward  w/o nn):{rf.mean()}, deviation:{rf.std()} {torch.std_mean(rf)}, median {rf.median()}, min/max {rf.min()}/{rf.max()}")
        print(f"Average time (backward w/o nn):{rb.mean()}, deviation:{rb.std()} {torch.std_mean(rb)}, median {rb.median()}, min/max {rb.min()}/{rb.max()}")
        print(f"Average time (forward  w/  nn):{nf.mean()}, deviation:{nf.std()} {torch.std_mean(nf)}, median {nf.median()}, min/max {nf.min()}/{nf.max()}")
        print(f"Average time (backward w/  nn):{nb.mean()}, deviation:{nb.std()} {torch.std_mean(nb)}, median {nb.median()}, min/max {nb.min()}/{nb.max()}")
    
    print("Cleanup...")
    kernelItf.cleanup()