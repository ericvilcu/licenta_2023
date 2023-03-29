import args



print("importing torch")
import torch
print("torch imported")
import trainer


from GLRenderer import Renderer
from time import time,sleep
from renderModule import renderModule
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
    r = Renderer()
    r.removeView('main')
    r.createView('interactive',0.,0.,.5,.5,0,True)
    r.createView('target'     ,.5,0.,1.,.5,0,True)
    r.createView('points'     ,0.,.5,.5,1.,0,True)
    r.createView('render'     ,.5,.5,1.,1.,0,True)

#ds = DataSet(scenes=args.scenes)
s=e=time()

pm=renderModule()
pm.to('cuda')
last_test_result=time()-args.example_interval
controller=cameraController.CameraController()
ev=[]



kernelItf.initialize()

if(args.train):
    t.start_trainer_thread()
try:
    if(args.live_render):
        while((not args.timeout or e-s<args.timeout_s) and (args.max_batches<0 or trainer.TOTAL_BATCHES_THIS_RUN<=args.max_batches)):
            if(r.is_window_minimized()):
                r.sleep_until_not_minimized()
                ev=r.update()
            else:
                ne=time()
                delta=ne-e
                e=ne
                
                controller.process_sdl_events(ev,delta)
                
                if(e-last_test_result>args.example_interval):
                    last_test_result=e
                    t.display_results_to_renderer(r,'points','render','target')
                #interactive view.
                
                cam=controller.get_camera()
                if(controller.use_neural):
                    view = t.size_safe_forward_nograd(controller.camera_type(),cam,0)
                else:
                    view = controller.only_shown_dimensions(pm.forward(controller.camera_type(),cam,ds.scenes[0].data.points,ds.scenes[0].data.environment)    )
                r.upload_tensor('interactive',view)
                ev=r.update()
                #TODO:autosave-check
                
                if(len([e1 for e1 in ev if(e1.type == sdl2.events.SDL_QUIT)])>0):
                    print("Quitting manually...")
                    break
    else:
        while((not args.timeout or e-s<args.timeout_s) and (args.max_batches<0 or trainer.TOTAL_BATCHES_THIS_RUN<=args.max_batches)):
            sleep(60)
except KeyboardInterrupt as ex:
    print(ex)
    print("KeyboardInterrupt detected, stopping and saving...")
finally:
    
    if(args.train):
        print("Waiting for last batch to be trained...")
        t.stop_trainer_thread()

    print("Saving...")
    t.save(args.workspace)

    print("Cleanup...")
    kernelItf.cleanup()