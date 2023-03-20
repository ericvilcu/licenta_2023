import args



print("importing torch")
import torch
print("torch imported")
import trainer


from GLRenderer import Renderer
from time import time
from renderModule import renderModule
from dataSet import DataSet
import kernelItf
import cameraController
import sdl2

def make_workspace():
    print(f"loading dataset from scenes={set(args.scenes)}")
    ds = DataSet(scenes=args.scenes)
    t=trainer.trainer(ds,**args.nn_args)
    print(f"saving workspace to {args.workspace}")
    t.save(args.workspace)
    


if(args.make_workspace):
    make_workspace()

print("loading workspace...")
t=trainer.trainer.load(args.workspace,args=args.nn_args)
print("workspace loaded...")
ds = t.data


r = Renderer()
r.removeView('main')
r.createView('interactive',0.,0.,.5,.5,0,True)
r.createView('target'     ,.5,0.,1.,.5,0,True)
r.createView('points'     ,0.,.5,.5,1.,0,True)
r.createView('render'     ,.5,.5,1.,1.,0,True)

#ds = DataSet(scenes=args.scenes)
s=e=time()

pm=renderModule()
t=trainer.trainer(ds,**args.nn_args)
pm.to('cuda')
last_test_result=time()-args.example_interval
controller=cameraController.CameraController()
ev=[]



kernelItf.initialize()

if(args.train):
    t.start_trainer_thread()
try:
    while(e-s<900):
        ne=time()
        delta=ne-e
        e=ne
        
        controller.process_sdl_events(ev,delta)
        
        if(e-last_test_result>args.example_interval):
            last_test_result=e
            t.display_results_to_renderer(r,'points','render','target')
        #interactive view.
        
        cam=controller.get_camera()
        view = pm.forward(controller.camera_type(),cam,ds.scenes[0].data.points,ds.scenes[0].data.environment)    
        r.upload_tensor('interactive',view)
        
        ev=r.update()
        #TODO:autosave-check
        
        for e1 in ev:
            if(e1.type == sdl2.events.SDL_QUIT):
                print("Quitting manually...")
                s=e-1e9

except KeyboardInterrupt as ex:
    print(ex)
    print("KeyboardInterrupt detected, stopping and saving...")
    s=e-1e9
finally:
    
    #TODO:save model
    
    if(args.train):
        t.stop_trainer_thread()

    kernelItf.cleanup()