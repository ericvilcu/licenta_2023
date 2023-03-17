import args



print("importing torch")
import torch
print("torch imported")
import trainer

from GLRenderer import Renderer
from time import time
from renderModule import renderModule
from dataSet import DataSet
import cameraController
import sdl2



r = Renderer()
r.removeView('main')
r.createView('interactive',0.,0.,.5,.5,0,True)
r.createView('points'     ,.5,0.,1.,.5,0,True)
r.createView('target'     ,0.,.5,.5,1.,0,True)
r.createView('render'     ,.5,.5,1.,1.,0,True)

ds = DataSet(scenes=args.scenes)
s=e=time()
identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
CW,CH=300.,300.
pm=renderModule()
t=trainer.trainer(ds)
pm.to('cuda')
last_test_result=time()-args.example_interval
controller=cameraController.CameraController()
ev=[]
while(e-s<900):
    try:
        ne=time()
        controller.process_sdl_events(ev,ne-e)
        e=ne
        
        if(e-last_test_result>args.example_interval):
            last_test_result=e
            t.display_results_to_renderer(r,'points','render','target')
        #interactive view.
        
        cam=controller.get_camera()
        view = pm.forward(controller.camera_type(),cam,ds.scenes[0].data.points,ds.scenes[0].data.environment)    
        r.upload_tensor('interactive',view)
        
        ev=r.update()
        #if(args.train):
        #    t.train_one()
    except KeyboardInterrupt as e:
        print(e)
        s=e-1e9
        #NOTE: this is here to ensure it save
    
    #TODO:autosave-check
        
    for e1 in ev:
        if(e1.type == sdl2.events.SDL_QUIT):
            print("Quitting manually...")
            s=e-1e9
            
#TODO:save model
