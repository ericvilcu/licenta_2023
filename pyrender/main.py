import args



print("importing torch")
import torch
print("torch imported")
import trainer

from GLRenderer import Renderer
from time import time
from renderModule import renderModule
from dataSet import DataSet
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
camera=[0.,0.,CW,CH,*identity,*(0.,0.,0.),CW/2,CH/2,CW/2,CH/2]#*(2+9+3+4)
pm=renderModule()
t=trainer.trainer(ds)
pm.to('cuda')
last_test_result=time()-args.example_interval
while(e-s<900):
    e=time()
    
    if(e-last_test_result<args.example_interval):
        last_test_result=e
        t.display_results_to_renderer(r,'points','render','target')
    #interactive view.
    
    view = pm.forward(0,torch.tensor(camera),ds.scenes[0].data.points,ds.scenes[0].data.environment)    
    r.upload_tensor('interactive',view)
    
    ev=r.update()
    for e1 in ev:
        if(e1.type == sdl2.events.SDL_QUIT):
            print("Quitting manually...")
            s=e-1e9
            
    if(args.train):
        t.train_one()


