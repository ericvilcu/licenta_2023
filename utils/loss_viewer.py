import os
import math
import matplotlib.pyplot as plt
print('Importing TORCH')
import torch
print('Imported  TORCH')
fig = plt.figure()
SCALE_BY_TIME=True
WORKSPACES=[]
WORKSPACE_LABELS=[]
#Just write the capital ones manually. Too lazy to make an input thing for this.
metadatas=[torch.load(os.path.join(WORKSPACE,"meta")) for WORKSPACE in WORKSPACES]
hist_s:list[dict[str,float]]=[metadata['hist'] for metadata in metadatas]
time_s:list[float]=[metadata['times'] for metadata in metadatas]
vhist_s:list[dict[str,float]]=[metadata['histValidation'] for metadata in metadatas]
vtime_s:list[float]=[metadata['timesValidation'] for metadata in metadatas]

batches:int=[metadata['batches'] for metadata in metadatas]
SKIPPED_LOSSES=set()
SKIPPED_LOSSES=set(['l1','psnr','ssim','lpips_alex','lpips_vgg','lpips_squeeze','l1+vgg'])
SKIPPED_LOSSES-={'l1','lpips_vgg'}
#SKIPPED_LOSSES-={'l1+vgg'}


if('l1+vgg' not in SKIPPED_LOSSES):
    hist_s=[[{**b,'l1+vgg':b['l1']+b['lpips_vgg']}for b in hist] for hist in hist_s]
    vhist_s=[[{**b,'l1+vgg':b['l1']+b['lpips_vgg']}for b in vhist] for vhist in vhist_s]

ax = fig.add_subplot(1, 1, 1)
clrs=['red','blue','green','yellow','violet','grey',
                   'darkred','darkblue','darkgreen','darkkhaki','darkviolet','black',
                  'coral','skyblue','lightgreen','lightyellow','thistle','lightgrey']
#clrs=['red','darkred','blue','darkblue','yellow','darkkhaki']
current_color_idx=0
for hist_id,(hist,times,vhist,vtime) in enumerate(zip(hist_s,time_s,vhist_s,vtime_s)):
    nl=len(hist[0])
    ll=list(range(len(hist)))
    lv=[i for i in vtime]
    if(SCALE_BY_TIME):
        ll:list[float]=[i for i in times]
        lv:list[float]=[i for i in vtime]
        run_hist:dict[int, tuple[dict[str, str or bool], float]]=metadatas[hist_id]['run_hist']
        run_hist[len(hist)]=({},math.inf)
        last_idx=-1
        previous_time=0
        for idx in sorted(run_hist.keys()):
            if(last_idx!=-1):
                start_time=run_hist[last_idx][1]
                for i in range(last_idx,idx):
                    ll[i]=ll[i]-start_time+previous_time
                previous_time=ll[idx-1]
            last_idx=idx
            #print(run_hist[idx][0]["loss_type"] if len(run_hist[idx])>0 and "loss_type" in run_hist[idx][0] else "no loss")
        v_idx=0;last_idx=-1;previous_time=0
        for idx in sorted(run_hist.keys()):
            if(last_idx!=-1):
                start_time=run_hist[last_idx][1]
                next_time=run_hist[idx][1]
                while(v_idx<len(lv) and start_time<lv[v_idx]):
                    lv[v_idx]=lv[v_idx]-start_time+previous_time
                    v_idx+=1
                previous_time=lv[v_idx-1]
            last_idx=idx
        
        ll=[i/3600 for i in ll] 
        lv=[i/3600 for i in lv] 
    else:
        ...#lv based on other batches?
    for train in [True,False]:
        for lt in hist[0]:
            if(lt in SKIPPED_LOSSES):
                continue
            if(train):
                loss_values:list[float]=[(hist[epoch][lt] if lt in hist[epoch] else 1.0) for epoch in range(len(hist))]
            else:
                loss_values:list[float]=[(vhist[epoch][lt] if lt in hist[epoch] else 1.0) for epoch in range(len(vhist))]
            mx=max(loss_values)
            t=1
            if(mx>10):
                mx/=20;t*=20
                while(mx>1.2):
                    mx/=2;t*=2
            if(t!=1):loss_values=[loss/t for loss in loss_values]
            #axL = fig.add_subplot(121, title=f"loss {lt}{'' if t==1 else f' (divided by {t})'}")
            ax.plot(ll if train else lv, loss_values, color=clrs[current_color_idx], label=f"{WORKSPACE_LABELS[hist_id]}{'(validation)' if not train else ''}->{lt}{'' if t==1 else f' (divided by {t})'}")
            current_color_idx+=1
ax.set_xlabel("training time (hrs)" if SCALE_BY_TIME else "Batches")
ax.set_ylabel("loss")
fig.legend(loc='upper left')
fig.show()

input()