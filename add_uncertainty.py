import read_json
import numpy as np
import torch
from torchtraj.utils import T, XY,WPTS
from torchtraj import flights, named, uncertainty
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,masked_generate,plot,recplot
import pandas as pd
import matplotlib.pyplot as plt
from torchtraj import fit, traj
import datetime
import geosphere
import operator as op
import matplotlib.animation as animation

DANGLE = "dangle"
MAN_WPTS = "man_wpts"
DT0 = "dt0"
DT1 = "dt1"
DSPEED = "dspeed"
LDSPEED = "ldspeed"


def plotanimate(lxy,s=1.5):
    fig,ax = plt.subplots() # initialise la figure
    scats = tuple(ax.scatter([],[],s=s) for _ in lxy)
    margin = 200000.
    xmin = min([named.nanmin(xy[...,0].rename(None)) for xy in lxy])-margin
    xmax = max([named.nanmax(xy[...,0].rename(None)) for xy in lxy])+margin
    ymin = min([named.nanmin(xy[...,1].rename(None)) for xy in lxy])-margin
    ymax = max([named.nanmax(xy[...,1].rename(None)) for xy in lxy])+margin

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().axis('equal')
    def init():
        for scat in scats:
            scat.set_offsets(np.array([[],[]]).T)
        return scats
    def animate(i):
        # print("pos")
        # print(pos)
        # raise Exception
        for scat,xy in zip(scats,lxy):
            pos = xy[...,i:i+40:4,:].rename(None).flatten(end_dim=-2).numpy()
            scat.set_offsets(pos)
        return scats
    ani = animation.FuncAnimation(fig, func=animate, init_func=init, frames=lxy[0].shape[-2],
                              interval=1, blit=True, repeat=True)

    plt.show()







# def compute_wpts_uncertainty(fdeviated,fothers):
#     diwpts = {}
#     dtwpts = {}
#     for v in ["tdeviation","tturn","trejoin"]:
#         dtwpts[v] = getattr(fdeviated,v)
#         fdeviated.fxy = fdeviated.fxy.add_wpt_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
#         fothers.fxy = fothers.fxy.add_wpt_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
#     print(fdeviated.fxy.duration.cumsum(axis=-1))
#     print(fothers.fxy.duration.cumsum(axis=-1))
#     # raise Exception
#     dothersiwpts = {}
#     for v in ["tdeviation","tturn","trejoin"]:
#         diwpts[v]= fdeviated.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
#         dothersiwpts[v]= fothers.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
#     return diwpts,dothersiwpts


def compute_iwpts(f,dtimes):
    stimes=set([t for times in dtimes.values() for t in times.values()])
    print(len(stimes))
    raise Exception
    for t in stimes:
        f.fxy = f.fxy.add_wpt_at_t(named.unsqueeze(t,-1,WPTS))
        f.fz = f.fz.add_wpt_at_t(named.unsqueeze(t,-1,WPTS))
    diwpts = {}
    for v in ["tdeviation","tturn","trejoin"]:
        diwpts[v]= fdeviated.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
        dothersiwpts[v]= fothers.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
    return diwpts



def uncertainty_others(fdeviated,fothers,diwpts,dothersiwpts):
    fdeviated.fxy = uncertainty.addangle(dangle,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # t0
    fdeviated.fxy = uncertainty.adddt_rotate(dt.rename(**{DT:"dt0"}),diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # t1
    fdeviated.fxy = uncertainty.adddt_rotate(dt,diwpts["tturn"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # speed
    fdeviated.fxy = uncertainty.changespeed_rotate(dspeed,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)#,contract=False)
    # longitudinalspeed
    fothers.fxy = uncertainty.change_longitudinal_speed(ldspeed,dothersiwpts["tdeviation"],dothersiwpts["trejoin"],fothers.fxy)
    return fdeviated,fothers,diwpts

def uncertainty_deviated(fdeviated,diwpts):
    fdeviated.fxy = uncertainty.addangle(dangle,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # t0
    fdeviated.fxy = uncertainty.adddt_rotate(dt.rename(**{DT:"dt0"}),diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # t1
    fdeviated.fxy = uncertainty.adddt_rotate(dt,diwpts["tturn"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)
    # speed
    fdeviated.fxy = uncertainty.changespeed_rotate(dspeed,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fdeviated.fxy,beacon=fdeviated.beacon)#,contract=False)
    # longitudinalspeed
    fothers.fxy = uncertainty.change_longitudinal_speed(ldspeed,dothersiwpts["tdeviation"],dothersiwpts["trejoin"],fothers.fxy)
    return fdeviated,fothers,diwpts

def generate(fdeviated,fothers,diwpts):
    fs = fdeviated.fxy
    fo = fothers.fxy
    modified_trejoin = uncertainty.gather_wpts(fs.duration.cumsum(axis=-1),diwpts["trejoin"]-1)[...,0]
    print(f"{fdeviated.trejoin=}")
    print(f"{modified_trejoin=}")
    max_duration = (fdeviated.trejoin - fdeviated.tdeviation).max().item()
    print(max_duration)
    t = torch.arange(start=0.,end=max_duration,step=1,device=device).rename(T)
    print(t)
    print(fdeviated.tdeviation)
    # a,b = named.align_common(fdeviated.tdeviation,t)
    t = op.add(*named.align_common(fdeviated.tdeviation,t)).align_to(...,T)
    t = torch.arange(start=0.,end=t.max(),step=1,device=device).rename(T)
    # print(fs.duration)
    #f = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    f = fs
    zs = masked_generate(fs.fz,t,fs.t)[...,0]
    zo = masked_generate(fo.fz,t,fo.t)[...,0]
    mask_closez = op.sub(*named.align_common(zs,zo)).abs() < 500.
    mask_not_closez = torch.logical_not(mask_closez)
    mask_closez = mask_closez / mask_closez
    mask_not_closez = mask_not_closez / mask_not_closez
    # print(mask_closez)
    xys = masked_generate(fs,t,fdeviated.t)
    xys = xys #* mask_closez.align_as(xys)
    xyo = masked_generate(fo,t,fothers.t)# * mask_closez / mask_closez
    xyoc = xyo * mask_closez.align_as(xyo)
    xyon = xyo * mask_not_closez.align_as(xyo)
    return [xys.cpu(),xyoc.cpu(),xyon.cpu()]
    # # print(xyo)
    # # print(xyo.shape)
    # # print(xys)
    # # print(xys.shape)
    # # print(zs)
    # # print(zo)
    # # raise Exception
    # # dist = op.sub(*named.align_common(xys,xyo.rename(**{BATCH:OTHERS}))).abs().align_to(...,OTHERS,T,XY)
    # # plotanimate([xys.cpu()],s=4)
    # print(xys.shape,xys.names)
    # print(xyo.shape,xyo.names)
    # plotanimate([xys.cpu(),xyoc.cpu(),xyon.cpu()],s=4)
    # # print(f.meanv())
    # # print(f.duration)
    # # print(fs.meanv())
    # #plot(fothers.fxy,t,xory)
    # print(diwpts["tdeviation"])
    # print(diwpts["tturn"])
    # print(diwpts["trejoin"])
    # xory=False
    # plot(fs,t,xory)
    # if not xory:
    #     recplot(fdeviated.beacon.cpu(),plt.scatter)
    #     plt.gca().axis('equal')
    # print(f)
    # print(fs)
    # # print(fdeviated.tdeviation)
    # # print(traj.generate(f,t))
    # plt.show()
def modify(dico,dmodif):
    res = dico.copy()
    for k,f in dmodif.items():
        res[k]=f(res[k])
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situation')
    args = parser.parse_args()
    # print(args.json)
    sit = load_situation(args.situation)
    device="cpu"
    dtimes = {k:getattr(sit["deviated"],k) for k in ["tdeviation","tturn","trejoin"]}
    dtimes_deviated = {
        DANGLE: dtimes,
        DT0: modify(dtimes,{"tdeviation":lambda x:x-1}),
        DT1: dtimes,
        DSPEED: dtimes,
    }
    dtimes_others = {
        LDSPEED: dtimes,
    }
    ditimes_deviated = compute_iwpts(sit["deviated"],dtimes_deviated)

    deviated_uncertainty = {
        DANGLE: torch.tensor([-0.,0.],device=device).reshape(-1).rename(DANGLE),
        DT0: 1*torch.tensor([0,0],device=device).reshape(-1).rename(DT0),
        DT1: 1*torch.tensor([0,0],device=device).reshape(-1).rename(DT1),
        DSPEED: 1*torch.tensor([0,0],device=device).reshape(-1).rename(DSPEED),
    }
    others_uncertainty = {
        LDSPEED: 1*torch.tensor([0,0],device=device).reshape(-1).rename(LDSPEED),
    }
    others_uncertainty_delta = {}
    fdeviated = uncertainty_deviated(sit["deviated"],diwpts)
    fothers = uncertainty_others(sit["others"],diwpts)
    situn = add_uncertainty(sit["deviated"],sit["others"],*diwpts)
    lxys = generate(*situn)
    plotanimate(lxys,s=4)
if __name__ == "__main__":
    main()
