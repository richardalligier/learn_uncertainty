import read_json
import numpy as np
import torch
from torchtraj.utils import T, XY,WPTS
from torchtraj import flights, named, uncertainty
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number
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

def plotanimate(lxy,s=1.5,margin=20.,equal=True):
    fig,ax = plt.subplots() # initialise la figure
    scats = tuple(ax.scatter([],[],s=s) for _ in lxy)
    xmin = min([named.nanmin(xy[...,0].rename(None)) for xy in lxy]).item()-margin

    xmax = max([named.nanmax(xy[...,0].rename(None)) for xy in lxy]).item()+margin
    ymin = min([named.nanmin(xy[...,1].rename(None)) for xy in lxy]).item()-margin
    ymax = max([named.nanmax(xy[...,1].rename(None)) for xy in lxy]).item()+margin
    # print(xmin,xmax)
    # print(ymin,ymax)
    # raise Exception
    plt.axis([xmin, xmax, ymin, ymax])
    if equal:
        plt.gca().set_aspect('equal', adjustable='box')
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




def apply_uncertainty(f,ljob):
    for transfo in ljob:
        f=transfo(f)
    return f

def apply_uncertainty_others(fothers,dothersiwpts,uparams):
    dixy = dothersiwpts["fxy"]
    uxy = uparams["fxy"]
    ljob_xy = [
        lambda f: uncertainty.change_longitudinal_speed(uxy[LDSPEED],dixy[LDSPEED]["tdeviation"],dixy[LDSPEED]["trejoin"],f)
    ]
    fothers.fxy = apply_uncertainty(fothers.fxy,ljob_xy)
    return fothers

def apply_uncertainty_deviated(fdeviated,diwpts,uparams):
    dixy = diwpts["fxy"]
    uxy = uparams["fxy"]
    ljob_xy = [
        lambda f:uncertainty.addangle(uxy[DANGLE],dixy[DANGLE]["tdeviation"],dixy[DANGLE]["tturn"],dixy[DANGLE]["trejoin"],f,beacon=fdeviated.beacon),
        lambda f:uncertainty.adddt_rotate(uxy[DT0],dixy[DT0]["tdeviation"],dixy[DT0]["tturn"],dixy[DT0]["trejoin"],f,beacon=fdeviated.beacon),
        lambda f:uncertainty.adddt_rotate(uxy[DT1],dixy[DT1]["tturn"],dixy[DT1]["tturn"],dixy[DT1]["trejoin"],f,beacon=fdeviated.beacon),
        lambda f:uncertainty.changespeed_rotate(uxy[DSPEED],dixy[DSPEED]["tdeviation"],dixy[DSPEED]["tturn"],dixy[DSPEED]["trejoin"],f,beacon=fdeviated.beacon),
    ]
    fdeviated.fxy = apply_uncertainty(fdeviated.fxy,ljob_xy)
    return fdeviated

def apply_mask(res,mask):
    return res * mask.align_as(res)

class WithUncertainty:
    def __init__(self,sitf,dtimes,apply_uncertainty):
        self.sitf = sitf.clone()
        self.idtimes = getiwpts(self.sitf,dtimes)
        self.apply_uncertainty = apply_uncertainty
    def add_uncertainty(self,uparams):
        return self.apply_uncertainty(self.sitf.clone(),self.idtimes,uparams)
    def generate_xy(self,uparams,t):
        return self.add_uncertainty(uparams).generate_xy(t)
    def generate_z(self,uparams,t):
        return self.sitf.generate_z(t)
    def generate_tz(self,uparams,t):
        return self.sitf.generate_tz(t)

def precompute_situation_uncertainty(sit):
    timesofinterest = {k:getattr(sit["deviated"],k) for k in ["tdeviation","tturn","trejoin"]}
    dtimes = {
        "deviated":{
            "fxy": {
                DANGLE: timesofinterest,
                DT0: modify(timesofinterest,{"tdeviation":lambda x:x-1}),
                DT1: timesofinterest,
                DSPEED: timesofinterest,
            }
        },
        "others":{
            "fxy": {
                LDSPEED: timesofinterest,
            }
        }
    }
    duncertainty = {
        "deviated":apply_uncertainty_deviated,
        "others":apply_uncertainty_others
    }
    return {k:WithUncertainty(s,dtimes[k],duncertainty[k]) for k,s in sit.items()}

def modify(dico,dmodif):
    res = dico.copy()
    for k,f in dmodif.items():
        res[k]=f(res[k])
    return res


def compute_iwpts(f,dtimes):
    stimes=set([t for times in dtimes.values() for t in times.values()])
    print(len(stimes))
    for t in stimes:
        f = f.add_wpt_at_t(named.unsqueeze(t,-1,WPTS))
    d={t:f.wpts_at_t(named.unsqueeze(t,-1,WPTS)) for t in stimes}
    res = {}
    for k,times in dtimes.items():
        res[k]={kk:d[t] for kk,t in times.items()}
    return f,res

def getiwpts(sitf,dtimes):
    ditimes = {k:compute_iwpts(getattr(sitf,k),v) for k,v in dtimes.items()}
    res = {}
    for k,(f,iwpts) in ditimes.items():
        res[k]=iwpts
        setattr(sitf,k,f)
    return res

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situation')
    parser.add_argument('-animate',default=None)
    parser.add_argument('-wpts',default=None)
    args = parser.parse_args()
    # print(args.json)
    sit = load_situation(args.situation)
    device="cpu"

    uparams = {
        "deviated":{
            "fxy":{
                DANGLE: torch.tensor([-0.2,0.2],device=device).reshape(-1).rename(DANGLE),
                DT0: 1*torch.tensor([0,0],device=device).reshape(-1).rename(DT0),
                DT1: 1*torch.tensor([0,0],device=device).reshape(-1).rename(DT1),
                DSPEED: 1*torch.tensor([1.,1.],device=device).reshape(-1).rename(DSPEED),
            }
        },
        "others":{
            "fxy":{
                LDSPEED: 1*torch.tensor([1.,1.],device=device).reshape(-1).rename(LDSPEED),
            }
        }
    }
    sit_uncertainty = precompute_situation_uncertainty(sit)
    max_duration = max(s.t.max() for s in sit.values())
    t = torch.arange(start=0.,end=max_duration,step=1,device=device).rename(T)
    masked_t = {k:s.generate_mask(t,thresh=20) for k,s in sit.items()}
    xy_u = {k:apply_mask(s.generate_xy(uparams[k],t),mask=masked_t[k])for k,s in sit_uncertainty.items()}
    z_u = {k:apply_mask(s.generate_tz(uparams[k],t),mask=masked_t[k])for k,s in sit_uncertainty.items()}
    diffz = torch.abs(z_u["others"]-z_u["deviated"].align_as(z_u["others"])) < 800

    wpts_xy = {k:s.add_uncertainty(uparams[k]).fxy.compute_wpts() for k,s in sit_uncertainty.items()}
    wpts_z = {k:s.add_uncertainty(uparams[k]).fz.compute_wpts() for k,s in sit_uncertainty.items()}

    # raise Exception
    if args.wpts == "xy":
        for k,wpts in wpts_xy.items():
            print(k)
            print(wpts)
            print(wpts.shape,wpts.names)
            # raise Exception
            recplot(wpts,scatter_with_number)
            plt.show()
    elif args.wpts =="z":
        for k,wpts in wpts_z.items():
            print(k)
            recplot(wpts,scatter_with_number)
            plt.show()
    if args.animate == "xy":
        plotanimate([xy_u["deviated"],apply_mask(xy_u["others"],diffz/diffz),apply_mask(xy_u["others"],(~diffz)/(~diffz))],s=4)
    elif args.animate =="z":
        plotanimate(list(z_u.values()),s=4,margin=10,equal=False)
if __name__ == "__main__":
    main()
