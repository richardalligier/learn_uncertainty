import read_json
import numpy as np
import torch
from torchtraj.utils import T, XY,WPTS, apply_mask
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
VSPEED = "vspeed"

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
    f = f.clone()
    for transfo in ljob:
        f=transfo(f)
    return f

def apply_uncertainty_others(fothers,dothersiwpts,dargs,uparams):
    dixy = dothersiwpts["fxy"]
    uxy = uparams["fxy"]
    ljob_xy = [
        lambda f: uncertainty.change_longitudinal_speed(uxy[LDSPEED],dixy[LDSPEED]["tdeviation"],dixy[LDSPEED]["trejoin"],f)
    ]
    fothers.fxy = apply_uncertainty(fothers.fxy,ljob_xy)
    uz = uparams["fz"]
    zargs = dargs["fz"]
    ljob_z = [
        lambda f:uncertainty.change_vertical_speed_fwd(uz[VSPEED],zargs[VSPEED]["tmin"],zargs[VSPEED]["tmax"],f,)
    ]
    fothers.fz = apply_uncertainty(fothers.fz,ljob_z)
    return fothers

def apply_uncertainty_deviated(fdeviated,diwpts,dargs,uparams):
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

# def apply_mask(res,mask):
#     return res * mask.align_as(res)

class WithUncertainty:
    def __init__(self,sitf,dtimes,dargs,apply_uncertainty):
        self.sitf = sitf.clone()
        self.idtimes = getiwpts(self.sitf,dtimes)
        self.dargs = dargs
        self.apply_uncertainty = apply_uncertainty
    def add_uncertainty(self,uparams):
        return self.apply_uncertainty(self.sitf.clone(),self.idtimes,self.dargs,uparams)
    def generate_xy(self,uparams,t):
        return self.add_uncertainty(uparams).generate_xy(t)
    def generate_z(self,uparams,t):
        return self.sitf.generate_z(t)
    def generate_tz(self,uparams,t):
        return self.sitf.generate_tz(t)

def precompute_situation_uncertainty(sit):
    timesofinterest = {k:getattr(sit["deviated"],k) for k in ["tdeviation","tturn","trejoin"]}
    ztimesofinterest = {
        "tmin":named.nanamin(sit["others"].t,dim=T),
        "tmax":named.nanamax(sit["others"].t,dim=T),
    }
    # print(ztimesofinterest["tmin"].min())
    # print(ztimesofinterest["tmax"].max())
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
            },
            "fz": {
                VSPEED: ztimesofinterest,
            },
        }
    }
    dargs = {
        "deviated":{},
        "others":{
            "fz": {
                VSPEED: ztimesofinterest,
            },
        }
    }
    duncertainty = {
        "deviated":apply_uncertainty_deviated,
        "others":apply_uncertainty_others
    }
    return {k:WithUncertainty(s,dtimes[k],dargs[k],duncertainty[k]) for k,s in sit.items()}

def modify(dico,dmodif):
    res = dico.copy()
    for k,f in dmodif.items():
        res[k]=f(res[k])
    return res



def make_t_without0(f,t):
    t1 = f.duration.align_to(...,WPTS)[...,:1]
    t = t.align_as(t1)
    mask = (t == 0.)
    return mask,named.where(mask,t1,t)

def add_wpt_at_t_robust(f,t):
    mask,t = make_t_without0(f,t)
    return f.add_wpt_at_t(t)

def wpts_at_t_robust(f,t):
    mask,t = make_t_without0(f,t)
    mask = mask.align_to(...,WPTS)
    assert(mask.shape[-1]==1)
    mask = mask[...,0]
    res = f.wpts_at_t(t).align_as(mask)
    print(res.names)
    print(mask.names)
    return named.where(mask,torch.zeros_like(res),res)


def compute_iwpts(f,dtimes):
    stimes=set([t for times in dtimes.values() for t in times.values()])
    print(dtimes)
    print(len(stimes))
    for t in stimes:
        f = add_wpt_at_t_robust(f,named.unsqueeze(t,-1,WPTS))#f.add_wpt_at_t(named.unsqueeze(t,-1,WPTS))#f.add_wpt_at_t(named.unsqueeze(t,-1,WPTS))
    # d={t:f.wpts_at_t(named.unsqueeze(t,-1,WPTS)) for t in stimes}
    d={t:wpts_at_t_robust(f,named.unsqueeze(t,-1,WPTS)) for t in stimes}
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

def generate_sitothers_test_vz_(sitothers):
    torch.manual_seed(44)
    vz = torch.randn_like(sitothers.fz.v)*3
    m = torch.abs(vz) < 200/60
    vz.rename(None)[m.rename(None)]=vz.rename(None)[m.rename(None)]/10#000000
    # m = ~m
    # vz.rename(None)[m.rename(None)]=vz.rename(None)[m.rename(None)]/200
    print(vz)
    vt = torch.ones_like(sitothers.fz.v)
    vxy = named.stack([vt,vz],XY).align_to(...,XY)
    v = torch.hypot(vxy[...,0],vxy[...,1])
    theta = torch.atan2(vxy[...,1],vxy[...,0])
    return v,theta
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
    sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...)[346:347])
    sit["others"].fz.v,sit["others"].fz.theta= generate_sitothers_test_vz_(sit["others"])
    device="cpu"

    uparams = {
        "deviated":{
            "fxy":{
                DANGLE: torch.tensor([0.],device=device).reshape(-1).rename(DANGLE),
                DT0: 1*torch.tensor([0],device=device).reshape(-1).rename(DT0),
                DT1: 1*torch.tensor([0],device=device).reshape(-1).rename(DT1),
                DSPEED: 1*torch.tensor([1.],device=device).reshape(-1).rename(DSPEED),
            }
        },
        "others":{
            "fxy":{
                LDSPEED: 1*torch.tensor([1.],device=device).reshape(-1).rename(LDSPEED),
            },
            "fz":{
                VSPEED: 1*torch.tensor([2,1.,0.7,0.5],device=device).reshape(-1).rename(VSPEED),
            }
        }
    }
    sit_uncertainty = precompute_situation_uncertainty(sit)
    max_duration = max(s.t.max() for s in sit.values())
    t = torch.arange(start=0.,end=max_duration,step=1,device=device).rename(T)
    masked_t = {k:s.generate_mask(t,thresh=20) for k,s in sit.items()}
    xy_u = {k:apply_mask(s.generate_xy(uparams[k],t),mask=masked_t[k])for k,s in sit_uncertainty.items()}
    tz_u = {k:apply_mask(s.generate_tz(uparams[k],t),mask=masked_t[k])for k,s in sit_uncertainty.items()}
    z_u = {k:apply_mask(s.generate_z(uparams[k],t),mask=masked_t[k])for k,s in sit_uncertainty.items()}
    diffz = torch.abs(z_u["others"]-z_u["deviated"].align_as(z_u["others"])) < 800
    print(z_u["others"].names)
    print(z_u["others"].shape)
    print(diffz.names)
    print(diffz.shape)
    # raise Exception
    # def compute_wpts_with_wpts0(f):
    #     wpts = f.compute_wpts().align_to(...,WPTS,XY)
    #     xy0 = f.xy0.align_as(wpts)
    #     s=list(named.broadcastshapes(wpts.shape,xy0.shape))
    #     s[-2]=1
    #     xy0 = torch.broadcast_to(xy0,s)
    #     print(wpts.names,wpts.shape)
    #     print(xy0.names,xy0.shape)
    #     return named.cat([xy0,wpts],dim=-2)
    wpts_xy = {k:s.add_uncertainty(uparams[k]).fxy.compute_wpts_with_wpts0() for k,s in sit_uncertainty.items()}
    wpts_z = {k:s.add_uncertainty(uparams[k]).fz.compute_wpts_with_wpts0() for k,s in sit_uncertainty.items()}

    # raise Exception
    if args.wpts == "xy":
        for k,wpts in wpts_xy.items():
            print(k)
            print(wpts)
            print(wpts.shape,wpts.names)
            # raise Exception
            recplot(wpts,lambda x,y:scatter_with_number(x,y,0))
            plt.show()
    elif args.wpts =="z":
        for k,wpts in wpts_z.items():
            if k=="others":
                print(k)
                # wpts =wpts.align_to(OTHERS,...)[45:]
                s = slice(45,46)
                s = slice(28,29)
                s = slice(29,30)
                wpts =wpts.align_to(OTHERS,...)#[s]
                # wpts =wpts.align_to(OTHERS,...)#[43:44]
                print(wpts.names)
                print(wpts.shape)
                #python3 add_uncertainty.py -situation ./situations/38893618_1657871463_1657872229.situation -wpts z
                recplot(wpts,lambda x,y:scatter_with_number(x,y,0))
                plt.show()
    # raise Exception
    if args.animate == "xy":
        plotanimate([xy_u["deviated"],apply_mask(xy_u["others"],diffz/diffz),apply_mask(xy_u["others"],(~diffz)/(~diffz))],s=4)
    elif args.animate =="z":
        plotanimate(list(tz_u.values()),s=4,margin=10,equal=False)
if __name__ == "__main__":
    main()
#python3 add_uncertainty.py -situation ./situations/38930310_1657885467_1657885501.situation -wpts z
