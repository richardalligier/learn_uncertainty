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
from torchtraj.qhull import QhullDist
VALUESTOTEST = "valuestotest"

def plotanimate(lxy,xlabel,ylabel,s=1.5,margin=20.,equal=True):
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
            pos = xy.cpu()[...,i:i+40:4,:].rename(None).flatten(end_dim=-2).numpy()
            scat.set_offsets(pos)
        return scats
    ani = animation.FuncAnimation(fig, func=animate, init_func=init, frames=lxy[0].shape[-2],
                              interval=1, blit=True, repeat=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()




def apply_uncertainty(f,ljob):
    f = f.clone()
    for transfo in ljob:
        f=transfo(f)
    return f


# def apply_mask(res,mask):
#     return res * mask.align_as(res)

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
    # print(res.names)
    # print(mask.names)
    return named.where(mask,torch.zeros_like(res),res)


def compute_iwpts(f,dtimes):
    stimes=set([t for times in dtimes.values() for t in times.values()])
    # print(dtimes)
    # print(len(stimes))
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
    # print(vz)
    vt = torch.ones_like(sitothers.fz.v)
    vxy = named.stack([vt,vz],XY).align_to(...,XY)
    v = torch.hypot(vxy[...,0],vxy[...,1])
    theta = torch.atan2(vxy[...,1],vxy[...,0])
    return v,theta

def minmax(a,dimsInSet):
    if len(dimsInSet) == 0.:
        return a,a
    else:
        amax = named.nanamax(a,dimsInSet)
        amin = named.nanamin(a,dimsInSet)
        return amin,amax

def distz(amin,amax,bmin,bmax):
    r1 = amin-bmax
    r2 = bmin-amax
    return named.maximum(r1,r2)











class WithUncertainty:
    def __init__(self,sitf,dtimes,dargs,apply_uncertainty):
        self.sitf = sitf.clone()
        self.idtimes = getiwpts(self.sitf,dtimes)
        self.dargs = dargs
        self.apply_uncertainty = apply_uncertainty
    def add_uncertainty(self,uparams):
        return self.apply_uncertainty(self.sitf.clone(),self.idtimes,self.dargs,uparams)
    # def generate_xy(self,uparams,t):
    #     return self.add_uncertainty(uparams).generate_xy(t)
    # def generate_z(self,uparams,t):
    #     return self.add_uncertainty(uparams).generate_z(t)
    # def generate_tz(self,uparams,t):
    #     return self.sitf.generate_tz(t)

class Uncertainty_model:
    DANGLE = "dangle"
    MAN_WPTS = "man_wpts"
    DT0 = "dt0"
    DT1 = "dt1"
    DSPEED = "dspeed"
    LDSPEED = "ldspeed"
    VSPEED = "vspeed"
    @classmethod
    def apply_uncertainty_others(cls,fothers,dothersiwpts,dargs,uparams):
        dixy = dothersiwpts["fxy"]
        uxy = uparams["fxy"]
        ljob_xy = [
            lambda f: uncertainty.change_longitudinal_speed(uxy[cls.LDSPEED],dixy[cls.LDSPEED]["tdeviation"],dixy[cls.LDSPEED]["trejoin"],f)
        ]
        fothers.fxy = apply_uncertainty(fothers.fxy,ljob_xy)
        uz = uparams["fz"]
        zargs = dargs["fz"]
        ljob_z = [
            lambda f:uncertainty.change_vertical_speed_fwd(uz[cls.VSPEED],zargs[cls.VSPEED]["tmin"],zargs[cls.VSPEED]["tmax"],f,)
        ]
        fothers.fz = apply_uncertainty(fothers.fz,ljob_z)
        return fothers
    @classmethod
    def apply_uncertainty_deviated(cls,fdeviated,diwpts,dargs,uparams):
        dixy = diwpts["fxy"]
        uxy = uparams["fxy"]
        ljob_xy = [
            lambda f:uncertainty.addangle(uxy[cls.DANGLE],dixy[cls.DANGLE]["tdeviation"],dixy[cls.DANGLE]["tturn"],dixy[cls.DANGLE]["trejoin"],f,beacon=fdeviated.beacon),
            lambda f:uncertainty.adddt_rotate(uxy[cls.DT0],dixy[cls.DT0]["tdeviation"],dixy[cls.DT0]["tturn"],dixy[cls.DT0]["trejoin"],f,beacon=fdeviated.beacon),
            lambda f:uncertainty.adddt_rotate(uxy[cls.DT1],dixy[cls.DT1]["tturn"],dixy[cls.DT1]["tturn"],dixy[cls.DT1]["trejoin"],f,beacon=fdeviated.beacon),
            lambda f:uncertainty.changespeed_rotate(uxy[cls.DSPEED],dixy[cls.DSPEED]["tdeviation"],dixy[cls.DSPEED]["tturn"],dixy[cls.DSPEED]["trejoin"],f,beacon=fdeviated.beacon),
        ]
        fdeviated.fxy = apply_uncertainty(fdeviated.fxy,ljob_xy)
        return fdeviated

    @classmethod
    def precompute_situation_uncertainty(cls,sit):
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
                cls.DANGLE: timesofinterest,
                cls.DT0: modify(timesofinterest,{"tdeviation":lambda x:x-1}),
                cls.DT1: timesofinterest,
                cls.DSPEED: timesofinterest,
            }
        },
        "others":{
            "fxy": {
                cls.LDSPEED: timesofinterest,
            },
            "fz": {
                cls.VSPEED: ztimesofinterest,
            },
        }
        }
        dargs = {
            "deviated":{},
            "others":{
                "fz": {
                    cls.VSPEED: ztimesofinterest,
                },
            }
        }
        duncertainty = {
            "deviated":cls.apply_uncertainty_deviated,
            "others":cls.apply_uncertainty_others
        }
        return {k:WithUncertainty(s,dtimes[k],dargs[k],duncertainty[k]) for k,s in sit.items()}
    @classmethod
    def build_uparams(cls,dangle,dt0,dt1,dspeed,ldspeed,vspeed):
        # print(dangle.names,VALUESTOTEST)
        # raise Exception
        uparams = {
            "deviated":{
                "fxy":{
                    cls.DANGLE: dangle.rename(**{VALUESTOTEST:cls.DANGLE}),
                    cls.DT0: dt0.rename(**{VALUESTOTEST:cls.DT0}),
                    cls.DT1: dt1.rename(**{VALUESTOTEST:cls.DT1}),
                    cls.DSPEED: dspeed.rename(**{VALUESTOTEST:cls.DSPEED}),
            }
            },
            "others":{
                "fxy":{
                    cls.LDSPEED: ldspeed.rename(**{VALUESTOTEST:cls.LDSPEED}),
                },
                "fz":{
                    cls.VSPEED: vspeed.rename(**{VALUESTOTEST:cls.VSPEED}),
                }
            }
        }
        return uparams


class Add_uncertainty:
    def __init__(self,sit,step,thresh_thole=20,dist_discretize=180,umodel=None):
        self.umodel = Uncertainty_model() if umodel is None else umodel
        self.t = sit["deviated"].generate_trange_conflict(step=step)
        self.masked_t = {k:s.generate_mask(self.t,thresh=20) for k,s in sit.items()}
        self.sit_uncertainty = self.umodel.precompute_situation_uncertainty(sit)
        self.qhulldist = QhullDist(n=dist_discretize,device=self.t.device)
    def compute_tz(self,duparams):
        uparams = self.umodel.build_uparams(**duparams)
        f_u = {k:s.add_uncertainty(uparams[k]) for k,s in self.sit_uncertainty.items()}
        return {k:apply_mask(s.generate_tz(self.t),mask=self.masked_t[k])for k,s in f_u.items()}

    def compute_all(self,duparams):
        uparams = self.umodel.build_uparams(**duparams)
        f_u = {k:s.add_uncertainty(uparams[k]) for k,s in self.sit_uncertainty.items()}
        xy_u = {k:apply_mask(s.generate_xy(self.t),mask=self.masked_t[k])for k,s in f_u.items()}
        z_u = {k:apply_mask(s.generate_z(self.t),mask=self.masked_t[k])for k,s in f_u.items()}
        assert("fz" not in list(uparams["deviated"]))
        dist_z= distz(*minmax(z_u["others"],list(uparams["others"]["fz"].keys())),
                      z_u["deviated"],z_u["deviated"])
        names_xy =list(set(uparams["deviated"]["fxy"].keys()).union(set(uparams["others"]["fxy"].keys())))
        dist_xy = self.qhulldist.dist(xy_u["others"],xy_u["deviated"],dimsInSet=names_xy)
        return dist_xy,dist_z,xy_u,z_u

    def compute_min_distance_xy_on_conflicting_z(self,duparams,thresh_z):
        dist_xy,dist_z,xy_u,z_u = self.compute_all(duparams)
        conflict_z = dist_z < thresh_z
        return named.nanamin(apply_mask(dist_xy,conflict_z/conflict_z),dim=(OTHERS,T))


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
    device="cuda"
    sit = load_situation(args.situation)
    sit = {k:s.to(device) for k,s in sit.items()}
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...))
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...)[346:347])
    # sit["others"].fz.v,sit["others"].fz.theta= generate_sitothers_test_vz_(sit["others"])
    add = Add_uncertainty(sit,step=2)
    uparams = {
        "dangle": torch.tensor([0.],device=device).rename(VALUESTOTEST),
        "dt0": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dt1": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "ldspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "vspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
    }

    # max_duration = max(s.t.max() for s in sit.values())
    # # t = torch.arange(start=0.,end=max_duration,step=1,device=device).rename(T)
    # t = sit["deviated"].generate_trange_conflict(step=10)
    # # print(t)
    # # raise Exception
    # masked_t = {k:s.generate_mask(t,thresh=20) for k,s in sit.items()}
    # sit_uncertainty = precompute_situation_uncertainty(sit)
    # qhulldist = QhullDist(device,n=180)

    # f_u = {k:s.add_uncertainty(uparams[k]) for k,s in sit_uncertainty.items()}
    # xy_u = {k:apply_mask(s.generate_xy(t),mask=masked_t[k])for k,s in f_u.items()}
    # tz_u = {k:apply_mask(s.generate_tz(t),mask=masked_t[k])for k,s in f_u.items()}
    # z_u = {k:apply_mask(s.generate_z(t),mask=masked_t[k])for k,s in f_u.items()}
    # diffz = distz(*minmax(z_u["others"],list(uparams["others"]["fz"].keys())),
    #               z_u["deviated"],z_u["deviated"]
    #               )
    # names_xy =list(set(uparams["deviated"]["fxy"].keys()).union(set(uparams["others"]["fxy"].keys())))
    # print(names_xy)
    # dist_xy = qhulldist.dist(xy_u["others"],xy_u["deviated"],dimsInSet=names_xy)
    dist_xy,dist_z,xy_u,z_u = add.compute_all(uparams)
    tz_u = add.compute_tz(uparams)
    print(add.compute_min_distance_xy_on_conflicting_z(uparams,thresh_z=800))
    raise Exception
    conflict_z = dist_z < 800
    conflict_xy = dist_xy < 8*1852

    if args.wpts == "xy":
        wpts_xy = {k:s.add_uncertainty(uparams[k]).fxy.compute_wpts_with_wpts0() for k,s in sit_uncertainty.items()}
        for k,wpts in wpts_xy.items():
            print(k)
            print(wpts)
            print(wpts.shape,wpts.names)
            # raise Exception
            recplot(wpts,lambda x,y:scatter_with_number(x,y,0))
            plt.show()
    elif args.wpts =="z":
        wpts_z = {k:s.add_uncertainty(uparams[k]).fz.compute_wpts_with_wpts0() for k,s in sit_uncertainty.items()}
        for k,wpts in wpts_z.items():
            if k=="others":
                print(k)
                # wpts =wpts.align_to(OTHERS,...)[45:]
                s = slice(45,46)
                s = slice(28,29)
                s = slice(29,30)
                wpts =wpts.align_to(OTHERS,...)#[s]
                # wpts =wpts.align_to(OTHERS,...)#[43:44]
                # print(wpts.names)
                # print(wpts.shape)
                #python3 add_uncertainty.py -situation ./situations/38893618_1657871463_1657872229.situation -wpts z
                recplot(wpts,lambda x,y:scatter_with_number(x,y,0))
                plt.show()
    # raise Exception
    if args.animate == "xy":
        conflict = torch.logical_and(conflict_z,conflict_xy)
        plotanimate([xy_u["deviated"],
                     # apply_mask(xy_u["others"],conflict_z/conflict_z),
                     apply_mask(xy_u["others"],(~conflict)/(~conflict)),
                     apply_mask(xy_u["others"],(~conflict_z)/(~conflict_z)),
                     apply_mask(xy_u["others"],conflict/conflict),
                     ],s=4,xlabel="x",ylabel="y")
    elif args.animate =="z":
        plotanimate(list(tz_u.values()),s=4,margin=10,xlabel="t",ylabel="z",equal=False)
if __name__ == "__main__":
    main()
#python3 add_uncertainty.py -situation ./situations/38930310_1657885467_1657885501.situation -wpts z
