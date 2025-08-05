import argparse
import os
import numpy as np
import torch
import pandas as pd
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import matplotlib.pyplot as plt
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config,Alignment,donothing,NoAlignedAfter,NoAlignedBefore
from learn_uncertainty import read_json
from torchtraj.utils import T, XY,WPTS, apply_mask
from torchtraj import named
from learn_uncertainty.add_uncertainty import VALUESTOTEST, Add_uncertainty
import matplotlib.lines as mlines


LDSPEED="ldspeed"
UNITDIST = 1852
LINEWIDTH_TARGET = 0.7
SIZE_MARKER_TPOINTS = 10
AFTER_COLOR = "black"
AFTER_MARKER = "1"
BEACON_MARKER = "2"
BEFORE_COLOR = "black"
BEFORE_MARKER = "3"
class COLOR_UNCER():
    def __init__(self):
        self.cpt = 0
        self.l= ["tab:blue","tab:orange"]
    def get(self):
        self.cpt = (self.cpt+1)%len(self.l)
        return self.l[self.cpt]
COLOR_UNCER=COLOR_UNCER()
def get_beacon(alignment,beacons):
    return min(beacons,key=lambda p:np.sum((alignment.beacon[0].numpy()-np.array([p.x,p.y]))**2))


def get_original_uparams(device):
    uparams = {
        "dangle": torch.tensor([0.],device=device).rename(VALUESTOTEST),
        "dt0": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dt1": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "ldspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "vspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
    }
    return uparams

def draw_others(sit,json,device,uparam,uncertainty_value,all_beacons=True,rotated =False,fname=None):
    add = Add_uncertainty.from_sit_step(sit,step=5)
    uparams = get_original_uparams(device)
    uparams[uparam] = uncertainty_value

    dist_xy,dist_z,dxy_u,z_u = add.compute_all(uparams)
    print(add.t)
    conflict_z = dist_z < 800
    mindist = named.nanamin(apply_mask(dist_xy,conflict_z/conflict_z),dim=(T,))
    print(mindist/UNITDIST)
    print(named.nanamin(dist_xy,dim=(T,))/UNITDIST)
    # print(sit["others"].fid)
    # for fid,mindisti in zip(mindist[:,0]/UNITDIST,sit["others"].fid[0,:]):
    #     print(fid,mindisti)
    xy_u = dxy_u["others"]
    print(xy_u.names)
    xy_u = xy_u.align_to(...,LDSPEED,T,XY)
    # raise Exception
    # tzero = sit["deviated"].tzero
    # print(sit["deviated"].trejoin)
    # tcpa = (1659347389-sit["deviated"].tzero).to(torch.float32)
    # print(tzero,tcpa)
    # pother = sit["others"].generate_xy(tcpa)
    # pdev = sit["deviated"].generate_xy(tcpa)
    # diff = pdev.align_as(pother)-pother
    # print(torch.sqrt(torch.sum(diff**2,dim=-1))/UNITDIST)
    # print(sit["others"].fid)
    # raise Exception
    # print(xy_u.names)
    # assert(xy_u.names[1]==OTHERS)
    # assert(mindist.names[1]==SITUATION)
    # print(mindist.names)
    # xy_u = xy_u.rename(None)[:,(named.nanamin(mindist,dim=(OTHERS,)).rename(None)==mindist[:,0].rename(None))].rename(*xy_u.names)
    tz_u = add.compute_tz(uparams)
    # print(add.compute_min_distance_xy_on_conflicting_z(uparams,thresh_z=800))
    # raise Exception
    # conflict_z = dist_z < 800
    # conflict_xy = dist_xy < 8*1852
    wpts_xy = {k:s.add_uncertainty(add.umodel.build_uparams(**uparams)[k]).fxy.compute_wpts_with_wpts0() for k,s in add.sit_uncertainty.items()}
    wpts_xy_orig = {k:s.add_uncertainty(add.umodel.build_uparams(**get_original_uparams(device))[k]).fxy.compute_wpts_with_wpts0() for k,s in add.sit_uncertainty.items()}
    def swap(xy):
        if rotated:
            names = xy.names
            return torch.flip(xy.rename(None),dims=[-1]).rename(*names)
        else:
            return xy
    def beacon_swap(b):
        if rotated:
            b.x,b.y = b.y,b.x
    def couple_swap(a):
        assert(len(a)==2)
        if rotated:
            return (a[1],a[0])
        else:
            return a
#     points = {
# #        "deviated": sit["deviated"].generate_xy(sit["deviated"].tdeviation)[0,0],
#         "beforedeviation": sit["others"].generate_xy(torch.arange(0,sit["deviated"].tdeviation.item()).rename(T))[0],
#         "deviated": sit["others"].generate_xy(torch.arange(sit["deviated"].tdeviation.item(),sit["deviated"].tturn.item()).rename(T))[0],
#         "rejoin": sit["others"].generate_xy(torch.arange(sit["deviated"].tturn.item(),sit["deviated"].trejoin.item()).rename(T))[0],
#         "next": sit["others"].generate_xy(torch.arange(sit["deviated"].trejoin.item(),sit["deviated"].trejoin.item()+1000).rename(T))[0],
#         "prejoin": sit["others"].generate_xy(sit["deviated"].trejoin)[0,0],
#         "pt0": sit["others"].generate_xy(sit["deviated"].tdeviation),
#     }
    t ={
        "deviated": sit["deviated"].tdeviation,
        "beforedeviation": torch.arange(0,sit["deviated"].tdeviation.item()).rename(T),
        "deviated": torch.arange(sit["deviated"].tdeviation.item(),sit["deviated"].tturn.item()).rename(T),
        "rejoin": torch.arange(sit["deviated"].tturn.item(),sit["deviated"].trejoin.item()).rename(T),
        "next": torch.arange(sit["deviated"].trejoin.item(),sit["deviated"].trejoin.item()+10).rename(T),
        "prejoin": sit["deviated"].trejoin,
        "pt0": sit["deviated"].tdeviation,
    }
    points = {k:apply_mask(sit["others"].generate_xy(ti),sit["others"].generate_mask(ti,thresh=20)) for k,ti in t.items()}
    wpts = swap(wpts_xy["deviated"])
    wpts_orig = swap(wpts_xy_orig["deviated"])
    points = {k:swap(v) for k,v in points.items()}
    xy_u = swap(xy_u)
    fig = plt.figure()
    ######### plot traj
    #recplot(xy_u["deviated"],lambda x,y:plt.plot(x,y))#scatter_with_number(x,y,None,marker="1"))
    #recplot(wpts_orig,lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color="black"))#scatter_with_number(x,y,None,marker="1"))
    # print(sit["deviated"].generate_xy(sit["deviated"].trejoin).names)
    # raise Exception
    recplot(points["beforedeviation"],lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color="black"))
    recplot(points["deviated"],lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color="black"))
    recplot(points["rejoin"],lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color="black"))
    line=recplot(points["next"],lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color="black"))
    line[0][0].set_label("actual trajectory")
    lines=recplot(xy_u,lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color=COLOR_UNCER.get()))
    lines[0][0].set_label("modified trajectory")
    lines[1][0].set_label("modified trajectory")
    line=recplot(points["pt0"],lambda x,y:plt.scatter(x/UNITDIST,y/UNITDIST,color="black",s=SIZE_MARKER_TPOINTS))
    line[0].set_label("position at $t_0$")
    print(xy_u.shape)
    def simplify(xy):
        tosqueeze = [i for i,(name,s) in enumerate(zip(xy.names,xy.shape)) if name not in [T,XY] and s==1]
        newnames = [name for i,name in enumerate(xy.names) if i not in tosqueeze]
        return torch.squeeze(xy.rename(None),dim=tosqueeze).rename(*newnames)
    sxy_u = simplify(xy_u)
    print(sxy_u.shape,sxy_u.names)
    plt.gca().set_aspect("equal")
    plt.setp(plt.gca().get_xticklabels(), rotation=0, va="top", ha="center")
    plt.setp(plt.gca().get_yticklabels(), rotation=0, va="center", ha="right")
    xlab = "x [NM]"
    ylab = "y [NM]"
    xlab,ylab = couple_swap((xlab,ylab))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # axes = plt.gca()
    # lafter = mlines.Line2D([], [], color=AFTER_COLOR, marker=AFTER_MARKER, label='target beacon after deviation')
    # plt.gca().legend(handles=[lafter])
    plt.gca().xaxis.set_inverted(True)
    plt.legend(ncol=1,
               loc='upper left',
               frameon=False,
               columnspacing=0.1,
               fontsize=8,
               )
    if fname is not None:
        fig.set_tight_layout({'pad':0})
        fig.set_figwidth(8)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situation')
    parser.add_argument('-json')
    args = parser.parse_args()
    # print(args.json)
    device="cpu"
    sit=load_situation(args.situation)
    if isinstance(sit,Exception):
        raise sit
    # print(sit)
    # config = read_config()
    # fname = os.path.join(config.FOLDER,"all_800_10_1800_2.dsituation")
    # #fname = "/disk2/jsonKimdebugBeacons/situations_800_120_120_10_1800/2201/34330127_1643280923_1643281399.situation"
    # DSITUATION = load_situation(fname)
    # print(list(DSITUATION.keys()))
    # #DSITUATION = {10:DSITUATION[10][:400]}
    # # DSITUATION = {10:DSITUATION[10][2876:2975]}
    # sit = DSITUATION[10][2876]
    # sit = {k:s.to(device) for k,s in sit.items()}
    print(sit["deviated"].tzero)
    print(sit["deviated"].tdeviation)
#    print(sit["deviated"].tzero+sit["deviated"].tdeviation)
    # raise Exception
    dev = np.radians(5)
    # draw_figure(sit,json,device,"dt1",torch.tensor([-60,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    # draw_figure(sit,json,device,"dt0",torch.tensor([-30,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    json = read_json.Situation.from_json(args.json)
    draw_others(sit,json,device,"ldspeed",torch.tensor([1.2,0.8],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True,fname="./figures/ldspeed.pdf")

main()
