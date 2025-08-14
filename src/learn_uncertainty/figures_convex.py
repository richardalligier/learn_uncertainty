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
from learn_uncertainty.add_uncertainty import VALUESTOTEST, Add_uncertainty
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull

UNITDIST = 1852
LINEWIDTH_TARGET = 0.7
LINEWIDTH_UN = 0.3
LINEWIDTH_CV = 1
SIZE_MARKER_TPOINTS = 10
SIZE_MARKER_UN = 3
AFTER_COLOR = "black"
AFTER_MARKER = "1"
BEACON_MARKER = "2"
BEFORE_COLOR = "black"
BEFORE_MARKER = "3"
rotated = True
class COLOR_UNCER():
    def __init__(self):
        self.cpt = 0
        self.l= ["tab:blue"]#,"tab:orange"]
    def get(self):
        self.cpt = (self.cpt+1)%len(self.l)
        return self.l[self.cpt]
COLOR_UNCER=COLOR_UNCER()
def get_beacon(alignment,beacons):
    return min(beacons,key=lambda p:np.sum((alignment.beacon[0].numpy()-np.array([p.x,p.y]))**2))


def plot_convex_hull(points, **kwargs):
    points = np.array(points)
    points = points.reshape(-1,2)
    if len(points) < 3:
        print("Convex hull requires at least 3 points.")
        return
    # Compute the convex hull
    hull = ConvexHull(points)

    # Plot the convex hull edges
    for simplex in hull.simplices:
        res=plt.plot(points[simplex, 0], points[simplex, 1], **kwargs)
    return res

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
def context(sit,json,device,duparams,all_beacons):
    add = Add_uncertainty.from_sit_step(sit,step=5)
    # print(add.t.min())
    # raise Exception
    uparams = duparams
    #print(add.sit_uncertainty["deviated"].sitf.tdeviation)
    #print(add.sit_uncertainty["deviated"].sitf.tturn)
    #print(add.sit_uncertainty["deviated"].sitf.trejoin)
    # uparams ={
    #     "dangle":torch.tensor([-0.029244276039794515,0.037060166120925685],device=device).rename(VALUESTOTEST),
    #     "dt0": torch.tensor([-24.545250983786982,11.00427059120603],device=device).rename(VALUESTOTEST),
    #     "dt1": torch.tensor([-20.872732711213867,31.485385897934272],device=device).rename(VALUESTOTEST),
    #     "dspeed": torch.tensor([0.8863890037284232,1.0582458280396083],device=device).rename(VALUESTOTEST),
    #     "ldspeed": torch.tensor([0.9223705789444759,1.0278987721304083],device=device).rename(VALUESTOTEST),
    #     "vspeed": torch.tensor([0.8584289297070437,1.0732723686587384],device=device).rename(VALUESTOTEST),
    # }
    dist_xy,dist_z,dxy_u,z_u = add.compute_all(uparams)
    # raise Exception
    xy_u = dxy_u["deviated"]
    tz_u = add.compute_tz(uparams)
    # print(add.compute_min_distance_xy_on_conflicting_z(uparams,thresh_z=800))
    # raise Exception
    # conflict_z = dist_z < 800
    # conflict_xy = dist_xy < 8*1852
    wpts_xy = {k:s.add_uncertainty(add.umodel.build_uparams(**uparams)[k]).fxy.compute_wpts_with_wpts0() for k,s in add.sit_uncertainty.items()}
    wpts_xy_orig = {k:s.add_uncertainty(add.umodel.build_uparams(**get_original_uparams(device))[k]).fxy.compute_wpts_with_wpts0() for k,s in add.sit_uncertainty.items()}
    if all_beacons:
        beacon_before = json.deviated.beacons[0]
        beacon_after = json.deviated.beacons[-1]
    else:
        beacon_before = get_beacon(sit["deviated"].align_before,json.deviated.beacons)
        beacon_after = get_beacon(sit["deviated"].align_after,json.deviated.beacons)
    istart = max(json.deviated.beacons.index(beacon_before)-3,0)
    iend = min(json.deviated.beacons.index(beacon_after)+2,len(json.deviated.beacons)-1)
    beacons=json.deviated.beacons[istart:iend+1]
    for b in beacons:
        beacon_swap(b)
    points = {
#        "deviated": sit["deviated"].generate_xy(sit["deviated"].tdeviation)[0,0],
        "beforedeviation": sit["deviated"].generate_xy(torch.arange(0,sit["deviated"].tdeviation.item()).rename(T))[0],
        "deviated": sit["deviated"].generate_xy(torch.arange(sit["deviated"].tdeviation.item(),sit["deviated"].tturn.item()).rename(T))[0],
        "rejoin": sit["deviated"].generate_xy(torch.arange(sit["deviated"].tturn.item(),sit["deviated"].trejoin.item()).rename(T))[0],
        "next": sit["deviated"].generate_xy(torch.arange(sit["deviated"].trejoin.item(),sit["deviated"].trejoin.item()+1000).rename(T))[0],
        "prejoin": sit["deviated"].generate_xy(sit["deviated"].trejoin)[0,0],
        "pt1": sit["deviated"].generate_xy(sit["deviated"].tturn-60)[0,0],
    }
    wpts = swap(wpts_xy["deviated"])
    wpts_orig = swap(wpts_xy_orig["deviated"])
    points = {k:swap(v) for k,v in points.items()}
    xy_u = swap(xy_u)
    bx = [b.x/UNITDIST for b in beacons]
    by = [b.y/UNITDIST for b in beacons]
    fig = plt.figure()
    for bxi,byi,bname in zip(bx,by,[b.name for b in beacons]):
        plt.text(bxi,byi,s=bname,rotation=0,horizontalalignment='left',verticalalignment='bottom')
    ############ plot route
    colorbeacons = "black"
    line,=plt.plot(bx,by,color=colorbeacons,linestyle="--")
    line.set_label("route")
    line = plt.scatter(bx,by,marker=BEACON_MARKER,color=colorbeacons)
    line.set_label("beacon")
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

    plt.plot([points["prejoin"][0]/UNITDIST,beacon_after.x/UNITDIST],[points["prejoin"][1]/UNITDIST,beacon_after.y/UNITDIST],color="black",linestyle="--",linewidth=LINEWIDTH_TARGET)
    print(xy_u.shape)
    linetstart = json.trajectories.query("timestamp==@json.deviated.start").query("flight_id==@json.deviated.flight_id")
    x,y = couple_swap(read_json.PROJ.transform(linetstart.longitude,linetstart.latitude))
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=BEFORE_COLOR,marker="s",s=SIZE_MARKER_TPOINTS)
#    line.set_label("end of alignment befor deviation")
    line,=plt.plot([points["beforedeviation"][-1,0]/UNITDIST,beacon_before.x/UNITDIST],[points["beforedeviation"][-1,1]/UNITDIST,beacon_before.y/UNITDIST],color=BEFORE_COLOR,linestyle="--",linewidth=LINEWIDTH_TARGET)
    line.set_label("alignment")
    x,y=beacon_before.x,beacon_before.y
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=BEFORE_COLOR,marker=BEFORE_MARKER)
    line.set_label("target beacon before deviation")
    linetstop = json.trajectories.query("timestamp==@json.deviated.stop").query("flight_id==@json.deviated.flight_id")
    x,y = couple_swap(read_json.PROJ.transform(linetstop.longitude,linetstop.latitude))
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=AFTER_COLOR,s=SIZE_MARKER_TPOINTS)
    line.set_label("start of alignment")
    #print(sit["deviated"].beacon[0])#[0],sit["deviated"].beacon[1])
    line=plt.scatter(*(points["prejoin"]/UNITDIST),color=AFTER_COLOR,marker="s",s=SIZE_MARKER_TPOINTS)
    line.set_label("end of alignment")

    x,y=couple_swap(sit["deviated"].beacon[0])
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=AFTER_COLOR,marker=AFTER_MARKER)
    line.set_label("target beacon after deviation")
    return fig,xy_u

def finalise(fig,fname):
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
    plt.legend(ncol=2,
               loc='upper right',
               frameon=False,
               columnspacing=0.1,
               fontsize=8,
               )
    if fname is not None:
        fig.set_tight_layout({'pad':0})
        fig.set_figwidth(8)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def draw_convex(sit,json,device,duparams,all_beacons=True,fname=None):
    fig,xy_u = context(sit,json,device,duparams,all_beacons)
    lines=recplot(xy_u,lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST,color=COLOR_UNCER.get(),linewidth=LINEWIDTH_UN))
    lines[0][0].set_label("modified trajectory")
    it = 100
    lines=recplot(xy_u[...,it,:],lambda x,y:plt.scatter(x/UNITDIST,y/UNITDIST,color="black",s=SIZE_MARKER_UN))
    lines[0].set_label(f"positions at t={it*5}s")
    lines = plot_convex_hull(xy_u[...,it,:]/UNITDIST,color="red",linewidth=LINEWIDTH_CV)
    lines[0].set_label(f"convex hull at t={it*5}s")
    finalise(fig,fname)

def draw_convexseq(sit,json,device,duparams,all_beacons=True,fname=None):
    fig,xy_u = context(sit,json,device,duparams,all_beacons)
    for it in range(0,171,10):
        lines = plot_convex_hull(xy_u[...,it,:]/UNITDIST,color="red",linewidth=LINEWIDTH_CV)
    lines[0].set_label(f"convex hulls")
    finalise(fig,fname)
    
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
    duparams = {'dangle': torch.tensor([-0.0798,  0.0798]),
                'dt0':    torch.tensor([-24.3474,   20.7026]),
                'dt1':    torch.tensor([-29.6732,   20.0920]),
                'dspeed': torch.tensor([0.9, 1.1]),
                'ldspeed':torch.tensor([0.9, 1.1]),
                'vspeed': torch.tensor([0.8628, 1.1059])
                }
    duparams = {k:v.rename(VALUESTOTEST).to(device) for k,v in duparams.items()}
    print(sit["deviated"].tzero)
    print(sit["deviated"].tdeviation)
#    print(sit["deviated"].tzero+sit["deviated"].tdeviation)
    # raise Exception
    dev = np.radians(5)
    # draw_figure(sit,json,device,"dt1",torch.tensor([-60,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    # draw_figure(sit,json,device,"dt0",torch.tensor([-30,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    json = read_json.Situation.from_json(args.json)
    draw_convex(sit,json,device,duparams,all_beacons=False,fname="./figures/convex.pdf")
    json = read_json.Situation.from_json(args.json)
    draw_convexseq(sit,json,device,duparams,all_beacons=False,fname="./figures/convexseq.pdf")

if __name__ == "__main__":
    main()
#explore()
# pas mal
#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34775092_1645199121_1645199590.situation -json /disk2/jsonKimdebug/json/2201/34775092_1645199121_1645199590.json

#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34775239_1645218241_1645218640.situation -json /disk2/jsonKimdebug/json/2201/34775239_1645218241_1645218640.json
#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34703603_1644924168_1644924694.situation -json /disk2/jsonKimdebug/json/2201/34703603_1644924168_1644924694.json

#python3 figures.py -situation /disk2/jsonKimdebugBeacons/situations_800/2201/34703603_1644924168_1644924694.situation -json /disk2/jsonKimdebugBeacons/json/2201/34703603_1644924168_1644924694.json

# double ?? python3 figures.py -situation /disk2/jsonKimdebugBeacons/situations_800/2201/34366247_1643434913_1643435388.situation -json /disk2/jsonKimdebugBeacons/json_selected/2201/34366247_1643434913_1643435388.json


# pas conflit ou résolution par une directe:
# python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34261785_1643313515_1643313558.situation -json /disk2/jsonKimdebug/json/2201/34261785_1643313515_1643313558.json
# python3 figures.py -situation /disk2/jsonKimdebugBeacons/situations_800/2201/34327324_1643288839_1643288873.situation -json /disk2/jsonKimdebugBeacons/json/2201/34327324_1643288839_1643288873.json
# 1645199121,
# 1645199590

#main()
