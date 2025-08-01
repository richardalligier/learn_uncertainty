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

UNITDIST = 1852
LINEWIDTH_TARGET = 0.7
SIZE_MARKER_TPOINTS = 20
AFTER_COLOR = "blue"
AFTER_MARKER = "+"
BEFORE_COLOR = "red"
BEFORE_MARKER = "x"

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
def draw_figure(sit,json,device,uparam,uncertainty_value,all_beacons=True,rotated =False,fname=None):
    add = Add_uncertainty.from_sit_step(sit,step=5)
    # print(add.t.min())
    # raise Exception
    uparams = get_original_uparams(device)
    uparams[uparam] = uncertainty_value
    #print(add.sit_uncertainty["deviated"].sitf.tdeviation)
    #print(add.sit_uncertainty["deviated"].sitf.tturn)
    #print(add.sit_uncertainty["deviated"].sitf.trejoin)
    dist_xy,dist_z,dxy_u,z_u = add.compute_all(uparams)
    xy_u = dxy_u["deviated"]
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
    }
    wpts = swap(wpts_xy["deviated"])
    wpts_orig = swap(wpts_xy_orig["deviated"])
    points = {k:swap(v) for k,v in points.items()}
    xy_u = swap(xy_u)
    bx = [b.x/UNITDIST for b in beacons]
    by = [b.y/UNITDIST for b in beacons]
    fig = plt.figure()
    for bxi,byi,bname in zip(bx,by,[b.name for b in beacons]):
        plt.text(bxi,byi,s=bname,rotation=0,horizontalalignment='right')
    ############ plot route
    colorbeacons = "black"
    line,=plt.plot(bx,by,color=colorbeacons,linestyle="--")
    line.set_label("route")
    line = plt.scatter(bx,by,marker='.',color=colorbeacons)
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
    lines=recplot(xy_u,lambda x,y:plt.plot(x/UNITDIST,y/UNITDIST))
    for line in lines:
        print(line)
        line[0].set_label("trajectory with uncertainty")
    # plt.plot([points["prejoin"][0]/UNITDIST,beacon_after.x/UNITDIST],[points["prejoin"][1]/UNITDIST,beacon_after.y/UNITDIST],color="blue",linestyle="--",linewidth=LINEWIDTH_TARGET)
    print(xy_u.shape)
    def simplify(xy):
        tosqueeze = [i for i,(name,s) in enumerate(zip(xy.names,xy.shape)) if name not in [T,XY] and s==1]
        newnames = [name for i,name in enumerate(xy.names) if i not in tosqueeze]
        return torch.squeeze(xy.rename(None),dim=tosqueeze).rename(*newnames)
    sxy_u = simplify(xy_u)
    print(sxy_u.shape,sxy_u.names)
    linetstart = json.trajectories.query("timestamp==@json.deviated.start").query("flight_id==@json.deviated.flight_id")
    x,y = couple_swap(read_json.PROJ.transform(linetstart.longitude,linetstart.latitude))
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=BEFORE_COLOR,marker="s",s=SIZE_MARKER_TPOINTS)
    line.set_label("end of alignment before deviation")
    plt.text(x/UNITDIST,y/UNITDIST,s="tstartdeviation")
    line,=plt.plot([points["beforedeviation"][-1,0]/UNITDIST,beacon_before.x/UNITDIST],[points["beforedeviation"][-1,1]/UNITDIST,beacon_before.y/UNITDIST],color=BEFORE_COLOR,linestyle="--",linewidth=LINEWIDTH_TARGET)
    line.set_label("alignment before deviation")
    x,y=beacon_before.x,beacon_before.y
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=BEFORE_COLOR,marker=BEFORE_MARKER)
    line.set_label("target beacon before deviation")
    linetstop = json.trajectories.query("timestamp==@json.deviated.stop").query("flight_id==@json.deviated.flight_id")
    x,y = couple_swap(read_json.PROJ.transform(linetstop.longitude,linetstop.latitude))
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=AFTER_COLOR,s=SIZE_MARKER_TPOINTS)
    line.set_label("start of alignment after deviation")
    #print(sit["deviated"].beacon[0])#[0],sit["deviated"].beacon[1])
    line=plt.scatter(*(points["prejoin"]/UNITDIST),color=AFTER_COLOR,marker="s",s=SIZE_MARKER_TPOINTS)
    line.set_label("end of alignment after deviation")
    for i in range(sxy_u.shape[0]):
        line,=plt.plot([sxy_u[i,-1,0]/UNITDIST,beacon_after.x/UNITDIST],[sxy_u[i,-1,1]/UNITDIST,beacon_after.y/UNITDIST],color=AFTER_COLOR,linestyle="--",linewidth=LINEWIDTH_TARGET)
    line.set_label("alignment after deviation")
    x,y=couple_swap(sit["deviated"].beacon[0])
    line=plt.scatter(x/UNITDIST,y/UNITDIST,color=AFTER_COLOR,marker=AFTER_MARKER)
    line.set_label("target beacon after deviation")
#    names = [x.name for x in json.deviated.beacons]
#    print(names)
    #recplot(clonedwpts_xy[k],lambda x,y:scatter_with_number(x,y,0))
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
    plt.legend(ncol=2,
               loc='upper left',
               frameon=False,
               columnspacing=0.1,
               )
    if fname is not None:
        fig.set_tight_layout({'pad':0})
        fig.set_figwidth(10)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    # print(dir(axes))
    # for line in axes.lines:
    #     newx = line.get_ydata()
    #     newy = line.get_xdata()
    #     line.set_xdata(newx)
    #     line.set_ydata(newy)
    # for line in axes.scatter:
    #     newx = line.get_ydata()
    #     newy = line.get_xdata()
    #     line.set_xdata(newx)
    #     line.set_ydata(newy)
    # print(json.deviated.predicted_pairwise)
    # print(json.deviated.actual_pairwise)
    # print(json.deviated.stop-json.deviated.start)
    # plt.gca().view_init(30,90)


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
    sit = {k:s.to(device) for k,s in sit.items()}
    #print((sit["deviated"].beacon))
    #raise Exception
    json = read_json.Situation.from_json(args.json)
    # clonedsit = {k:s.clone() for k,s in sit.items()}
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...))
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...)[346:347])
    # sit["others"].fz.v,sit["others"].fz.theta= generate_sitothers_test_vz_(sit["others"])
    # dev = np.radians(5);draw_figure(sit,json,device,"dangle",torch.tensor([-dev,dev],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    # draw_figure(sit,json,device,"dt1",torch.tensor([-60,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    # draw_figure(sit,json,device,"dt0",torch.tensor([-30,60],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True)
    draw_figure(sit,json,device,"dspeed",torch.tensor([0.8,1.1],device=device).rename(VALUESTOTEST),all_beacons=False,rotated=True,fname="./figures/dspeed.pdf")

def explore():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situation')
    parser.add_argument('-json')
    args = parser.parse_args()
    # print(args.json)
    dcheck={
        "34533465_1644226719_1644227194":"GUERE",
        "34858257_1645556761_1645557369":"TIVLI",
        "34455456_1643880051_1643880698":"MADOT",
        "34867253_1645615200_1645615336":"BAMES",
        "34430466_1643727915_1643728472":"DGO",
        "34577914_1644390010_1644390297":"UVELI",
        "34815134_1645358449_1645358876":"MEN",
        "34802451_1645347304_1645347868":"URUNA",
        "34617980_1644595154_1644595390":"LOMRA",
        "34809153_1645368809_1645369407":"LERGA",
        "34780675_1645274866_1645275124":"AGN",
        "34408006_1643636013_1643636312":"GALOF",
        "34464319_1643890091_1643890543":"BAMES",
        "34437146_1643795116_1643795382":"ALBER",
        #"34608964_1644590255_1644590841":"MEN",
        "34456724_1643880678_1643881146":"BAMES",
        "34366565_1643450878_1643451265":"PPN",
        "34775092_1645199121_1645199590":"GALOF",
        "34789216_1645268590_1645269053":"TIVLI",
        "34851691_1645527328_1645528122":"TIVLI",
        "34684111_1644844564_1644845106":"TIVLI",
        "34467197_1643900767_1643901213":"BOKNO",
        "34767724_1645269534_1645269991":"LUKEV",
        "34681270_1644835801_1644835968":"PINAR",
        "34442449_1643800855_1643801580":"ROCAN",
        "34596740_1644497117_1644497248":"ANETO",
        "34799233_1645294240_1645294968":"AMB",
        "34555203_1644299413_1644299741":"ANETO",
        "34616951_1644567834_1644568408":"PPN",
        "34640766_1644650619_1644650989":"ALBER",
        "34814080_1645385806_1645387253":"ANG",
        "34540452_1644255913_1644256179":"TIVLI",
        "34642066_1644665879_1644666078":"DIKRO",
        "34477130_1643980821_1643981287":"AMB",
        "34684270_1644857689_1644858156":"OSMOB",
        "34703603_1644924168_1644924694":"AKIKI",
        "34815201_1645377088_1645377713":"ANETO",
        "34774398_1645197342_1645197662":"AGN",
        "34786636_1645255895_1645256203":"PPN",
        "34780251_1645260168_1645260504":"TSU",
        "34516026_1644130547_1644130985":"UVELI",
        "34853094_1645547765_1645548349":"AGN",
        "34462440_1643888129_1643888329":"KANIG",
        "34499780_1644058748_1644059384":"TIVLI",

    }
    device="cpu"
    for root, dirs, files in os.walk(args.situation, topdown=False):
        for name in files:
            fnamesit = os.path.join(root, name)
            rootjson= root.replace(args.situation,args.json)
            fnamejson = os.path.join(rootjson,os.path.splitext(name)[0])+".json"
            #print(fnamesit)
            #print(fnamejson)
            sit=load_situation(fnamesit)
            if isinstance(sit,Exception):
                continue
            json = read_json.Situation.from_json(fnamejson)
            idname = name[:-len(".situation")]
            sit = {k:s.to(device) for k,s in sit.items()}
            beac = get_beacon(sit["deviated"].align_after,json.deviated.beacons)
            #print(beac.name)
            if idname in dcheck:
                if (dcheck[idname]!=beac.name):
                    print(idname,beac.name,dcheck[idname])
                    dev = np.radians(5)
                    draw_figure(sit,json,device,"dangle",torch.tensor([-dev,0,dev],device=device).rename(VALUESTOTEST))
            else:
                print(f'"{idname}":"{beac.name}",')
                dev = np.radians(5)
                draw_figure(sit,json,device,"dangle",torch.tensor([-dev,0,dev],device=device).rename(VALUESTOTEST))

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
