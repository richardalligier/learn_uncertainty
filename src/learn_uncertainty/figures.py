import argparse
import os
import numpy as np
import torch
import pandas as pd
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import matplotlib.pyplot as plt
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config
from learn_uncertainty import read_json

from learn_uncertainty.add_uncertainty import VALUESTOTEST, Add_uncertainty

def draw_figure(sit,json,device,uparam,uncertainty_value):
    add = Add_uncertainty.from_sit_step(sit,step=5)
    uparams = {
        "dangle": torch.tensor([0.],device=device).rename(VALUESTOTEST),
        "dt0": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dt1": torch.tensor([0],device=device).rename(VALUESTOTEST),
        "dspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "ldspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
        "vspeed": torch.tensor([1.],device=device).rename(VALUESTOTEST),
    }
    uparams[uparam] = uncertainty_value
    print(add.sit_uncertainty["deviated"].sitf.tdeviation)
    print(add.sit_uncertainty["deviated"].sitf.tturn)
    print(add.sit_uncertainty["deviated"].sitf.trejoin)
    dist_xy,dist_z,xy_u,z_u = add.compute_all(uparams)
    tz_u = add.compute_tz(uparams)
    # print(add.compute_min_distance_xy_on_conflicting_z(uparams,thresh_z=800))
    # raise Exception
    # conflict_z = dist_z < 800
    # conflict_xy = dist_xy < 8*1852
    wpts_xy = {k:s.add_uncertainty(add.umodel.build_uparams(**uparams)[k]).fxy.compute_wpts_with_wpts0() for k,s in add.sit_uncertainty.items()}
    #clonedwpts_xy = {k:s.fxy.compute_wpts_with_wpts0() for k,s in clonedsit.items()}
    wpts = wpts_xy["deviated"]
    beacon_rejoin = min(json.deviated.beacons,key=lambda p:np.sum((sit["deviated"].beacon[0].numpy()-np.array([p.x,p.y]))**2))
    print(beacon_rejoin.name)
    prejoin = sit["deviated"].generate_xy(sit["deviated"].trejoin)[0,0]
    print(sit["deviated"].trejoin)
    print(prejoin)
    df = json.trajectories.query("flight_id==@json.deviated.flight_id").copy()
    df["timestampf"] = df["timestamp"]
    df["timestamp"] = [pd.Timestamp(t,unit="s") for t in df["timestamp"].values]
    print(df)
    deviated = TrafficFlight(df)
    for f in deviated.aligned_on_navpoint(json.deviated.beacons,min_distance=200,angle_precision=2):
        print(f.data[["timestamp","timestampf","navaid"]])
    # raise Exception
    print((sit["deviated"].trejoin.to(dtype=torch.int64)+sit["deviated"].tzero)[0].item())
    for b in json.deviated.beacons:
        plt.text(b.x,b.y,s=b.name)
    bx = [b.x for b in json.deviated.beacons]
    by = [b.y for b in json.deviated.beacons]
    ############ plot route
    colorbeacons = "black"
    plt.plot(bx,by,color=colorbeacons)
    plt.scatter(bx,by,marker='*',color=colorbeacons)
    ######### plot traj
    recplot(wpts,lambda x,y:scatter_with_number(x,y,None,marker="1"))
    linetstart = json.trajectories.query("timestamp==@json.deviated.start").query("flight_id==@json.deviated.flight_id")
    x,y = read_json.PROJ.transform(linetstart.longitude,linetstart.latitude)
    plt.scatter(x,y,color="red",marker="s")
    linetstop = json.trajectories.query("timestamp==@json.deviated.stop").query("flight_id==@json.deviated.flight_id")
    x,y = read_json.PROJ.transform(linetstop.longitude,linetstop.latitude)
    plt.scatter(x,y,color="blue")
    plt.scatter(prejoin[0],prejoin[1],color="blue",marker="s")
    print(sit["deviated"].beacon[0])#[0],sit["deviated"].beacon[1])
    x,y=sit["deviated"].beacon[0]
    plt.scatter(x,y,color="blue",marker="*")
#    names = [x.name for x in json.deviated.beacons]
#    print(names)
    #recplot(clonedwpts_xy[k],lambda x,y:scatter_with_number(x,y,0))
    plt.gca().set_aspect("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
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
    # print(sit)
    sit = {k:s.to(device) for k,s in sit.items()}
    #print((sit["deviated"].beacon))
    #raise Exception
    json = read_json.Situation.from_json(args.json)
    # clonedsit = {k:s.clone() for k,s in sit.items()}
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...))
    # sit["others"] = sit["others"].dmap(sit["others"],lambda v:v.align_to(OTHERS,...)[346:347])
    # sit["others"].fz.v,sit["others"].fz.theta= generate_sitothers_test_vz_(sit["others"])
    dev = np.radians(3)
    draw_figure(sit,json,device,"dangle",torch.tensor([-dev,0,dev],device=device).rename(VALUESTOTEST))
# pas mal
#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34775092_1645199121_1645199590.situation -json /disk2/jsonKimdebug/json/2201/34775092_1645199121_1645199590.json

#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34775239_1645218241_1645218640.situation -json /disk2/jsonKimdebug/json/2201/34775239_1645218241_1645218640.json
#python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34703603_1644924168_1644924694.situation -json /disk2/jsonKimdebug/json/2201/34703603_1644924168_1644924694.json


# pas conflit ou résolution par une directe:
# python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34261785_1643313515_1643313558.situation -json /disk2/jsonKimdebug/json/2201/34261785_1643313515_1643313558.json
# python3 figures.py -situation /disk2/jsonKimdebug/situations_800/2201/34327324_1643288839_1643288873.situation -json /disk2/jsonKimdebug/json/2201/34327324_1643288839_1643288873.json
# 1645199121,
# 1645199590

main()
