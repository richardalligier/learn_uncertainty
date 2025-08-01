import argparse
import os
import numpy as np
import torch
import pandas as pd
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import matplotlib.pyplot as plt
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config,Alignment
from learn_uncertainty import read_json
import tqdm


def fixed_thresh(json,keep_duration_longer_than):
    dmin_hat = json.deviated.predicted_pairwise.lateral_dist_at_tcpa.min()
    dmin = json.deviated.actual_pairwise.lateral_dist_at_tcpa.min()
    epsilon = dmin - dmin_hat
    imin = json.deviated.predicted_pairwise.lateral_dist_at_tcpa.argmin()
    tcpa_dmin_hat = json.deviated.predicted_pairwise.time_at_cpa.iloc[imin] - json.deviated.start
    deviationduration = json.deviated.stop - json.deviated.start
    keep = epsilon > 0.1 and dmin_hat < 8 and tcpa_dmin_hat > 50
    if keep_duration_longer_than is None:
        return keep
    else:
        return keep and deviationduration >= keep_duration_longer_than


def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-jsonfolderin')
    parser.add_argument('-jsonfolderout')
    parser.add_argument('-keep_duration_longer_than',default="all",type=float)
    args = parser.parse_args()
    if args.keep_duration_longer_than == 'all':
        keep_duration_longer_than = None
    assert(args.keep_duration_longer_than is None or args.keep_duration_longer_than>=0.)
    for root, dirs, files in tqdm.tqdm(os.walk(args.jsonfolderin, topdown=False)):
        for name in files:
            #print(root,dirs,files)
            fname = os.path.join(root, name)
            json = read_json.Situation.from_json(fname)
            keep = fixed_thresh(json,args.keep_duration_longer_than)
            #print(fname,keep)
            if keep:
                rootout = root.replace(args.jsonfolderin,args.jsonfolderout)
                fnameout = os.path.join(rootout,name)
                #print(fnameout)
                if not os.path.exists(rootout):
                    os.makedirs(rootout)
                os.symlink(fname,fnameout)

main()
