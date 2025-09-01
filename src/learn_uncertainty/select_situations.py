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
import graph_tool.all as gt

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

class Node:
    def __init__(self,fname,json):
        self.fname=fname
        self.json = json
    def get_id_situation(self):
        return (self.fid(),self.tstart())
    def fid(self):
        return self.json.deviated.flight_id
    def tstart(self):
        return self.json.deviated.start
    def tstop(self):
        return self.json.deviated.stop
    def __repr__(self):
        return str(self.get_id_situation())

def keep_duration_longer_than(x):
    if x == 'all':
        return None
    else:
        return float(x)

def write(rootout,node):
    # rootout = os.path.dir(node.fnameout)
    if not os.path.exists(rootout):
        os.makedirs(rootout)
    fnameout = os.path.join(rootout,os.path.basename(node.fname))
    os.symlink(node.fname, fnameout)

def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-jsonfolderin')
    parser.add_argument('-jsonfolderout')
    parser.add_argument('-tau',type=int,default=20*60)
    parser.add_argument('-keep_duration_longer_than',default='all',type=keep_duration_longer_than)
    args = parser.parse_args()
    assert(args.keep_duration_longer_than is None or args.keep_duration_longer_than>=0.)
    d = {}
    nsitutations = 0
    for root, dirs, files in tqdm.tqdm(os.walk(args.jsonfolderin, topdown=False)):
        for name in tqdm.tqdm(files):
            #print(root,dirs,files)
            fname = os.path.join(root, name)
            json = read_json.Situation.from_json(fname)
            # keep = fixed_thresh(json,args.keep_duration_longer_than)
            #print(fname,keep)
            rootout = root.replace(args.jsonfolderin,args.jsonfolderout)
            node = Node(fname,json)
            nsitutations+=1
            if node.fid() in d:
                d[node.fid()].append(node)
            else:
                d[node.fid()]=[node]
            #     #print(fnameout)
            #     if not os.path.exists(rootout):
            #         os.makedirs(rootout)
            #     os.symlink(fname,fnameout)
    g = gt.Graph(g=nsitutations,directed=True)
    edges = []
    dnumber = {}
    i=-1
    for ln in d.values():
        for n in ln:
            i+=1
            dnumber[n]=i
    # print(dnumber)
    dn={v:k for k,v in dnumber.items()}
    # def iterate(d):
    #     return list((fidi,ni) for fidi,lni in d.items() for ni in lni)
    for fidi,lni in d.items():
        for ni in lni:
            for _,line in ni.json.deviated.predicted_pairwise.iterrows():
                if line.fixed_threshold != 'false':
                    if line.flight_id in d:
                        for nj in d[line.flight_id]:
                            # print(nj)
                            if ni.tstart()<=nj.tstart()<=ni.tstart()+args.tau:
                                # print("edge")
                                edges.append((dnumber[ni],dnumber[nj]))
                        # edges.append((dnumber[i],dnumber[line.flight_id]))
            # nj = d[j]
            # if i!=j:
            #     if ni.json.deviated.start <= nj.json.deviated.start <=ni.json.deviated.start+ tau:
            #         fid = nj.json.deviated.flight_id
            #         print(type(fid))
            #         df = ni.json.deviated.predicted_pairwise.query("flight_id==@fid")
            #         if df.shape[0]>0:
            #             assert(df.shape[0]==1)
            #             line = df.iloc[0]
            #             print(line)
            #             if line.fixed_threshold != 'false':
            #                 edges.append((i,j))
    g.add_edge_list(edges)
    # print(g)
    comp, hist = gt.label_components(g, attractors=False,directed=False)
    res = {}
    for i,c in enumerate(comp.a):
        if c not in res:
            res[c]=[dn[i]]
        else:
            res[c].append(dn[i])
    # print(res)
    for c,v in res.items():
        nb_sit = len(v)
        rootout = os.path.join(args.jsonfolderout,str(nb_sit))
        writeit = any([fixed_thresh(sit.json,args.keep_duration_longer_than) for sit in v])
        if writeit:
            for sit in v:
                write(os.path.join(rootout,str(c)),sit)

main()
# python3 select_situations.py -jsonfolderin /disk2/newjson/json/2201 -jsonfolderout /disk2/newjson/test
