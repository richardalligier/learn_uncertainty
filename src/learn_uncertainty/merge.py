import os
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict, Alignment,donothing,NoAlignedAfter,NoAlignedBefore,TcpaAfterEndofAlignmentAfter
import add_uncertainty
import torch
from torchtraj import named
from torchtraj.utils import T, XY,WPTS, apply_mask
import tqdm

def merge(lsit):
    d = {k:[s[k] for s in lsit] for k in lsit[0]}
    for k,v in d.items():
        # print(v)
        if len(v)==1:
            d[k] = v[0]
        else:
            d[k] = type(v[0]).cat(v,dimname=SITUATION)
    return d

def partition(criteria,l,nmax):
    d = {}
    for x in  tqdm.tqdm(l):
        k = criteria(x)
        if k in d:
            if len(d[k][-1])<nmax:
                d[k][-1].append(x)
            else:
                d[k].append([x])
        else:
            d[k]=[[x]]
    return d


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situationsfolder')
    parser.add_argument('-dsituation')
    parser.add_argument('-nsituation_max',type=int,default=100000)
    # parser.add_argument('-nclosest',type=int)
    # parser.add_argument('-thresh_z',type=float)
    args = parser.parse_args()
    lfname=[]
    for root, dirs, files in os.walk(args.situationsfolder, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            lfname.append(fname)
            # print(fname)
    lsitraw = [load_situation(fname) for fname in lfname]
    print(f"{len(lsitraw)=}")
    # print({type(k) for k in lsitraw})
    lsitraw = [sit for sit in lsitraw if not isinstance(sit,Exception)]
    # print({type(k) for k in lsitraw})
    # print({type(k["others"]) for k in lsitraw})
    # print({type(k["deviated"]) for k in lsitraw})
    print(f"{len(lsitraw)=}")
    with torch.no_grad():
        lsit = lsitraw# if args.nclosest is None else [keep_closest(sit,args.nclosest,args.thresh_z) for sit in tqdm.tqdm(lsitraw)]
    dlsit = partition(lambda sit: sit["others"].t.shape[sit["others"].t.names.index(OTHERS)],lsit,args.nsituation_max)
    print(sorted([(k,len(v)) for k,v in dlsit.items()]))
    # print(dlsit[355][0]["deviated"].fid)
    # print(dlsit[355][0]["deviated"].tdeviation)
    # print(dlsit[355][0]["others"].fid)
    for k,v in tqdm.tqdm(dlsit.items()):
        # print(k)
        # if len(v)==1:
        #     print("merged")
        #     dlsit[k]=v
        # else:
        dlsit[k]=[merge(vi) for vi in tqdm.tqdm(v)]
    # print(dlsit)
    save_situation(dlsit,args.dsituation)


if __name__ == "__main__":
    main()
