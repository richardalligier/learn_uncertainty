import os
import fit_traj
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict
import torch

def merge(lsit):
    d = {k:[s[k] for s in lsit] for k in lsit[0]}
    for k,v in d.items():
        d[k] = type(v[0]).cat(v,dimname=SITUATION)
    return d

def partition(criteria,l):
    d = {}
    for x in l:
        k = criteria(x)
        if k in d:
            d[k].append(x)
        else:
            d[k]=[x]
    return d

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument( '-foldersituations')
    parser.add_argument('-dsituation')
    args = parser.parse_args()
    dsit = load_situation(args.dsituation)

    what = ["deviated","others"]
    for k,dsitk in dsit.items():
        t = dsitk["deviated"].generate_trange_conflict(step=1)
        dsitkxy = [dsitk[w].generate_xy(t) for w in what]
        dsitkz = [dsitk[w].generate_z(t) for w in what]
        toiterate = [getattr(dsitk["deviated"],x) for x in ["fid","tzero","tdeviation","tturn"]]+dsitkxy+dsitkz+[t]
        print(dsitkxy[0].shape)
        for fid,tzero,tdeviation,tturn,xyrd,xyro,zrd,zro,ti in zip(*toiterate):
            fname = f"{fid.item()}_{round(tzero.item()+tdeviation.item())}_{round(tzero.item()+tturn.item())}.situation"
            print(fname)
            s = load_situation(os.path.join(args.foldersituations,fname))
            xyso = s["others"].generate_xy(ti)
            xysd = s["deviated"].generate_xy(ti)
            assert(torch.abs(xysd-xyrd).max()<0.1)
            assert(torch.abs(xyso-xyro).max()<0.1)
            zso = s["others"].generate_z(ti)
            zsd = s["deviated"].generate_z(ti)
            assert(torch.abs(zsd-zrd).max()<0.1)
            assert(torch.abs(zso-zro).max()<0.1)
            # print((xyso==xyro).rename(None).all())
    #         raise Exception
    # raise Exception
    # print(sit["deviated"].tzero)
    # print(sit["deviated"].tdeviation)
    # print(sit["deviated"].trejoin)
    # raise Exception
    # nothers = sit["others"].fxy.v.shape[sit["others"].fxy.v.names.index(OTHERS)]
    # s = dsit[nothers]
    # print(s["deviated"].fid)
    # print(sit["deviated"].fid)
    # print(nothers)
    # max_duration = (sit["deviated"].trejoin - sit["deviated"].tdeviation).max().item()
    # t = torch.arange(start=0,end=max_duration,dtype=torch.float32).rename("t")
    # print(sit["deviated"].t)
    # print(sit["deviated"].tdeviation)
    # sitxy = fit_traj.masked_generate(sit["deviated"].fxy,t,sit["deviated"].t)
    # sxy = fit_traj.masked_generate(s["deviated"].fxy,t,s["deviated"].t)
    # select = (s["deviated"].fid == sit["deviated"].fid.item())
    # iselect = [ i for i,x in enumerate(select) if x]
    # assert(len(iselect)==1)
    # i=iselect[0]
    # print((sxy[i]==sitxy).all())


if __name__ == "__main__":
    main()
