import os
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict, Alignment,donothing,NoAlignedAfter,NoAlignedBefore
import add_uncertainty
import torch
from torchtraj import named
from torchtraj.utils import T, XY,WPTS, apply_mask
import tqdm



def mask_on_others(k,v,mask):
    if k=="tzero":
        return v
    vnames = v.names
    # print(v)
    # print(mask)
    v = v.align_to(OTHERS,...)
    vnamesaligned = v.names
    newv = v.rename(None)[mask.rename(None)].clone()
    return newv.rename(*vnamesaligned).align_to(*vnames)

def keep_closest(sit, nclosest,thresh_z):
    add = add_uncertainty.Add_uncertainty.from_sit_step(sit,step=1)#,capmem=1000000)
    dargs = {
        "dangle": torch.tensor([0.]),
        "dt0": torch.tensor([0]),
        "dt1": torch.tensor([0]),
        "dspeed": torch.tensor([1]),
        "ldspeed": torch.tensor([1]),
        "vspeed": torch.tensor([1]),
    }
    # print(sit["deviated"].fid)
    dargs = {k:v.rename(add_uncertainty.VALUESTOTEST) for k,v in dargs.items()}
    dist_xy,dist_z,xy_u,_ = add.compute_all(dargs)
    conflict_z = dist_z < thresh_z
    # print("add.masked_t")
    # print(torch.any(add.masked_t["others"].rename(None)==1,axis=-1))
    # raise Exception
    # dist = named.nanamin(apply_mask(dist_xy,conflict_z/conflict_z),dim=(T,))
    dist = named.nanamin(dist_xy,dim=(T,))
    assert(dist.names[-1] == SITUATION)
    dist = dist[:,0]
    assert(dist.names == (OTHERS,))
    sorteddist = torch.sort(dist.rename(None)).values
    # print(sorteddist)
    # print(nclosest)
    thresh = sorteddist[min(nclosest,sorteddist.shape[0])-1]
    # print(sorteddist.names)
    # raise Exception
    thresh = min(named.nanamax(sorteddist.rename(OTHERS),dim=(OTHERS,)),thresh)
    tokeep = dist <= thresh
    # print(dist)
    # print(tokeep)
    sit["others"] = sit["others"].dmap(sit["others"],lambda k,v:mask_on_others(k,v,tokeep))
    return sit
    raise Exception
    print(dist)
    print(thresh)
    print(sorteddist)
    # print(torch.isfinite(xy_u["others"]))
    raise Exception




def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situationin')
    parser.add_argument('-situationout')
    parser.add_argument('-nclosest',type=int)
    parser.add_argument('-thresh_z',type=float)
    args = parser.parse_args()
    sit = load_situation(args.situationin)
    if isinstance(sit,Exception):
        sitout = sit
    else:
        with torch.no_grad():
            sitout = sit if args.nclosest is None else keep_closest(sit,args.nclosest,args.thresh_z)
    save_situation(sitout,args.situationout)


if __name__ == "__main__":
    main()
