import os
import fit_traj
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS,WPTS, deserialize_dict
import torch
from torchtraj import qhull,named,uncertainty


THRESHT = 20 # seconds
THRESHALT = 800 # feet
DANGLE = "dangle"
MAN_WPTS = "man_wpts"
DT = "dt1"
DSPEED = "dspeed"
LDSPEED = "ldspeed"
PARAMS = "params"

#DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_distance_function(dsituation,step):
    lsit = [{k:v.to(DEVICE) for k,v in s.items()} for s in dsituation.values()]#[:1]
    tv = ["tdeviation","tturn","trejoin"]
    for s in lsit:
        for t in tv:
            s["others"].fxy=s["others"].fxy.add_wpt_at_t(named.unsqueeze(getattr(s["deviated"],t),-1,WPTS))
    # raise Exception
    # sit=lsit[0]
    # print(sit["deviated"].fid)
    # print(sit["others"].fid)
    # raise Exception
    lt = [s["deviated"].generate_trange_conflict(step=step) for s in lsit]
    def compute_iwpts(s,k):
        return {t:s[k].fxy.wpts_at_t(named.unsqueeze(getattr(s["deviated"],t),-1,WPTS)) for t in tv}
    liwpts = [{k:compute_iwpts(s,k) for k in s.keys()} for s in lsit]
    # print(lt[0])
    # raise Exception
    def compute_zmask(fothers,fdeviated,t):
        mothers = fothers.generate_z(t)
        dz =  mothers - fdeviated.generate_z(t).align_as(mothers)
        mask  = torch.abs(dz) < THRESHALT
        res = mask/mask
        # print(torch.abs(dz))
        # print(res)
        # raise Exception
        return res
    # def compute_tmask(fothers,fdeviated,t):
    #     mothers = fothers.generate_mask(t,THRESHT)
    #     return mothers * fdeviated.generate_mask(t,THRESHT).align_as(mothers)
    ltmask = [ {k:v.generate_mask(t,THRESHT) for k,v in s.items()} for (s,t) in zip(lsit,lt)]
    lzmask = [ compute_zmask(s["others"],s["deviated"],t) for (s,t) in zip(lsit,lt)]
    lmask = [ {k:v.align_as(zmask)*zmask for k,v in tmask.items()} for (tmask,zmask) in zip(ltmask,lzmask)]
    qhulldist = qhull.QhullDist(DEVICE,n=3000)
    def compute_distxy(params,fothers,fdeviated,t,mask,iwpts):
        fdeviated = fdeviated.clone()
        ldspeed = params[:,:2].rename(...,LDSPEED)
        fothers.fxy = uncertainty.change_longitudinal_speed(ldspeed,iwpts["others"]["tdeviation"],iwpts["others"]["trejoin"],fothers.fxy)
        others = fothers.generate_xy(t)
        others = (others * mask["others"].align_as(others))
        deviated = fdeviated.generate_xy(t).align_as(others)
        deviated = deviated * (mask["deviated"]).align_as(deviated)
        # raise Exception#[:,:1]
        dimsInSet = tuple()
        #return named.amin(qhulldist.distone(others,deviated,dimsInSet=("others",)),dim=("t",))
        return qhulldist.distone(others,deviated,dimsInSet=())
    def distance(params):
        assert(params.shape[-1]==2)
        with torch.no_grad():
            lxy = [compute_distxy(params,s["others"],s["deviated"],t,mask,iwpts) for (s,t,mask,iwpts) in zip(lsit,lt,lmask,liwpts)]
        return lxy
    params = torch.tensor([[0.5,2.]]).rename(PARAMS,...).to(DEVICE)
    d = distance(params)
    print(named.nanamin(d[0],dim=("t",))/1852)
    print(lsit[0]["others"].fid)
    dn = distance(params)
    print(dn[0]-d[0])
    # print(named.nanamin(d[0],dim=("t",))/1852)
    # print(lsit[0]["others"].fid)
    return distance





#DSITUATION = load_situation("all.dsituation")
DSITUATION = { 0: load_situation("./situations/38949417_1657956297_1657956993.situation")}
distance = generate_distance_function(DSITUATION,step=1)
# def distance(params):
#     nind = params.shape[0]
#     print(nind)
#     def ghman(dt0,dt1,dangle,dspeed):
#         return estimate_uncertainty.generatehulls_maneuvered(flights,t,dt0, dt1, dangle, dspeed)
#     def ghunman(dt0,dt1,dangle,dspeed):
#         return estimate_uncertainty.generatehulls_notmaneuvered(flights_others, t, dspeed)
#         #return estimate_uncertainty.generatehulls_notmaneuvered(flights_others,t, dspeed)
#     qhulldist = Qhulldist(flights.device(),n=3000)
#     if nind >= NMAX:
#         res = [distancecut(x,ghman,ghunman,qhulldist) for x in np.array_split(params,nind//NMAX)]#nind//NMAX)]
#         return np.concatenate(res)
#     else:
#         return distancecut(params,ghman,ghunman,qhulldist)




def main():
    DSITUATION = { 0: load_situation("./situations/38949417_1657956297_1657956993.situation")}
    distance = generate_distance_function(DSITUATION,step=1)
    pass

if __name__ == '__main__':
    main()
