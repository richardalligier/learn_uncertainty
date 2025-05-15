import os
import fit_traj
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS,WPTS, deserialize_dict
import torch
from torchtraj import qhull,named,uncertainty
from add_uncertainty import Add_uncertainty,VALUESTOTEST


THRESHT = 20 # seconds
THRESHALT = 800 # feet
PARAMS = "params"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_distance_function(dsituationvalues,step):
    lsit = [{k:v.to(DEVICE) for k,v in s.items()} for s in dsituationvalues]#[:1]
    ladd = [Add_uncertainty(s,step=step) for s in lsit]#[:1]
    def distance(dargs):
        for x in dargs.values():
            assert(x.dim()<=2)
            assert(x.dim()==1 or x.shape[-1]<=2)
            for y in x.names:
                assert(y is None)
        dargs = {k:v.rename(PARAMS,VALUESTOTEST) for k,v in dargs.items()}
        # dangle,dt0,dt1,dspeed,ldspeed,vspeed = [x.rename(PARAMS,VALUESTOTEST) for x in args]
        # uparams = {
        #     "dangle": dangle,
        #     "dt0": dt0,
        #     "dt1": dt1,
        #     "dspeed": dspeed,
        #     "ldspeed": ldspeed,
        #     "vspeed": vspeed,
        # }
        with torch.no_grad():
            l_xy_z = [add.compute_dist(dargs)[:2] for add in ladd]
        return l_xy_z

    return distance





#DSITUATION = load_situation("all.dsituation")
DSITUATION = { 0: load_situation("/disk2/jsonKim/situations/2207/39068828_1658651339_1658651900.situation")}
distance = generate_distance_function(list(DSITUATION.values()),step=1)
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
    # DSITUATION = { 0: load_situation("disk2/jsonKim/situations/2207/39068828_1658651339_1658651900.situation")}
    # distance = generate_distance_function(DSITUATION,step=1)
    params = torch.tensor([[0.5,2.]])
    uparams = {
        "dangle": dangle,
        "dt0": dt0,
        "dt1": dt1,
        "dspeed": dspeed,
        "ldspeed": ldspeed,
        "vspeed": vspeed,
    }
    d = distance(params)
    print(named.nanamin(d[0],dim=("t",))/1852)
    print(lsit[0]["others"].fid)
    dn = distance(params)
    print(dn[0]-d[0])

if __name__ == '__main__':
    main()
