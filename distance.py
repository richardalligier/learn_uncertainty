import os
import fit_traj
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS,WPTS, deserialize_dict
import torch
# torch.use_deterministic_algorithms(True)
from torchtraj import qhull,named,uncertainty
from add_uncertainty import Add_uncertainty,VALUESTOTEST
import tqdm
from collections import namedtuple

THRESHT = 20 # seconds
THRESHALT = 800 # feet
PARAMS = "params"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def read_config():
    ''' read config to get folders '''
    with open("CONFIG","r") as f:
        l = [[x.strip() for x in line.split("=")] for line in f if len(line.strip())>0 and line.strip()[0]!="#"]
        d = {k:v for k,v in l}
    Config = namedtuple("config",list(d))
    return Config(**d)

# class 

def generate_distance_function(dsituation,step):
    # dsituationvalues = [{k:v for k,v in s.items()} for s in dsituationvalues]#[:1]
    print("formatting traj data started")
    ladd=[]
#    dsituation ={82:dsituation[82]}
    for k,ss in tqdm.tqdm(dsituation.items()):
        for s in tqdm.tqdm(ss):
            # print("nb OTHERS",k)
            # print(s["deviated"].tdeviation.names)
            # print(s["deviated"].tdeviation.shape)
            # print(s["deviated"].tdeviation.dtype)
            ladd.append(Add_uncertainty(s,step=step,capmem=None))# for s in dsituationvalues]#[:1]
            # 1000000000000000000000
            # 10000000000
            #  1000000000
            # 1000000000000000
            # 1000000000000
            # 100000000000
    print("formatting traj data done")
    def distance(dargs):
        for x in dargs.values():
            assert(x.dim()<=2)
            assert(x.dim()==1 or x.shape[-1]<=2)
            for y in x.names:
                assert(y is None)
        dargs = {k:v.rename(PARAMS,VALUESTOTEST) for k,v in dargs.items()}
        print(f"{len(ladd)=}")
        with torch.no_grad():
            l_min_xy = []
            l_id = []
            for add in tqdm.tqdm(ladd):
            # for add in tqdm.tqdm(ladd):
                #print(i,torch.cuda.memory_allocated()/1024/1024)
                torch.cuda.empty_cache()
                # print(i,torch.cuda.memory_allocated()/1024/1024)
                add.to(DEVICE)
                # print(i,torch.cuda.memory_allocated()/1024/1024)
                l_min_xy.append(add.compute_min_distance_xy_on_conflicting_z(dargs,thresh_z=800))
                l_id.append(add.sit_uncertainty["deviated"].sitf.fid)
                add.to("cpu")
        return named.cat(l_min_xy, dim=l_min_xy[0].names.index(SITUATION))/1852,named.cat(l_id,dim=l_id[0].names.index(SITUATION))
    return distance

CONFIG = read_config()

DSITUATION = load_situation(os.path.join(CONFIG.FOLDER,"all_800_10_1800_2.dsituation"))



# print(DSITUATION)
# DSITUATION = { 4: DSITUATION[4]}
distance = generate_distance_function(DSITUATION,step=5)#list(DSITUATION.values()),step=5)



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
    import time
    nparams = 100
    duparams = {
        "dangle": torch.tensor([[-0.1,0.1]]*nparams)*0,
        "dt0": torch.tensor([[-10,10]]*nparams)*0,
        "dt1": torch.tensor([[-10,10]]*nparams)*0,
        "dspeed": torch.tensor([[1,1]]*nparams),
        "ldspeed": torch.tensor([[1,1.]]*nparams),
        "vspeed": torch.tensor([[1,1]]*nparams),
    }
    duparams = {k:v.to(DEVICE) for k,v in duparams.items()}
    for _ in range(1):
        st = time.perf_counter()
        d,lid = distance(duparams)
        print(time.perf_counter()-st)
    print(d)
    print(d.shape)
    vmin,imin = torch.min(d[0].rename(None),dim=-1)
    print(torch.sum(d[0].rename(None)==0.))
    print(lid[imin])

if __name__ == '__main__':
    main()
