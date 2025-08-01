from learn_uncertainty import read_json, douglas_peucker,geosphere
import numpy as np
import torch
from torchtraj.utils import T, XY,WPTS
from torchtraj import flights, named, uncertainty
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import pandas as pd
import matplotlib.pyplot as plt
from torchtraj import fit, traj
import datetime
import operator as op
import matplotlib.animation as animation
from pathlib import Path
import os
from collections import namedtuple

class NoAlignedAfter(Exception):pass
class NoAlignedBefore(Exception):pass

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def read_config():
    ''' read config to get folders '''
    with open("CONFIG","r") as f:
        l = [[x.strip() for x in line.split("=")] for line in f if len(line.strip())>0 and line.strip()[0]!="#"]
        d = {k:v for k,v in l}
    Config = namedtuple("config",list(d))
    return Config(**d)

#KIM_PARAMETERS = {"angle_precision":0.5,"min_distance":200.}
NM2METERS = 1852

THRESH_XY_MODEL = 100#m
THRESH_Z_MODEL = 100#m

BATCH = "batch"
SITUATION = "situation"
OTHERS = "others"
DTYPE = torch.float32

ALIGN_MIN_DISTANCE = 200
ALIGN_ANGLE_PRECISION = 2

def unix2datetime(u):
    return datetime.datetime.fromtimestamp(u,datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')


def check_is_named_tensor(t,dtype):
    return isinstance(t,torch.Tensor) and t.dtype == dtype

def deserialize_dict(x):
    return {k:deserialize(v) for k,v in x.items()}

def deserialize_list(x):
    return [deserialize(v) for v in x]

def donothing(x):
    return x

def serialize(a):
    #print(type(a))
    if isinstance(a, dict):
        return deserialize_dict, {k:serialize(v) for k,v in a.items()}
    if isinstance(a, list):
        return deserialize_list, [serialize(v) for v in a]
    elif isinstance(a, SituationOthers):
        return a.serialize()
    elif isinstance(a, Exception):
        return donothing, a
    else:
        raise Exception(f"Serialize {type(a)} not supported")

def deserialize(a):
    return a[0](a[1])

def pad_dimname(input,dimname,dimpad,mode="constant",value=None):
    i = input.names.index(dimname)
    n = len(input.shape)
    pad = list((0,0) * (n-i))
    pad[:2] = dimpad
    # print(pad)
    # print(input.names)
    return named.pad(input,tuple(pad),mode,value)

def masked_generate(f,t,tobs):
    xy = traj.generate(f,t)
    newaxis = "tobs"
    assert(newaxis not in t.names)
    assert(newaxis not in tobs.names)
    diff = op.sub(*named.align_common(t,tobs.rename(**{T:newaxis}))).align_to(...,newaxis)
    mask = (diff.abs().min(axis=-1).values < 20.).align_as(xy)
    return xy * mask / mask


class SituationOthers:
    def __init__(self,tzero,fid,fxy,fz,t):
        assert check_is_named_tensor(fid,torch.int64)
        assert check_is_named_tensor(tzero,torch.int64)
        assert check_is_named_tensor(t,DTYPE)
        self.tzero = tzero
        self.fid = fid
        self.fxy = fxy
        self.fz = fz
        self.t = t
    def dictparams(self):
        return {k:getattr(self,k) for k in ["fid","fxy","fz","t","tzero"]}
    @classmethod
    def dmap(cls,fsit,f):
        res = {}
        for k,v in fsit.dictparams().items():
            # print(k)
            if isinstance(v,torch.Tensor):
                res[k]=f(k,v)
            elif isinstance(v,flights.Flights):
                res[k]= v.dmap(v,f)
            elif isinstance(v,Alignment):
                res[k]= v.dmap(v,f)
            else:
                raise Exception(f"dmap {k} not Tensor nor Flights")
        return cls(**res)
    def serialize(self):
        res = {}
        for k,v in self.dictparams().items():
            if isinstance(v,torch.Tensor):
                res[k] = named.serialize(v)
            elif isinstance(v,flights.Flights):
                res[k] = v.serialize()
            elif isinstance(v,Alignment):
                res[k] = v.serialize()
            else:
                raise Exception(f"dmap {k} not Tensor nor Flights")
        return self.deserialize,res
    @classmethod
    def deserialize(cls,d):
        res = {}
        for k,v in d.items():
            res[k]=deserialize(v)
        return cls(**res)

    def generate_mask(self,t,thresh):
        newaxis = "tobs"
        tobs = self.t
        assert(newaxis not in t.names)
        assert(newaxis not in tobs.names)
        diff = op.sub(*named.align_common(t,tobs.rename(**{T:newaxis}))).align_to(...,newaxis)
        mask = (diff.abs().min(axis=-1).values < thresh)
        assert(tobs.names[-1]==T)
        tmin = torch.min(tobs,dim=-1).values
        tmax = torch.max(tobs,dim=-1).values
        # print(tobs)
        # print(tmin)
        # print(tmax)
        # print(t)
        mrange = torch.logical_and(op.le(*named.align_common(tmin,t)),
                                   op.le(*named.align_common(t,tmax)))
        # print(op.lt(*named.align_commontmin<=t)
        mask = torch.logical_and(mrange,mask)
        # raise Exception
        return mask / mask

    def generate_xy(self,t):
        return traj.generate(self.fxy,t)
    def generate_tz(self,t):
        return traj.generate(self.fz,t)#[...,0]
    def generate_z(self,t):
        return traj.generate(self.fz,t)[...,1]
    @classmethod
    def cat(cls,lsit,dimname):
        res =  {k:[getattr(s,k) for s in lsit] for k in lsit[0].dictparams()}
        d = {}
        for k,v in res.items():
            # print(k)
            if isinstance(v[0],torch.Tensor):
                # print(v[0].shape)
                # print(v[0].names)
                if k =="t":
                    it = v[0].names.index("t")
                    maxt = max(t.shape[it] for t in v)
                    v = [pad_dimname(x,"t",(0,maxt-x.shape[it]),mode="replicate") for x in v]
                d[k] = named.cat(v,dim=v[0].names.index(dimname))
                # print(k,v[0].shape,d[k].shape)
            elif isinstance(v[0],flights.Flights):
                d[k] = flights.named_cat_lflights(v,dimname)
            elif isinstance(v[0],Alignment):
                d[k] = Alignment.cat(v,dimname=dimname)
            else:
                raise Exception(f"cat {k} not Tensor nor Flights")
        return cls(**d)

flights.add_tensors_operations(SituationOthers)


class SituationDeviated(SituationOthers):
    def __init__(self,tzero,fid,fxy,fz,t,tdeviation,tturn,align_before,align_after,actual_min_dist):#,trejoin,beacon):#,wpt_start,wpt_turn,wpt_rejoin):
        super().__init__(tzero,fid,fxy,fz,t)
        assert check_is_named_tensor(tdeviation,DTYPE)
        assert check_is_named_tensor(tturn,DTYPE)
        # assert check_is_named_tensor(trejoin,DTYPE)
        # assert check_is_named_tensor(beacon,DTYPE)
        self.tdeviation = tdeviation
        self.tturn = tturn
        # self.trejoin = trejoin
        # self.beacon = beacon
        self.align_before = align_before
        self.align_after = align_after
        self.actual_min_dist = actual_min_dist
    @property
    def trejoin(self):
        return self.align_after.tend
    @property
    def beacon(self):
        return self.align_after.beacon
    def generate_trange_conflict(self,step):
        max_duration = (self.trejoin - self.tdeviation).max().item()
        tfromzero = torch.arange(start=0,end=max_duration,step=step,dtype=self.tturn.dtype,device=self.tturn.device).rename(T)
        res = op.add(*named.align_common(self.tdeviation,tfromzero))
        # print((self.tdeviation+self.tzero)[0].item())
        # print(1657955999)
        # raise Exception
        return res.align_to(...,T)
    def dictparams(self):
        res = super().dictparams()
        for v in ["tdeviation","tturn","align_before","align_after","actual_min_dist"]:
            res[v]=getattr(self,v)
        return res


def quality_check(sit, sitflights):
    t_zero_situation = sit.trajectories.timestamp.min()
    xy = sitflights.generate_xy(sitflights.t).cpu()
    z =  sitflights.generate_z(sitflights.t).cpu()
    for i,fid in enumerate(sitflights.fid):
        # print(f"{fid=}")
        df = sit.trajectories.query("flight_id==@sitflights.fid[@i].item()")
        nt = df.timestamp.values.shape[0]
        assert((sitflights.t[i].cpu()[:nt] == df.timestamp.values-sit.trajectories.timestamp.min()).all())
        if isinstance(sitflights,SituationDeviated):
            assert(sitflights.tdeviation==sit.deviated.start-t_zero_situation)
        dxy = xy[i,:nt]-df[["x","y"]].values
        #dist = torch.hypot(dxy[...,0],dxy[...,1])
        dist= torch.linalg.norm(dxy,axis=-1)
        # print(dist.min(),dist.mean(),dist.max())
        # print(dist.max())
        assert(dist.max()<THRESH_XY_MODEL)
        dz = (z[i,:nt]-df["altitude"].values).abs()
        # print(dist.min(),dist.mean(),dist.max())
        # print(z)
        # print(df[["x","y","altitude"]])
        # print(dz.max())
        assert(dz.max()<THRESH_Z_MODEL)
        # raise Exception
        # raise Exception
        # dfin = sit..query("flight_id == @fid")


def trajreal_from_df(df,device,v):
    assert(type(v)==list)
    return torch.tensor(df[v].values,device=device,dtype=torch.float32).unsqueeze(0).rename(BATCH,T,XY)#.align_to(BATCH,T,XY)

def t_from_df(df,device):
    return torch.tensor(df.timestamp.values,device=device,dtype=torch.float32).rename(T)

def initialize_gen_xyz(df,device,v):
    trajreal = trajreal_from_df(df,device,v)
    # print(df.timestamp)
    t_zero = df.timestamp.values[0]
    t = t_from_df(df,device)
    t = t.align_as(trajreal)
    # dxy0 = trajreal[...,1,:]-trajreal[...,0,:]

    # trajreal[...,0,:] = trajreal[...,0,:] + dxy0 -
    # t[...,0]=0
    xy0 = trajreal[...,0,:]
    # dx = trajreal[...,1:,0]-trajreal[...,:-1,0]
    # dy = trajreal[...,1:,1]-trajreal[...,:-1,1]
    # # print(trajreal)
    # dxy = torch.hypot(dx,dy)
    dxy = torch.linalg.norm((trajreal[...,1:,:]-trajreal[...,:-1,:]).rename(None),dim=-1).rename(*trajreal.names[:-1])
    # print(dxy.shape,dxy.names,dx.shape,trajreal.shape)
    duration = t[...,1:,0]-t[...,:-1,0]
    # print(duration)
    # # print(t)
    # # print(duration,duration.shape)
    # print(trajreal)
    # print(trajreal.shape)
    # print(dxy)
    v = dxy / duration
    wpts = trajreal[...,1:,:]
    wpts = wpts.rename(**{T:WPTS})
    v = v.rename(**{T:WPTS})
    turn_rate = 0.01 * torch.ones((1,1),device=device).rename(BATCH,WPTS)
    #turn_rate = torch.ones_like(duration).rename(*duration.names)#.rename(*duration.names)
    # print(turn_rate.names,duration.names)
    assert(turn_rate.shape==(1,1))
    return {"xy0":xy0,"v":v, "turn_rate":turn_rate, "wpts":wpts}

def initialize(df,device):#(trajreal,t):
    # res = initialize_gen_xyz(df,device,["x","y"])
    # fxy=flights.Flights.from_wpts(**res)
    # assert(fxy.turn_rate.shape==(1,1))
    # fxy.duration = torch.round(fxy.duration)
    # res = initialize_gen_xyz(df,device,["timestamp","altitude"])
    # fz=flights.Flights.from_wpts(**res)
    # assert(fz.turn_rate.shape==(1,1))
    # fz.duration = torch.round(fz.duration)
    # return fxy,fz
    return initialize_xy(df,device),initialize_z(df,device)



def initialize_xy(df,device):#(trajreal,t):
    res = initialize_gen_xyz(df,device,["x","y"])
    fxy=flights.Flights.from_wpts(**res)
    assert(fxy.turn_rate.shape==(1,1))
    fxy.duration = torch.round(fxy.duration)
    return fxy
def initialize_z(df,device):#(trajreal,t):
    res = initialize_gen_xyz(df,device,["timestamp","altitude"])
    fz=flights.Flights.from_wpts(**res)
    assert(fz.turn_rate.shape==(1,1))
    fz.duration = torch.round(fz.duration)
    return fz


# def initialize_acc(df,device):#(trajreal,t):
#     t_zero,d=initialize_gen(df,device)
#     meanv = d["v"].align_to(...,WPTS)
#     v = meanv.clone().detach()
#     for i in range(v.shape[-1]-2,-1,-1):
#         v[...,i] = - v[...,i+1] + 2 * meanv[...,i]
#     d["v"]=v
#     f = flights.FlightsWithAcc.from_wpts(**d)
#     return  f

def find_longest_aligned(df,beacons,thresh):
    # flight = TrafficFlight(df)
    l = []
    print(df.shape)
    for beacon in beacons:
        lat1 = df.latitude.values[0]
        lon1 = df.longitude.values[0]
        lat2 = np.float64(beacon.latitude)
        lon2 = np.float64(beacon.longitude)
        latm = df.latitude.values
        lonm = df.longitude.values
        d=geosphere.distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm)
        l.append(d)
    res=torch.tensor(np.array(l)).cummax(axis=1).values
    # print(res)
    npts = res.shape[1]
    mask = res > thresh
    res = res - torch.arange(npts) * thresh
    res[mask]=0.
    # print(res)
    vi,i = torch.min(res,dim=0,keepdim=False)
    j = torch.min(vi,dim=0,keepdim=False).indices.item()
    i = i[j].item()
    whichbeacon = i
    selectednbok = j
    return whichbeacon,selectednbok


class Alignment:
    def __init__(self,beacon,tstart,tend):
        self.beacon = beacon
        self.tstart = tstart
        self.tend = tend
    @classmethod
    def from_aligned(cls,alignedflight,t_zero_situation,dbeacons,device):
        tstart = torch.tensor([alignedflight.timestampf.min()-t_zero_situation],device=device,dtype=DTYPE,names=(BATCH,))
        tend = torch.tensor([alignedflight.timestampf.max()-t_zero_situation],device=device,dtype=DTYPE,names=(BATCH,))
        beacon = dbeacons[alignedflight.navaid.iloc[0]]
        beacon = torch.tensor([[beacon.x,beacon.y]],device=device,dtype=DTYPE,names=(BATCH,XY))
        return Alignment(beacon,tstart,tend)
    @classmethod
    def dmap(cls,alignment,f):
        res = {}
        for k,v in alignment.dictparams().items():
            # print(k)
            if isinstance(v,torch.Tensor):
                res[k]=f(k,v)
            else:
                raise Exception(f"dmap {k} not Tensor")
        return cls(**res)
    def dictparams(self):
        return {k:getattr(self,k) for k in ["beacon","tstart","tend"]}
    def serialize(self):
        res = {}
        for k,v in self.dictparams().items():
            if isinstance(v,torch.Tensor):
                res[k] = named.serialize(v)
            else:
                raise Exception(f"dmap {k} not Tensor nor Flights")
        return self.deserialize,res
    @classmethod
    def deserialize(cls,d):
        res = {}
        for k,v in d.items():
            res[k]=deserialize(v)
        return cls(**res)
    @classmethod
    def cat(cls,lsit,dimname):
        res =  {k:[getattr(s,k) for s in lsit] for k in lsit[0].dictparams()}
        d = {}
        for k,v in res.items():
            if isinstance(v[0],torch.Tensor):
                d[k] = named.cat(v,dim=v[0].names.index(dimname))
            else:
                raise Exception(f"cat {k} not Tensor nor Flights")
        return cls(**d)
    def __str__(self):
        return f"beacon={self.beacon}, tstart={self.tstart}, tend={self.tend}"

flights.add_tensors_operations(Alignment)

DEBUG = False

def decide_laligned_old(laligned):
    ts = np.sort(np.unique(np.concat([f.timestampf.values for f in laligned])))
    lres = []
    lastnavaid = None
    lastt = 0
    for t in ts:
        lalignedt = [f.query("timestampf==@t") for f in laligned]
        lalignedt = [f for f in lalignedt if len(f)==1]
        alignedt = min(lalignedt,key= lambda f: f.delta.values[0])
        if alignedt.navaid.values[0]!=lastnavaid or abs(t-lastt) > 60:
            lres.append([alignedt])
            lastnavaid = alignedt.navaid.values[0]
        else:
            lres[-1].append(alignedt)
        lastt=t
    lres = [pd.concat((x)) for x in lres]
#    lres.reverse()
    lres = [x for x in lres if x.timestampf.values[-1]-x.timestampf.values[0]>60]
    return lres

def decide_laligned(laligned):
    import graph_tool.all as gt
    import time
    ts = np.sort(np.unique(np.concat([f.timestampf.values for f in laligned])))
    n = sum([f.shape[0] for f in laligned])
    print(n)
    g = gt.Graph(g=n,directed=True)
    dlines = {i: line for i,line in enumerate(line for f in laligned for _,line in f.iterrows())}
    edges = []
    lweights = []
    for i in dlines:
        for j in dlines:
            if i!=j:
                #print(dlines[i])
                if dlines[i].timestampf+60<dlines[j].timestampf or (dlines[i].timestampf<dlines[j].timestampf and dlines[i].navaid==dlines[j].navaid):
                    edges.append((i,j))
                    lweights.append(-(ALIGN_ANGLE_PRECISION - dlines[i].delta))
    g.add_edge_list(edges)
    start = time.perf_counter()
    eprop_cost = g.new_edge_property("float")
    # for e in g.iter_edges():
    #     i = e[0]
    #     eprop_cost[e] = -(ALIGN_ANGLE_PRECISION - dlines[i].delta)
    n = len(dlines)
    # print(n)
    # a = np.zeros((n,n),dtype=float)
    # print(a.shape)
    # for i in dlines:
    #     a[i]=-(ALIGN_ANGLE_PRECISION - dlines[i].delta)
    # print(type(eprop_cost.a))
    eprop_cost.a =lweights
    #eprop_cost.set_values(deprop_cost)
    # print(time.perf_counter()-start)
    vend = g.add_vertex()
    vstart = g.add_vertex()
    for v in g.iter_vertices():
        if v != vend:
            e = g.add_edge(v, vend)
            if v!=vstart:
                eprop_cost[e] = -(ALIGN_ANGLE_PRECISION - dlines[v].delta)
        if v != vstart:
            e = g.add_edge(vstart, v)
            eprop_cost[e] = 1
    # print("graph done")
    start = time.perf_counter()
    longest_path, _ = gt.shortest_path(
        g,
        source=vstart,
        target=vend,
        weights=eprop_cost,
        negative_weights=True,
        dag=True,
    )
    # print(time.perf_counter()-start)
    longest_path = [int(x) for x in longest_path if int(x) in dlines]
    # print(longest_path)
    lres = []
    lastnavaid = None
    lastt=0
    for i in longest_path:
        t = dlines[i].timestampf
        if lres==[] or lastnavaid != dlines[i].navaid or abs(t-lastt) > 60:
            lres.append([dlines[i]])
            lastnavaid = dlines[i].navaid
        else:
            lres[-1].append(dlines[i])
        lastt = t
    lres = [pd.DataFrame((x)) for x in lres]
    return lres

def compute_alignments(sit,t_zero_situation,device):
    df = sit.trajectories.query("flight_id==@sit.deviated.flight_id").copy()
    df["timestampf"] = df["timestamp"]
    df["timestamp"] = [pd.Timestamp(t,unit="s") for t in df["timestamp"].values]
    deviated = TrafficFlight(df)
    # print(sit.deviated.beacons)
    laligned = [f.data for f in deviated.aligned_on_navpoint(sit.deviated.beacons,min_distance=ALIGN_MIN_DISTANCE,angle_precision=ALIGN_ANGLE_PRECISION,min_time="30s")]
    laligned = decide_laligned(laligned)
    if DEBUG:
        print(sit.deviated.start)
        print(sit.deviated.stop)
        for f in laligned:
            print(list(f))
            print(f[["timestampf",'bearing', 'shift', 'delta',"navaid"]])
        # raise Exception

    dbeacons = {b.name:b for b in sit.deviated.beacons}

    aligned_before = [f for f in laligned if f.timestampf.max()< sit.deviated.stop]
    if aligned_before == []:
        raise NoAlignedBefore
    tmax_before = max([f.timestampf.max() for f in aligned_before])
    aligned_before = [f for f in aligned_before if f.timestampf.max() == tmax_before][0]
    before = Alignment.from_aligned(aligned_before,t_zero_situation,dbeacons,device)

    aligned_after = [f for f in laligned if f.timestampf.min()> sit.deviated.start]
    if aligned_after == []:
        raise NoAlignedAfter
    # print("aligned_after",sit.deviated.start,sit.deviated.stop)
    # for f in aligned_after:
    #     print(f[["timestampf",'bearing', 'shift', 'delta',"navaid"]])
    tmin_after = min([f.timestampf.min() for f in aligned_after])
    aligned_after = [f for f in aligned_after if f.timestampf.min() == tmin_after]
    # print(len(aligned_after))
    aligned_after = aligned_after[0]
    after = Alignment.from_aligned(aligned_after,t_zero_situation,dbeacons,device)
    if DEBUG:
        print(dbeacons[aligned_after.navaid.iloc[0]].name)
        raise Exception
    return before,after


def compute_t_rejoin(sit,thresh,nwpts):
    df = sit.trajectories.query("flight_id == @sit.deviated.flight_id").copy()
    # print(sit.deviated.flight_id,df.shape,sit.deviated.stop)
    df = df.query("timestamp>=@sit.deviated.stop").reset_index(drop=True)
    # df["timestamp"]=pd.to_datetime(df["timestamp"],unit='s')
    l = []
    dfi = df.copy()
    # print(dfi.shape)
    beacons = sit.deviated.beacons
    for _ in range(nwpts):
        # print(dfi.shape)
        whichbeacon, selectednbok = find_longest_aligned(dfi,beacons,thresh)
        l.append((sit.deviated.beacons[whichbeacon],dfi.iloc[selectednbok]))
        dfi = dfi.query("timestamp>=@l[-1][-1].timestamp")
        beacons = beacons#[whichbeacon:]
        # print(dfi.index)
    return l
    # print(l)
    # print(whichbeacon,selectednbok)
    # raise Exception

# new axis SITUATION et axe OTHERS pour les autres avions
def convert_situation_to_flights(sit,initialize,device,thresh_xy,thresh_z):
    df = sit.trajectories.copy()
    #df = df[df.flight_id.isin([38910912,38909998])]
    # print(f"{df.flight_id.unique()=}")
    t_zero_situation = df.timestamp.values.min()
    df["timestamp"] = df["timestamp"] - t_zero_situation
    t_deviation = sit.deviated.start - t_zero_situation
    t_turn = sit.deviated.stop - t_zero_situation
    # [(beacon_rejoin,line_rejoin)] = compute_t_rejoin(sit,thresh=200,nwpts=1)
    align_before,align_after = compute_alignments(sit,t_zero_situation,device)
    # t_rejoin =  line_rejoin.timestamp - t_zero_situation
    t_rejoin = align_after.tend[0].item()
    def convert(df):
        lfxy = []
        lfz = []
        lt = []
        lfid = []
        for fid,dfin in df.groupby("flight_id"):
            if dfin.shape[0]==1:
                continue
            if dfin.shape[0]==0:
                raise Exception
            lfid.append(fid)
            lttoinclude = sorted([t_deviation,t_turn,t_rejoin])
            vitoinclude = [np.where(dfin.timestamp.values==t)[0] for t in lttoinclude]
            itoinclude = []
            # print(vitoinclude)
            for vi in vitoinclude:
                assert(len(vi)<=1)
                if len(vi)==1:
                    itoinclude.append(vi[0])
            itoinclude.append(0)
            itoinclude.append(len(dfin.timestamp.values)-1)
            itoinclude = np.unique(np.sort(np.array(itoinclude)))
            mask_xy = np.zeros(len(dfin.timestamp.values),dtype=bool)
            mask_z = np.zeros(len(dfin.timestamp.values),dtype=bool)
            mask = np.zeros(len(dfin.timestamp.values),dtype=bool)
            for (ia,ib) in zip(itoinclude[:-1],itoinclude[1:]):
                # mask_xy[ia:ib+1]=douglas_peucker.douglas_peucker(dfin[["x","y"]].values[ia:ib+1],dfin.timestamp.values[ia:ib+1],eps=thresh_xy)
                # mask_z[ia:ib+1]=douglas_peucker.douglas_peucker(dfin[["altitude","timestamp"]].values[ia:ib+1],dfin.timestamp.values[ia:ib+1],eps=thresh_z)
                # mask[ia:ib+1]=douglas_peucker.douglas_peucker(dfin[["timestamp","altitude","x","y"]].values[ia:ib+1],dfin.timestamp.values[ia:ib+1],eps=min(thresh_z,thresh_xy))
                mask_xy[ia:ib+1]=douglas_peucker.douglas_peucker(dfin[["x","y"]].values[ia:ib+1],dfin.timestamp.values[ia:ib+1],eps=thresh_xy)
                mask_z[ia:ib+1]=douglas_peucker.douglas_peucker(dfin[["timestamp","altitude"]].values[ia:ib+1],dfin.timestamp.values[ia:ib+1],eps=thresh_z)
            # print(mask)
            # mask = np.logical_or(mask_z,mask_xy)
            # dfwpts = dfin.where(pd.Series(mask,index=dfin.index)).dropna(subset=["x"]).reset_index()
            # # assert (dfwpts.timestamp.values[0]==dfin.timestamp.values[0])
            # fxy,fz = initialize(dfwpts,device)
            # print(dfin.shape,fid)
            fxy = initialize_xy(dfin.where(pd.Series(mask_xy,index=dfin.index)).dropna(subset=["x"]).reset_index(),device)
            fz = initialize_z(dfin.where(pd.Series(mask_z,index=dfin.index)).dropna(subset=["x"]).reset_index(),device)
            fxy = fxy.shift_xy0(float(dfin.timestamp.values[0]))
            fz = fz.shift_xy0(float(dfin.timestamp.values[0]))
            assert(fxy.turn_rate.shape==(1,1))
            assert(fz.turn_rate.shape==(1,1))
            lfxy.append(fxy)
            lfz.append(fz)
            lt.append(dfin.timestamp.values)
            # lt.append(torch.tensor(dfin.timestamp.values,device=device,names=(T,)))
        lfid = torch.tensor(lfid,device = device,dtype=torch.int64,names=(BATCH,))
        # print(f"{np.array(lt)=}")
        maxt = max(t.shape[0] for t in lt )
        lt =  torch.tensor(np.array([np.pad(t,(0,maxt-t.shape[0]),mode="edge") for t in lt]),device=device,dtype=DTYPE,names=(BATCH,T))
        # print(lt)
        lfxy = flights.cat_lflights(lfxy)
        lfz = flights.cat_lflights(lfz)
        return {"fid":lfid,"fxy":lfxy,"fz":lfz,"t":lt}
    ddeviated = convert(df.query("flight_id==@sit.deviated.flight_id"))
    ddeviated["tzero"] = torch.tensor([t_zero_situation],device=device,dtype=torch.int64,names=(BATCH,))
    ddeviated["tdeviation"] = torch.tensor([t_deviation],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["tturn"] = torch.tensor([t_turn],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["actual_min_dist"] = torch.tensor([sit.deviated.actual_pairwise.lateral_dist_at_tcpa.min()],device=device,dtype=DTYPE,names=(BATCH,))
    # ddeviated["trejoin"] = torch.tensor([t_rejoin],device=device,dtype=DTYPE,names=(BATCH,))
    # ddeviated["beacon"] = torch.tensor([[beacon_rejoin.x,beacon_rejoin.y]],device=device,dtype=DTYPE,names=(BATCH,XY))
    ddeviated["align_before"] = align_before
    ddeviated["align_after"] = align_after
    deviated = SituationDeviated(**ddeviated)
    quality_check(sit,deviated)
    deviated = deviated.dmap(deviated,lambda k,v:v.rename(**{BATCH:SITUATION}))
    dothers = convert(df.query("flight_id!=@sit.deviated.flight_id"))
    dothers["tzero"] = ddeviated["tzero"].clone().detach()
    others = SituationOthers(**dothers)
    quality_check(sit,others)
    # others = others.dmap(others,lambda v:named.unsqueeze(v,0,SITUATION))
    others = others.dmap(others,lambda k,v: v.rename(**{BATCH:SITUATION}) if k=="tzero" else named.unsqueeze(v.rename(**{BATCH:OTHERS}),0,SITUATION))
    return deviated, others
# linear regression on "position=f(v)"
# def fit_speed(trajreal,f,t_zero,device):#(trajreal,t):
#     t = t_from_df(df,t_zero,device)
#     d=initialize_gen(df,t_zero,device)
#     meanv = d["v"].align_to(...,WPTS)
#     v = meanv.clone().detach()
#     for i in range(v.shape[-1]-2,-1,-1):
#         v[...,i] = - v[...,i+1] + 2 * meanv[...,i]
#     d["v"]=v
#     f = flights.FlightsWithAcc.from_wpts(**d)
#     return f



def scatter_with_number(x,y,istart=1,marker="o",linestyle="-"):
    plt.scatter(x,y,marker=marker)
    plt.plot(x,y,linestyle=linestyle)
    if istart is not None:
        for i in range(len(x)):
            plt.text(x[i],y[i],i+istart)


def plot_xy(xy,wpts):
    import matplotlib.pyplot as plt
    if xy.ndim == 2:
        plt.plot(xy[:,0],xy[:,1])
        scatter_with_number(wpts[:,0],wpts[:,1])
    else:
        for i in range(xy.shape[0]):
            plot_xy(xy[i],wpts[i])


def recplot(xy,plotfunction):
    if xy.ndim == 2:
        return [plotfunction(xy[:,0],xy[:,1])]
    else:
        res = []
        for i in range(xy.shape[0]):
            res += recplot(xy[i],plotfunction)
        return res


def plot_xory(xy,wpts,duration,c):

    if xy.ndim == 2:
        t = np.arange(xy.shape[0])
        plt.plot(t,xy[:,c])
        # print(duration.shape,duration.names,wpts.shape,wpts.names)
        scatter_with_number(np.cumsum(duration),wpts[:,c])
        # plt.scatter(np.cumsum(duration),wpts[:,c])
    else:
        for i in range(xy.shape[0]):
            plot_xory(xy[i],wpts[i],duration[i],c)


def plot(f,t,xory):
    xy  = traj.generate(f,t).cpu()
    # print(xy.shape)
    # print(t.shape)
    # plt.scatter(t.cpu(),xy[0,:,1])
    # return
    # print(t)
    # print(xy)
    wpts = f.compute_wpts().cpu()
    if xory:
        plot_xory(xy,wpts,f.duration.cpu(),0)
    else:
        recplot(xy,plt.plot)
        recplot(wpts,scatter_with_number)
        # plot_xy(xy,wpts)


# def error_xy(dfin,sit):
#     t=torch.tensor(dfin.timestamp.values,device=f.device())
#     traj.generate(f,t)


# def uncertainty_others(fothers,ldspeed):
#     fothers = fothers.clone()
#     fothers = uncertainty.change_longitudinal_speed(ldspeed,dothersiwpts["tdeviation"],dothersiwpts["trejoin"],fothers.fxy)

# def uncertainty_others(fothers,ldspeed):
#     fothers = fothers.clone()
#     fothers = uncertainty.change_longitudinal_speed(ldspeed,dothersiwpts["tdeviation"],dothersiwpts["trejoin"],fothers.fxy)


# def test_error(sit):

# raise Exception



def save_situation(o,fname):
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    torch.save(serialize(o),fname)

def load_situation(fname):
    return deserialize(torch.load(fname,map_location="cpu",weights_only=False))#.items()#{k:deserialize(v) for k,v in torch.load(fname,map_location="cpu").items()}

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-json')
    parser.add_argument('-situation')
    parser.add_argument("-cutalt",type = float)
    args = parser.parse_args()
    # print(args.json)
    sit = read_json.Situation.from_json(args.json)
    if args.cutalt is not None:
        sit = sit.cut_diffalt(args.cutalt)
    # plt.scatter(sit.trajectories.x,sit.trajectories.y)
    # plt.show()
    # raise Exception
    device = "cpu"
    try:
        fdeviated,fothers = convert_situation_to_flights(sit,initialize,device,thresh_xy=THRESH_XY_MODEL*0.99,thresh_z=THRESH_Z_MODEL * 0.99)
        save_situation({"deviated":fdeviated,"others":fothers},args.situation)
    except (NoAlignedAfter,NoAlignedBefore) as e:
        print(type(e))
        save_situation(e,args.situation)

if __name__ == "__main__":
    main()
