import read_json
import numpy as np
import torch
from torchtraj.utils import T, XY,WPTS
from torchtraj import flights, named, uncertainty
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import pandas as pd
import douglas_peucker
import matplotlib.pyplot as plt
from torchtraj import fit, traj
import datetime
import geosphere
import operator as op
import matplotlib.animation as animation



KIM_PARAMETERS = {"angle_precision":0.5,"min_distance":200.}
NM2METERS = 1852

THRESH_XY_MODEL = 100#m
THRESH_Z_MODEL = 100#m

BATCH = "batch"
SITUATION = "situation"
OTHERS = "others"
DTYPE = torch.float32


def unix2datetime(u):
    return datetime.datetime.fromtimestamp(u,datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')


def check_is_named_tensor(t,dtype):
    return isinstance(t,torch.Tensor) and t.dtype == dtype

class SituationOthers:
    def __init__(self,fid,fxy,fz,t):
        assert check_is_named_tensor(fid,torch.int64)
        assert check_is_named_tensor(t,DTYPE)
        self.fid = fid
        self.fxy = fxy
        self.fz = fz
        self.t = t
    def dictparams(self):
        return {k:getattr(self,k) for k in ["fid","fxy","fz","t"]}
    @classmethod
    def dmap(cls,fsit,f):
        res = {}
        for k,v in fsit.dictparams().items():
            if isinstance(v,torch.Tensor):
                res[k]=f(v)
            elif isinstance(v,flights.Flights):
                res[k]= v.dmap(v,f)
            else:
                raise Exception(f"dmap {k} not Tensor nor Flights")
        return cls(**res)

class SituationDeviated(SituationOthers):
    def __init__(self,fid,fxy,fz,t,tdeviation,tturn,trejoin,beacon):#,wpt_start,wpt_turn,wpt_rejoin):
        super().__init__(fid,fxy,fz,t)
        assert check_is_named_tensor(tdeviation,DTYPE)
        assert check_is_named_tensor(tturn,DTYPE)
        assert check_is_named_tensor(trejoin,DTYPE)
        assert check_is_named_tensor(beacon,DTYPE)
        self.tdeviation = tdeviation
        self.tturn = tturn
        self.trejoin = trejoin
        self.beacon = beacon
    def dictparams(self):
        res = super().dictparams()
        for v in ["tdeviation","tturn","trejoin","beacon"]:
            res[v]=getattr(self,v)
        return res


def quality_check(sit, sitflights):
    t_zero_situation = sit.trajectories.timestamp.min()
    xy = traj.generate(sitflights.fxy,sitflights.t).cpu()
    z =  traj.generate(sitflights.fz,sitflights.t)[...,0].cpu()
    for i,fid in enumerate(sitflights.fid):
        print(f"{fid=}")
        df = sit.trajectories.query("flight_id==@sitflights.fid[@i].item()")
        nt = df.timestamp.values.shape[0]
        assert((sitflights.t[i].cpu()[:nt] == df.timestamp.values-sit.trajectories.timestamp.min()).all())
        if isinstance(sitflights,SituationDeviated):
            assert(sitflights.tdeviation==sit.deviated.start-t_zero_situation)
        dxy = xy[i,:nt]-df[["x","y"]].values
        dist = torch.hypot(dxy[...,0],dxy[...,1])
        print(dist.min(),dist.mean(),dist.max())
        assert(dist.max()<THRESH_XY_MODEL*1.3)
        dz = (z[i,:nt]-df["altitude"].values).abs()
        print(dist.min(),dist.mean(),dist.max())
        # print(z)
        # print(df[["x","y","altitude"]])
        assert(dist.max()<THRESH_Z_MODEL*1.3)
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
    print(duration)
    # print(t)
    # print(duration,duration.shape)
    print(trajreal)
    print(trajreal.shape)
    print(dxy)
    v = dxy / duration
    wpts = trajreal[...,1:,:]
    wpts = wpts.rename(**{T:WPTS})
    v = v.rename(**{T:WPTS})
    turn_rate = 0.01 * torch.ones((1,1),device=device).rename(BATCH,WPTS)
    #turn_rate = torch.ones_like(duration).rename(*duration.names)#.rename(*duration.names)
    print(turn_rate.names,duration.names)
    assert(turn_rate.shape==(1,1))
    return {"xy0":xy0,"v":v, "turn_rate":turn_rate, "wpts":wpts}

def initialize(df,device):#(trajreal,t):
    res = initialize_gen_xyz(df,device,["x","y"])
    fxy=flights.Flights.from_wpts(**res)
    assert(fxy.turn_rate.shape==(1,1))
    fxy.duration = torch.round(fxy.duration)
    res = initialize_gen_xyz(df,device,["altitude","timestamp"])
    fz=flights.Flights.from_wpts(**res)
    assert(fz.turn_rate.shape==(1,1))
    fz.duration = torch.round(fz.duration)
    return fxy,fz

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
    for beacon in beacons:
        lat1 = df.latitude.values[0]
        lon1 = df.longitude.values[0]
        lat2 = np.float64(beacon.latitude)
        lon2 = np.float64(beacon.longitude)
        latm = df.latitude.values
        lonm = df.longitude.values
        d=geosphere.distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm)
        l.append(d)
    res=torch.tensor(l).cummax(axis=1).values
    print(res)
    npts = res.shape[1]
    mask = res > thresh
    res = res - torch.arange(npts) * thresh
    res[mask]=0.
    print(res)
    vi,i = torch.min(res,dim=0,keepdim=False)
    j = torch.min(vi,dim=0,keepdim=False).indices.item()
    i = i[j].item()
    whichbeacon = i
    selectednbok = j
    return whichbeacon,selectednbok

def compute_t_rejoin(sit,thresh,nwpts):
    df = sit.trajectories.query("flight_id == @sit.deviated.flight_id").copy()
    df = df.query("timestamp>=@sit.deviated.stop").reset_index(drop=True)
    # df["timestamp"]=pd.to_datetime(df["timestamp"],unit='s')
    l = []
    dfi = df.copy()
    beacons = sit.deviated.beacons
    for _ in range(nwpts):
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
def convert_situation_to_flights(sit,initialize,device,thresh_xy):
    df = sit.trajectories.copy()
    print(f"{df.flight_id.unique()=}")
    t_zero_situation = df.timestamp.values.min()
    df["timestamp"] = df["timestamp"] - t_zero_situation
    t_deviation = sit.deviated.start - t_zero_situation
    t_turn = sit.deviated.stop - t_zero_situation
    [(beacon_rejoin,line_rejoin)] = compute_t_rejoin(sit,thresh=200,nwpts=1)
    t_rejoin =  line_rejoin.timestamp - t_zero_situation
    def convert(df):
        lfxy = []
        lfz = []
        lt = []
        lfid = []
        for fid,dfin in df.groupby("flight_id"):
            lfid.append(fid)
            mask = douglas_peucker.douglas_peucker(dfin[["x","y"]].values,dfin.timestamp.values,eps=thresh_xy)
            mask = np.logical_or(mask, dfin.timestamp.values==t_deviation)
            mask = np.logical_or(mask, dfin.timestamp.values==t_turn)
            mask = np.logical_or(mask, dfin.timestamp.values==t_rejoin)
            dfwpts = dfin.where(pd.Series(mask,index=dfin.index)).dropna(subset=["x"]).reset_index()
            fxy,fz = initialize(dfwpts,device)
            fxy = fxy.shift_xy0(float(dfwpts.timestamp.values[0]))
            fz = fz.shift_xy0(float(dfwpts.timestamp.values[0]))
            assert(fxy.turn_rate.shape==(1,1))
            assert(fz.turn_rate.shape==(1,1))
            lfxy.append(fxy)
            lfz.append(fz)
            lt.append(dfin.timestamp.values)
            # lt.append(torch.tensor(dfin.timestamp.values,device=device,names=(T,)))
        lfid = torch.tensor(lfid,device = device,dtype=torch.int64,names=(BATCH,))
        # print(f"{np.array(lt)=}")
        maxt = max(t.shape[0] for t in lt )
        lt =  torch.tensor([np.pad(t,(0,maxt-t.shape[0]),mode="edge") for t in lt],device=device,dtype=DTYPE,names=(BATCH,T))
        print(lt)
        lfxy = flights.cat_lflights(lfxy)
        lfz = flights.cat_lflights(lfz)
        return {"fid":lfid,"fxy":lfxy,"fz":lfz,"t":lt}
    ddeviated = convert(df.query("flight_id==@sit.deviated.flight_id"))
    # ddeviated["fxy"] = flights.cat_lflights(ddeviated["fxy"])
    # ddeviated["fz"] = flights.cat_lflights(ddeviated["fz"])
    ddeviated["tdeviation"] = torch.tensor([t_deviation],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["tturn"] = torch.tensor([t_turn],device=device,dtype=DTYPE,names=(BATCH,))
    # print(beacon_rejoin.x)
    ddeviated["trejoin"] = torch.tensor([t_rejoin],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["beacon"] = torch.tensor([[beacon_rejoin.x,beacon_rejoin.y]],device=device,dtype=DTYPE,names=(BATCH,XY))
    deviated = SituationDeviated(**ddeviated)
    quality_check(sit,deviated)
    deviated = deviated.dmap(deviated,lambda v:v.rename(**{BATCH:SITUATION}))
    dothers = convert(df.query("flight_id!=@sit.deviated.flight_id"))
    others = SituationOthers(**dothers)
    quality_check(sit,others)
    others = others.dmap(others,lambda v:named.unsqueeze(v.rename(**{BATCH:OTHERS}),0,SITUATION))
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



def scatter_with_number(x,y):
    plt.scatter(x,y)
    for i in range(len(x)):
        plt.text(x[i],y[i],i+1)


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
        plotfunction(xy[:,0],xy[:,1])
    else:
        for i in range(xy.shape[0]):
            recplot(xy[i],plotfunction)


def plot_xory(xy,wpts,duration,c):

    if xy.ndim == 2:
        t = np.arange(xy.shape[0])
        plt.plot(t,xy[:,c])
        print(duration.shape,duration.names,wpts.shape,wpts.names)
        scatter_with_number(np.cumsum(duration),wpts[:,c])
        # plt.scatter(np.cumsum(duration),wpts[:,c])
    else:
        for i in range(xy.shape[0]):
            plot_xory(xy[i],wpts[i],duration[i],c)

def plotanimate(lxy,s=1.5):
    fig,ax = plt.subplots() # initialise la figure
    scats = tuple(ax.scatter([],[],s=s) for _ in lxy)
    margin = 200000.
    xmin = min([named.nanmin(xy[...,0].rename(None)) for xy in lxy])-margin
    xmax = max([named.nanmax(xy[...,0].rename(None)) for xy in lxy])+margin
    ymin = min([named.nanmin(xy[...,1].rename(None)) for xy in lxy])-margin
    ymax = max([named.nanmax(xy[...,1].rename(None)) for xy in lxy])+margin

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().axis('equal')
    def init():
        for scat in scats:
            scat.set_offsets(np.array([[],[]]).T)
        return scats
    def animate(i):
        # print("pos")
        # print(pos)
        # raise Exception
        for scat,xy in zip(scats,lxy):
            pos = xy[...,i:i+1:4,:].rename(None).flatten(end_dim=-2).numpy()
            scat.set_offsets(pos)
        return scats
    ani = animation.FuncAnimation(fig, func=animate, init_func=init, frames=lxy[0].shape[-2],
                              interval=1, blit=True, repeat=True)

    plt.show()
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


def test_uncertainty(sit,fdeviated,fothers,device):
    # fname = "data/AA38909998_1657916586_1657917209.json"
    # #fname = "data/AA38932291_1657920628_1657920857.json"
    # #fname = "data/AA38944134_1658001122_1658001604.json"
    # sit = read_json.Situation.from_json(fname)#.cut()
    # print(sit.deviated.flight_id)
    selected_fid = sit.deviated.flight_id
    print(sit.trajectories.groupby("flight_id").count())
    dfin = sit.trajectories.query("flight_id==@selected_fid")#.loc[1:]
    DANGLE = "dangle"
    MAN_WPTS = "man_wpts"
    DT = "dt1"
    DSPEED = "dspeed"
    LDSPEED = "ldspeed"
    print(fdeviated.fxy.duration.names)
    print(fdeviated.tdeviation)
    diwpts = {}
    dtwpts = {}
    print(fdeviated.fxy.duration.cumsum(axis=-1))
    print(fothers.fxy.duration.cumsum(axis=-1))
    for v in ["tdeviation","tturn","trejoin"]:
        dtwpts[v] = getattr(fdeviated,v)
        fdeviated.fxy = fdeviated.fxy.add_wpt_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
        fothers.fxy = fothers.fxy.add_wpt_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
    print(fdeviated.fxy.duration.cumsum(axis=-1))
    print(fothers.fxy.duration.cumsum(axis=-1))
    # raise Exception
    dothersiwpts = {}
    for v in ["tdeviation","tturn","trejoin"]:
        diwpts[v]= fdeviated.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
        dothersiwpts[v]= fothers.fxy.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
    # diwpts["tturn"]=diwpts["tturn"]-2
    # diwpts["trejoin"]=diwpts["trejoin"]-2

    # diwpts["tdeviation"]=torch.tensor([3],device=device,dtype=torch.int64).rename(BATCH)#.reshape(-1).rename(BATCH)
    # diwpts["tturn"]=diwpts["tdeviation"]+2
    # diwpts["trejoin"]=diwpts["tturn"]+2

    # wpts_start = 5+torch.tensor([0],device=device,dtype=torch.int64).reshape(-1).rename(BATCH)#.reshape(-1).rename(BATCH)#,DANGLE)
    # wpts_turn = fmodel.wpts_at_t(fdeviated.tturn)
    # wpts_rejoin = fmodel.wpts_at_t(fdeviated.trejoin)
    for v in ["tdeviation","tturn","trejoin"]:
        print(f"{v} {diwpts[v]=}")
    dangle = 1*torch.tensor([-0.1,0.1],device=device).reshape(-1).rename(DANGLE)
    # dt = 300+20*torch.arange(1,device=device).reshape(-1).rename(DT)
    dt = 1*torch.tensor([-10,10],device=device).reshape(-1).rename(DT)
    dspeed = torch.tensor([0.9,1.1],device=device).reshape(-1).rename(DSPEED)
    ldspeed = torch.tensor([0.9,1.1],device=device).reshape(-1).rename(LDSPEED)
    # dangle[0]=1#-0.3
    # print(dangle)
    #dt = 40+torch.arange(1,device=device,dtype=torch.int64).reshape(1,1).rename(BATCH,DANGLE)#torch.arangeones_like(wpts_turn)*10#np.pi/2
    # print(wpts_start,wpts_turn,wpts_rejoin)
    # raise Exception
    fs = fdeviated.fxy
    # angle
    fs = uncertainty.addangle(dangle,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fs,beacon=fdeviated.beacon)
    # t0
    fs = uncertainty.adddt_rotate(dt.rename(**{DT:"dt0"}),diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fs,beacon=fdeviated.beacon)
    # t1
    fs = uncertainty.adddt_rotate(dt,diwpts["tturn"],diwpts["tturn"],diwpts["trejoin"],fs,beacon=fdeviated.beacon)
    # speed
    fs = uncertainty.changespeed_rotate(dspeed,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fs,beacon=fdeviated.beacon)#,contract=False)
    # longitudinalspeed
    fo = uncertainty.change_longitudinal_speed(ldspeed,dothersiwpts["tdeviation"],dothersiwpts["trejoin"],fothers.fxy)
    modified_trejoin = uncertainty.gather_wpts(fs.duration.cumsum(axis=-1),diwpts["trejoin"]-1)[...,0]
    print(f"{fdeviated.trejoin=}")
    print(f"{modified_trejoin=}")
    max_duration = (fdeviated.trejoin - fdeviated.tdeviation).max().item()
    print(max_duration)
    t = torch.arange(start=0.,end=max_duration,step=1,device=device).rename(T)
    print(t)
    print(fdeviated.tdeviation)
    # a,b = named.align_common(fdeviated.tdeviation,t)
    t = op.add(*named.align_common(fdeviated.tdeviation,t)).align_to(...,T)
    t = torch.arange(start=0.,end=t.max(),step=1,device=device).rename(T)
    # print(fs.duration)
    #f = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    f = fs
    def masked_generate(f,t,tobs):
        xy = traj.generate(f,t)
        # print(t.names)
        # print(tobs.names)
        newaxis = "tobs"
        assert(newaxis not in t.names)
        assert(newaxis not in tobs.names)
        diff = op.sub(*named.align_common(t,tobs.rename(**{T:newaxis}))).align_to(...,newaxis)
        mask = (diff.abs().min(axis=-1).values < 20.).align_as(xy)
        return xy * mask / mask
    zs = masked_generate(fdeviated.fz,t,fdeviated.t)[...,0]
    zo = masked_generate(fothers.fz,t,fothers.t)[...,0]
    mask_closez = op.sub(*named.align_common(zs,zo)).abs() < 500.
    mask_not_closez = torch.logical_not(mask_closez)
    mask_closez = mask_closez / mask_closez
    mask_not_closez = mask_not_closez / mask_not_closez
    # print(mask_closez)
    xys = masked_generate(fs,t,fdeviated.t)
    xys = xys #* mask_closez.align_as(xys)
    xyo = masked_generate(fo,t,fothers.t)# * mask_closez / mask_closez
    xyoc = xyo * mask_closez.align_as(xyo)
    xyon = xyo * mask_not_closez.align_as(xyo)
    # print(xyo)
    # print(xyo.shape)
    # print(xys)
    # print(xys.shape)
    # print(zs)
    # print(zo)
    # raise Exception
    # dist = op.sub(*named.align_common(xys,xyo.rename(**{BATCH:OTHERS}))).abs().align_to(...,OTHERS,T,XY)
    # plotanimate([xys.cpu()],s=4)
    plotanimate([xys.cpu(),xyoc.cpu(),xyon.cpu()],s=4)
    # print(f.meanv())
    # print(f.duration)
    # print(fs.meanv())
    #plot(fothers.fxy,t,xory)
    print(diwpts["tdeviation"])
    print(diwpts["tturn"])
    print(diwpts["trejoin"])
    xory=False
    plot(fs,t,xory)
    if not xory:
        recplot(fdeviated.beacon.cpu(),plt.scatter)
        plt.gca().axis('equal')
    print(f)
    print(fs)
    # print(fdeviated.tdeviation)
    # print(traj.generate(f,t))
    plt.show()

# def test_error(sit):

# raise Exception
def main():

    import torchtraj

    import torch
    import matplotlib.pyplot as plt
    from traffic.data import opensky
    torch.random.manual_seed(0)
    fname = "data/AA38909998_1657916586_1657917209.json"
    #fname = "data/AA38932291_1657920628_1657920857.json"
    #fname = "data/AA38944134_1658001122_1658001604.json"
    sit = read_json.Situation.from_json(fname)#.cut()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fdeviated,fothers = convert_situation_to_flights(sit,initialize,device,thresh_xy=THRESH_XY_MODEL)

    test_uncertainty(sit,fdeviated,fothers,device)
    raise Exception
    f = fdeviated.fxy[0]
    t = torch.arange(0,f.duration.sum(axis=-1).max()+100,device=device).rename(T)
    xory=False
    plot(f,t,xory)
    if not xory:
        plt.gca().axis('equal')
    deviated = sit.trajectories.query("flight_id==@sit.deviated.flight_id").query("timestamp==@sit.deviated.stop")
    # plt.scatter(beacon.x,beacon.y,marker="*")
    plt.scatter(deviated.x, deviated.y)
    t_pts = torch.tensor([fdeviated.tdeviation,fdeviated.tturn,fdeviated.trejoin],dtype=torch.float,names=(WPTS,),device=device)
    #t_pts = torch.tensor([fdeviated.tturn,fdeviated.tturn,fdeviated.tturn],dtype=torch.float,names=(WPTS,),device=device)
    xy_pts = traj.generate(fdeviated.fxy[0],t_pts.rename(**{WPTS:T})).cpu()
    recplot(xy_pts,plt.scatter)
    # fnew = fdeviated.fxy[0].add_wpt_at_t(t_pts)
    # plot(fnew,t,xory)
    # raise Exception
    # print(fdeviated.fxy[0].wpts_at_t(t_pts))
    # raise Exception
    # t = torch.arange(fdeviated.fxy[0].duration.sum(),device=device).rename(T)
    # xy_pts = traj.generate(fdeviated.fxy[0],t).cpu()
    # # recplot(xy_pts,plt.scatter)
    # xy_pts2 = traj.generate(fnew,t).cpu()
    # recplot(xy_pts,plt.scatter)
    # print(xy_pts)
    # raise Exception
    # for beacon,trajp in lwpts:
    #     plt.scatter(beacon.x,beacon.y,marker="*")
    #     plt.scatter(trajp.x,trajp.y)
    plt.show()
    raise Exception
    # print(sit.trajectories.query("timestamp==@sit.deviated.start"))
    print(sit.deviated.flight_id)
    selected_fid = sit.deviated.flight_id
    # selected_fid = np.int64(38910912)
    print(sit.trajectories.groupby("flight_id").count())
    # raise Exception
    #selected_fid = np.int64(38926951)
    dfin = sit.trajectories.query("flight_id==@selected_fid")#.loc[1:]
    lres = []
    # for f in Traffic(dfin):
    #     print(f.aligned_on_navpoint(sit.deviated.beacons))pd.concat(lres,ignore_index=True)
    # df = Traffic(dfin).simplify(100.).data


    # da = dfin.copy()
    # da["track"]=da["track_angle"]
    # da["timestamp"]=pd.to_datetime(da["timestamp"],unit='s')
    # # print(df.dtypes)
    # for flight in Traffic(da):
    #     print(flight.data)
    #     print(list(flight.data))
    #     for x in flight.aligned_on_navpoint(points=sit.deviated.beacons):
    #         print(x)
    # raise Exception
    # print(df)
    # raise Exception
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sitflights = convert_situation_to_flights(sit,initialize,device)#.fxy[2]
    quality_check(sit, sitflights)

    # if not xory:
    #     plt.gca().axis('equal')
    # plot(sitflights.fxy[0],sitflights.t[0],xory)
    # xydev = traj.generate(sitflights.fxy[0],torch.tensor([sitflights.tdeviation],device=device,names=(T,))).cpu()
    # plt.scatter(xydev[...,0],xydev[...,1])
    # real = sit.trajectories.query("flight_id==@sit.deviated.flight_id").query("timestamp==@sit.deviated.start")
    # plt.scatter(real.x,real.y)
    # plt.show()
    # raise Exception

    # df = sit.trajectories#.query("flight_id == @selected_fid")
    # print(df.flight_id)
    # # raise Exception
    # df["timestamp"] = pd.to_datetime(df["timestamp"],unit="s")
    # # print(df)
    # # print(df.dtypes)
    # # raise Exception
    # icao24 = df.icao24.values[0]
    # print(icao24)
    # # plt.show()
    # # df = opensky.history(start=unix2datetime(df.timestamp.values[0]),#.strftime('%Y-%m-%d %H:%M:%S')
    # #                      stop=unix2datetime(df.timestamp.values[-1]),#.strftime('%Y-%m-%d %H:%M:%S'),
    # #                      icao24=icao24)
    # # df = df.compute_xy(projection=read_json.PROJ).data
    # plt.plot(df.x,df.y)
    # print(unix2datetime(sit.deviated.start))
    # # plt.plot(df.timestamp,df.track)
    # # plt.show()
    # # plt.scatter(*read_json.PROJ.transform([p.longitude for p in sit.deviated.beacons],[p.latitude for p in sit.deviated.beacons]),marker="*",color="red")

    # dbeacons={}
    # for flight in Traffic(df):
    #     for beacon,color in zip(sit.deviated.beacons,["red","blue","green","brown","black","yellow"]*4):
    #         lbeacon = [beacon]
    #         for x in flight.aligned_on_navpoint(points=lbeacon,**KIM_PARAMETERS):
    #             for _,line in x.data.iterrows():
    #                 dbeacons[line.timestamp]=dbeacons.get(line.timestamp,[])+[beacon]
    #             plt.scatter(*read_json.PROJ.transform([p.longitude for p in lbeacon],[p.latitude for p in lbeacon]),marker="*",color=color)
    #             plt.scatter(x.data.x,x.data.y,color=color)
    # for k,v in dbeacons.items():
    #     print(k,len(v))
    # # print(dbeacons)
    # # print(df.query("timestamp==@datetime.datetime.fromtimestamp(@sit.deviated.start,datetime.UTC)")[["timestamp","longitude","latitude"]])
    # # raise Exception

    # #df = sit.trajectories.query("flight_id == @sitflights.fid[0]")
    # #plt.scatter(df.x,df.y)

    # plt.show()
    # raise Exception

    fmodel = sitflights.fxy[0]
    DANGLE = "dangle"
    MAN_WPTS = "man_wpts"
    DT = "dt1"
    DSPEED = "dspeed"
    #wpts_start = 3+torch.tensor([0],device=device,dtype=torch.int64).reshape(1,-1).rename(BATCH,MAN_WPTS)#.reshape(-1).rename(BATCH)#,DANGLE)
    wpts_start = 6+torch.tensor([0],device=device,dtype=torch.int64).reshape(-1).rename(BATCH)#.reshape(-1).rename(BATCH)#,DANGLE)
    # print(f"{wpts_start=}")
    wpts_turn = wpts_start+2
    wpts_rejoin = wpts_turn+2
    dangle = 1 + torch.arange(1,device=device).reshape(-1).rename(DANGLE)
    dt = 300+20*torch.arange(1,device=device).reshape(-1).rename(DT)
    dspeed = 1.5+torch.arange(1,device=device).reshape(-1).rename(DSPEED)
    # dangle[0]=1#-0.3
    # print(dangle)
    #dt = 40+torch.arange(1,device=device,dtype=torch.int64).reshape(1,1).rename(BATCH,DANGLE)#torch.arangeones_like(wpts_turn)*10#np.pi/2
    # print(wpts_start,wpts_turn,wpts_rejoin)
    # raise Exception
    #fs = uncertainty.addangle(dangle,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    fs = uncertainty.addangle(dangle,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    # fs = uncertainty.adddt_old(dt,wpts_start[0].item(),(wpts_start[0].item(),wpts_turn[0].item()),fmodel)
    #fs = uncertainty.adddt_translate(dt,wpts_start,wpts_turn,fmodel)
    #fs = uncertainty.adddt_rotate(dt,wpts_start,wpts_turn,fmodel)
    # fs = uncertainty._contract(fold,fmodel,wpts_start,wpts_turn)
    #fs = uncertainty.adddt_rotate(dt,wpts_turn,wpts_rejoin,fmodel)
    # fs = uncertainty.contract(fs,fmodel,wpts_turn,wpts_rejoin,ratio)
    #fs = uncertainty.changespeed(dspeed,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    #fc = uncertainty.addangle_old(dangle,wpts_start[0,0].item(),(wpts_start[0,0].item(),wpts_turn[0,0].item()),fmodel)
    # fs = uncertainty.adddt(dangle,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    # fc = uncertainty.adddt_old(dangle,wpts_start[0].item(),(wpts_start[0].item(),wpts_turn[0].item()),fmodel)
    # fs = initialize(df,t_zero,device)
    #t = t_from_df(dfin,device)
    t = torch.arange(0.,fmodel.duration.sum(axis=-1).max()+400,device=device).rename(T)
    #torch.tensor(dfin.timestamp.values-dfin.timestamp.values[0],device=device,dtype=torch.float32).rename(T)
    trajreal = trajreal_from_df(dfin,device)
    print(fmodel.duration)
    #f = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    f = fmodel
    print(f.meanv())
    print(f.duration)
    print(fs.meanv())


    # f = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    # print(f)
    # print(fc)
    # print(fs)
    # trajreal = trajreal.cpu()
    # plt.scatter(dfin.x,dfin.y)
    # xy  = traj.generate(f,t).cpu()
    # plt.scatter(xy[0,:,0],xy[0,:,1])

    plot(f,t,xory)
    plot(fs,t,xory)
    if not xory:
        plt.gca().axis('equal')
    # plot_xy(xy,wpts)
    # def printxy(xy,wpts):
    # n_dim_param = xy.ndim() - 2
    # for ibatch  in range(xy.shape[0]):
    #     for iangle  in range(xy.shape[1]):
    #         xyc  = xy[ibatch,iangle]
    #         wptsc  = wpts[ibatch,iangle]
    #         plt.scatter(xyc[:,0],xyc[:,1])
    #         plt.scatter(wptsc[:,0],wptsc[:,1])
    # plt.scatter(df.x,df.y)
    # plt.gca().axis('equal')
    # print(f)
    # print(f.compute_wpts().cpu())
    plt.show()

    raise Exception
    print(t.shape)
    print(xy[0,:,1].shape)
    # plt.scatter(torch.arange(0,t.max()),xy[0,:,1])
    # xy = traj.generate(f,t)
    print(fmodel.meanv())
    print(fs.meanv())
    errx=dfin.x.values-xy[0,:,0].cpu().numpy()
    erry=dfin.y.values-xy[0,:,1].cpu().numpy()
    plt.scatter(t.cpu(),np.sqrt(errx**2+erry**2))
    xy  = traj.generate(fs,t).cpu()
    errx=dfin.x.values-xy[0,:,0].cpu().numpy()
    erry=dfin.y.values-xy[0,:,1].cpu().numpy()
    plt.scatter(t.cpu(),np.sqrt(errx**2+erry**2))
    plt.show()

    t = torch.arange(0,t.max(),device=device,dtype=torch.float32).rename(T)
    xy =  traj.generate(f,t).cpu()
    dist = xy[0,:,0]#np.sqrt(np.diff(xy[0,:,0]) ** 2 + np.diff(xy[0,:,1]) ** 2)
    plt.plot(dist)
    # plt.plot(dist/np.diff(t.cpu()))
    xy  = traj.generate(fs,t).cpu()
    dist = xy[0,:,0]#np.sqrt(np.diff(xy[0,:,0]) ** 2 + np.diff(xy[0,:,1]) ** 2)
    plt.plot(dist)
    # dist = np.sqrt(np.diff(dfin.x.values) ** 2 + np.diff(dfin.y.values) ** 2)
    # plt.plot(dist/np.diff(dfin.timestamp.values))
    plt.show()
    t = torch.arange(0,t.max(),device=device,dtype=torch.float32).rename(T)
    xy =  traj.generate(f,t).cpu()
    dist = np.sqrt(np.diff(xy[0,:,0]) ** 2 + np.diff(xy[0,:,1]) ** 2)
    plt.plot(dist/np.diff(t.cpu()))
    xy  = traj.generate(fs,t).cpu()
    dist = np.sqrt(np.diff(xy[0,:,0]) ** 2 + np.diff(xy[0,:,1]) ** 2)
    plt.plot(dist/np.diff(t.cpu()))
    plt.show()


    # fopti = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    # xy  = traj.generate(fopti,t)
    # print(xy,xy.shape)
    # print(trajreal)
    # print(fopti)
    # for fid in sit.trajectories.flight_id.unique():
    #     traj = sit.trajectories.query("flight_id==@fid").query("timestamp==@sit.deviated.start")
    #     plt.scatter(traj.longitude,traj.latitude)
    # plt.show()

if __name__ == "__main__":
    main()
