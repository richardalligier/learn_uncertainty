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

KIM_PARAMETERS = {"angle_precision":0.5,"min_distance":200.}
NM2METERS = 1852

BATCH = "batch"
DTYPE = torch.float32


def unix2datetime(u):
    return datetime.datetime.fromtimestamp(u,datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')


def check_is_named_tensor(t,dtype):
    return isinstance(t,torch.Tensor) and t.dtype == dtype

# modelisation de l'altitude ?????
# axe supplémentaire pour gérer divers "trous temporels"
class SituationOthers:
    def __init__(self,fid,fxy,fz,t):
        assert check_is_named_tensor(fid,torch.int64)
        assert check_is_named_tensor(t,DTYPE)
        self.fid = fid
        self.fxy = fxy
        self.fz = fz
        self.t = t

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


def quality_check(sit, sitflights):
    t_zero_situation = sit.trajectories.timestamp.min()
    xy = traj.generate(sitflights.fxy,sitflights.t).cpu()
    for i,fid in enumerate(sitflights.fid):
        print(fid)
        df = sit.trajectories.query("flight_id==@sitflights.fid[@i].item()")
        assert((sitflights.t[i].cpu() == df.timestamp.values-sit.trajectories.timestamp.min()).all())
        assert(sitflights.tdeviation==sit.deviated.start-t_zero_situation)
        dxy = xy[i]-df[["x","y"]].values
        dist = torch.hypot(dxy[...,0],dxy[...,1])
        print(dist.min(),dist.mean(),dist.max())
        # raise Exception
        # dfin = sit..query("flight_id == @fid")


def trajreal_from_df(df,device):
    return torch.tensor(np.array([df.x.values,df.y.values]),device=device,dtype=torch.float32).unsqueeze(0).rename(BATCH,XY,T).align_to(BATCH,T,XY)

def t_from_df(df,device):
    return torch.tensor(df.timestamp.values,device=device,dtype=torch.float32).rename(T)

def initialize_gen(df,device):#trajreal,t):
    trajreal = trajreal_from_df(df,device)
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
    # print(t)
    # print(duration,duration.shape)
    v = dxy / duration
    wpts = trajreal[...,1:,:]
    wpts = wpts.rename(**{T:WPTS})
    v = v.rename(**{T:WPTS})
    turn_rate = 0.01 * torch.ones((1,t.shape[-1]),device=device).rename(BATCH,WPTS)
    #turn_rate = torch.ones_like(duration).rename(*duration.names)#.rename(*duration.names)
    print(turn_rate.names,duration.names)
    return {"xy0":xy0,"v":v, "turn_rate":turn_rate, "wpts":wpts}

def initialize(df,device):#(trajreal,t):
    res = initialize_gen(df,device)
    return flights.Flights.from_wpts(**res)

def initialize_acc(df,device):#(trajreal,t):
    t_zero,d=initialize_gen(df,device)
    meanv = d["v"].align_to(...,WPTS)
    v = meanv.clone().detach()
    for i in range(v.shape[-1]-2,-1,-1):
        v[...,i] = - v[...,i+1] + 2 * meanv[...,i]
    d["v"]=v
    f = flights.FlightsWithAcc.from_wpts(**d)
    return  f

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
def convert_situation_to_flights(sit,initialize,device,thresh=100):
    df = sit.trajectories.copy()
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
            mask = douglas_peucker.douglas_peucker(dfin[["x","y"]].values,dfin.timestamp.values,thresh)
            mask = np.logical_or(mask, dfin.timestamp.values==t_deviation)
            mask = np.logical_or(mask, dfin.timestamp.values==t_turn)
            mask = np.logical_or(mask, dfin.timestamp.values==t_rejoin)
            dfwpts = dfin.where(pd.Series(mask,index=dfin.index)).dropna(subset=["x"]).reset_index()
            f = initialize(dfwpts,device).shift_xy0(float(dfwpts.timestamp.values[0]))
            lfxy.append(f)
            lfz.append(f)
            lt.append(dfin.timestamp.values)
            # lt.append(torch.tensor(dfin.timestamp.values,device=device,names=(T,)))
        lfid = torch.tensor(lfid,device = device,dtype=torch.int64,names=(BATCH,))
        lt = torch.tensor(lt,device=device,dtype=DTYPE,names=(BATCH,T))
        # print(lt)
        return {"fid":lfid,"fxy":lfxy,"fz":lfz,"t":lt}
    ddeviated = convert(df.query("flight_id==@sit.deviated.flight_id"))
    ddeviated["fxy"] = flights.cat_lflights(ddeviated["fxy"])
    ddeviated["fz"] = flights.cat_lflights(ddeviated["fz"])
    ddeviated["tdeviation"] = torch.tensor([t_deviation],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["tturn"] = torch.tensor([t_turn],device=device,dtype=DTYPE,names=(BATCH,))
    # print(beacon_rejoin.x)
    ddeviated["trejoin"] = torch.tensor([t_rejoin],device=device,dtype=DTYPE,names=(BATCH,))
    ddeviated["beacon"] = torch.tensor([[beacon_rejoin.x,beacon_rejoin.y]],device=device,dtype=DTYPE,names=(BATCH,XY))
    deviated = SituationDeviated(**ddeviated)
    return deviated
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


def test_uncertainty(sit,fdeviated,device):
    # fname = "data/AA38909998_1657916586_1657917209.json"
    # #fname = "data/AA38932291_1657920628_1657920857.json"
    # #fname = "data/AA38944134_1658001122_1658001604.json"
    # sit = read_json.Situation.from_json(fname)#.cut()
    # print(sit.deviated.flight_id)
    selected_fid = sit.deviated.flight_id
    print(sit.trajectories.groupby("flight_id").count())
    dfin = sit.trajectories.query("flight_id==@selected_fid")#.loc[1:]
    xory=False
    DANGLE = "dangle"
    MAN_WPTS = "man_wpts"
    DT = "dt1"
    DSPEED = "dspeed"
    print(fdeviated.fxy.duration.names)
    print(fdeviated.tdeviation)
    fmodel = fdeviated.fxy
    dtwpts = {}
    for v in ["tdeviation","tturn","trejoin"]:
        dtwpts[v] = getattr(fdeviated,v)
        fmodel = fmodel.add_wpt_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
        print(fmodel.duration.cumsum(axis=-1))
    diwpts = {}
    for v in ["tdeviation","tturn","trejoin"]:
        diwpts[v]= fmodel.wpts_at_t(named.unsqueeze(dtwpts[v],-1,WPTS))
    # wpts_start = 5+torch.tensor([0],device=device,dtype=torch.int64).reshape(-1).rename(BATCH)#.reshape(-1).rename(BATCH)#,DANGLE)
    # wpts_turn = fmodel.wpts_at_t(fdeviated.tturn)
    # wpts_rejoin = fmodel.wpts_at_t(fdeviated.trejoin)
    for v in ["tdeviation","tturn","trejoin"]:
        print(f"{v} {diwpts[v]=}")
    dangle = 0.2+torch.arange(1,device=device).reshape(-1).rename(DANGLE)
    # dt = 300+20*torch.arange(1,device=device).reshape(-1).rename(DT)
    dt = 50+torch.arange(1,device=device).reshape(-1).rename(DT)
    dspeed = 1.5+torch.arange(1,device=device).reshape(-1).rename(DSPEED)
    # dangle[0]=1#-0.3
    # print(dangle)
    #dt = 40+torch.arange(1,device=device,dtype=torch.int64).reshape(1,1).rename(BATCH,DANGLE)#torch.arangeones_like(wpts_turn)*10#np.pi/2
    # print(wpts_start,wpts_turn,wpts_rejoin)
    # raise Exception
    #fs = uncertainty.addangle(dangle,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    fs = uncertainty.addangle(dangle,diwpts["tdeviation"],diwpts["tturn"],diwpts["trejoin"],fmodel)
    modified_trejoin = uncertainty.gather_wpts(fs.duration.cumsum(axis=-1),diwpts["trejoin"]-1)
    print(f"{fdeviated.trejoin=}")
    print(f"{modified_trejoin=}")
    # fs = uncertainty.adddt_old(dt,wpts_start[0].item(),(wpts_start[0].item(),wpts_turn[0].item()),fmodel)
    # fs = uncertainty.adddt_translate(dt,diwpts["tdeviation"],diwpts["tturn"],fmodel)
    # fs = uncertainty.adddt_translate(dt,wpts_start,wpts_turn,fmodel)
    # fs = uncertainty.adddt_rotate(dt,wpts_start,wpts_turn,fmodel)
    # fs = uncertainty._contract(fold,fmodel,wpts_start,wpts_turn)
    #fs = uncertainty.adddt_rotate(dt,wpts_turn,wpts_rejoin,fmodel)
    # fs = uncertainty.contract(fs,fmodel,wpts_turn,wpts_rejoin,ratio)
    #fs = uncertainty.changespeed(dspeed,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    #fc = uncertainty.addangle_old(dangle,wpts_start[0,0].item(),(wpts_start[0,0].item(),wpts_turn[0,0].item()),fmodel)
    # fs = uncertainty.adddt(dangle,wpts_start,wpts_turn,wpts_rejoin,fmodel)
    # fc = uncertainty.adddt_old(dangle,wpts_start[0].item(),(wpts_start[0].item(),wpts_turn[0].item()),fmodel)
    # fs = initialize(df,t_zero,device)
    # t = t_from_df(dfin,device)
    # t = torch.arange(0.,fmodel.duration.sum(axis=-1).max()+400,0.001,device=device).rename(T)
    max_duration = (fdeviated.trejoin - fdeviated.tdeviation).max().item()
    print(max_duration)
    t = torch.arange(start=0.,end=max_duration,step=0.001,device=device).rename(T)
    print(t)
    print(fdeviated.tdeviation)
    a,b = named.align_common(fdeviated.tdeviation,t)
    t = (a + b).align_to(...,T)
    t[t > a]=float("nan")
    print(t.names)
    # raise Exception
    #t = torch.arange(fdeviated.tdeviation.min(),fdeviated.trejoin.max(),0.001,device=device).unsqueeze(0).rename(BATCH,T)
    # t = torch.arange(t.min(),t.max()+400,device=device).rename(T)
    #torch.tensor(dfin.timestamp.values-dfin.timestamp.values[0],device=device,dtype=torch.float32).rename(T)
    trajreal = trajreal_from_df(dfin,device)
    print(fmodel.duration)
    #f = fit.fit(fmodel,trajreal,t,traj.generate,10000)
    f = fmodel
    # print(f.meanv())
    # print(f.duration)
    # print(fs.meanv())
    plot(f,t,xory)
    plot(fs,t,xory)
    recplot(fdeviated.beacon.cpu(),plt.scatter)
    if not xory:
        plt.gca().axis('equal')
    print(f.duration)
    print(fs.duration)
    print(t)
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
    fdeviated = convert_situation_to_flights(sit,initialize,device,thresh=100)
    quality_check(sit,fdeviated)
    test_uncertainty(sit,fdeviated,device)
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
