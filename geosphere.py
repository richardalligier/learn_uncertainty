import numpy as np
import pyproj
from pyproj import Geod, CRS
import pygplates
import shapely
import shapely.ops

NM2METER = 1852


def diff_track(a,b):
    d = a-b
    return d + (d<-180)*360 - (d>=180)*360

def distance_degree(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2 = [np.radians(x) for x in (lat1,lon1,lat2,lon2)]
    r = 6370997#6371000
    phi1 = lat1
    phi2 = lat2
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res

def orthodromy(lat1,lon1,lat2,lon2,npts):
    geod = Geod(ellps="sphere")
    return np.array([
        (lat, lon)
        for (lon, lat) in geod.npts(lon1, lat1, lon2, lat2, npts=npts,initial_idx=0,terminus_idx=0)
    ])


def loxodromy(lat1,lon1,lat2,lon2,npts):
    proj = pyproj.Proj(proj="merc",ellps='sphere')
    p1=np.array(proj.transform(lon1,lat1))
    p2=np.array(proj.transform(lon2,lat2))
    ps = np.linspace(p1,p2,npts)
    lon,lat=proj.transform(ps[:,0],ps[:,1],direction=pyproj.enums.TransformDirection.INVERSE)
    return np.array([lat,lon]).T


def distance_without_time_exact(lat1,lon1,lat2,lon2):
    assert(lat1.ndim==1)
    assert(lat2.ndim==1)
    assert(lon1.ndim==1)
    assert(lon2.ndim==1)
    r = distance_degree(np.expand_dims(lat1,axis=1),np.expand_dims(lon1,axis=1),lat2,lon2)
    return r.min(axis=1)



# def lag(horizon: int, v):
#     res = np.empty((horizon, v.shape[0]))
#     res.fill(np.nan)
#     for i in range(horizon):
#         res[i, : v.shape[0] - i] = v[i:]
#     return res

# def lagfuture(horizon: int, v):
#     res = np.empty((horizon, v.shape[0]))
#     res.fill(np.nan)
#     n = v.shape[0]
#     for i in range(horizon):
#         res[i,  i:] = v[:n-i]
#     return res
# def center(horizon,v):
#     return np.concatenate([lag(horizon,v),lagfuture(horizon,v)],axis=0)

# def distance_without_time(lat1,lon1,lat2,lon2):
# #    print(f"{lat2.shape=}")
#     assert(lat1.ndim==1)
#     assert(lat2.ndim==1)
#     assert(lon1.ndim==1)
#     assert(lon2.ndim==1)
#     n = lat2.shape[0]
#     h = min(200,n)
#     hlat2 = center(h,lat2) # h x n
# #    print(f"{hlat2.shape=}")
#     #raise Exception
#     hlon2 = center(h,lon2) # h x n
#     r = distance_degree(np.expand_dims(lat1,axis=0), # 1 x n
#                         np.expand_dims(lon1,axis=0), # 1 x n
#                         hlat2, # h x n
#                         hlon2) # h x n
#     # r lat2 x lat1 
#     return np.nanmin(r,axis=0)

def distance_ortho_sampling(lat1,lon1,lat2,lon2,latm,lonm):
    assert(type(lat1)==np.float64)
    assert(type(lon1)==np.float64)
    assert(type(lat2)==np.float64)
    assert(type(lon2)==np.float64)
    assert(latm.ndim==1)
    assert(lonm.ndim==1)
    dlat = diff_track(lat1,lat2)
    dlon = diff_track(lon1,lon2)
    npts = int(np.sqrt(dlat**2+dlon**2)*NM2METER*10)+2
    # print(npts)
    lat,lon = orthodromy(lat1,lon1,lat2,lon2,npts).T
    return distance_without_time_exact(latm,lonm,lat,lon)


def distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm):
    assert(type(lat1)==np.float64)
    assert(type(lon1)==np.float64)
    assert(type(lat2)==np.float64)
    assert(type(lon2)==np.float64)
    assert(latm.ndim==1)
    assert(lonm.ndim==1)
    p1 = pygplates.PointOnSphere(latitude=lat1,longitude=lon1)
    p2 = pygplates.PointOnSphere(latitude=lat2,longitude=lon2)
    pm = [pygplates.PointOnSphere(latitude=latmi,longitude=lonmi) for (latmi,lonmi) in zip(latm,lonm)]
    gc = pygplates.PolylineOnSphere([p1,p2])#,is_solid=True)
    # print([p.to_lat_lon() for p in pm])
    # la = []
    # lb = []
    # for pmi in pm:
    #     di,pa,pb = pygplates.GeometryOnSphere.distance(gc,pmi,return_closest_positions=True)
    #     la.append(pa.to_lat_lon())
    #     lb.append(pb.to_lat_lon())
    # alat,alon = np.array(la).T
    # blat,blon = np.array(lb).T
    # dmano = distance_degree(alat,alon,blat,blon)
    # return dmano
        
    #     print(di,pa.to_lat_lon(),pb.to_lat_lon(),pmi.to_lat_lon(),p1.to_lat_lon(),p2.to_lat_lon())
    return 6370997 * np.array([  pygplates.GeometryOnSphere.distance(gc,pmi) for pmi in pm])


def distance_loxo(lat1,lon1,lat2,lon2,latm,lonm,nptsfactor=100):
    assert(type(lat1)==np.float64)
    assert(type(lon1)==np.float64)
    assert(type(lat2)==np.float64)
    assert(type(lon2)==np.float64)
    assert(latm.ndim==1)
    assert(lonm.ndim==1)
    proj = pyproj.Proj(proj="merc",ellps='sphere')
    p1=np.array(proj.transform(lon1,lat1))
    p2=np.array(proj.transform(lon2,lat2))
    dlat = diff_track(lat1,lat2)
    dlon = diff_track(lon1,lon2)
    npts = int(np.sqrt(dlat**2+dlon**2)*nptsfactor)+2# if npts is None else npts
    ps = np.linspace(p1,p2,npts)
    lon,lat=proj.transform(ps[:,0],ps[:,1],direction=pyproj.enums.TransformDirection.INVERSE)
    pm = [pygplates.PointOnSphere(latmi,lonmi) for (latmi,lonmi) in zip(latm,lonm)]
    gc = pygplates.PolylineOnSphere([pygplates.PointOnSphere(lati,loni) for lati,loni in zip(lat,lon)])
    return 6370997 * np.array([  pygplates.GeometryOnSphere.distance(gc,pmi) for pmi in pm])


def distance_loxo_fast(lat1,lon1,lat2,lon2,latm,lonm):# slow and inaccurate as it is right now
    assert(type(lat1)==np.float64)
    assert(type(lon1)==np.float64)
    assert(type(lat2)==np.float64)
    assert(type(lon2)==np.float64)
    assert(latm.ndim==1)
    assert(lonm.ndim==1)
    proj = pyproj.Proj(proj="merc",ellps='sphere')
    p1 = np.array(proj.transform(lon1,lat1))
    p2 = np.array(proj.transform(lon2,lat2))
    xpm,ypm = np.array(proj.transform(lonm,latm))
    # print(lonm,latm)
    # print(xpm,ypm)
    # raise Exception
    seg = shapely.LineString([p1,p2])
    # print(xpm)
    # print(ypm)
    nearest = [ shapely.ops.nearest_points(seg,shapely.Point(*pmi))[0] for pmi in zip(xpm,ypm)]
    xnearest = [ p.x for p in nearest]
    ynearest = [ p.y for p in nearest]
    lonc,latc=proj.transform(xnearest,ynearest,direction=pyproj.enums.TransformDirection.INVERSE)
    # print(lonc,latc)
    # print(lonm,latm)
    return distance_degree(latc,lonc,latm,lonm)



def distance_loxo_ortho(lat1,lon1,lat2,lon2,nptsfactor=1000):
    assert(lat2==lat2)
    assert(lat1==lat1)
    assert(lon2==lon2)
    assert(lon1==lon1)
    assert((lat1!=lat2) or (lon1!=lon2))
    # print(lat1,lon1,lat2,lon2)
    dlat = diff_track(lat1,lat2)
    dlon = diff_track(lon1,lon2)
    npts = int(np.sqrt(dlat**2+dlon**2)*nptsfactor)+2
    # print(f"{np.sqrt(dlat**2+dlon**2)*nptsfactor=} {npts=}")
    # print(npts)
    latm,lonm = loxodromy(lat2,lon2,lat1,lon1,npts).T
    # print(latm)
    # d=distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm)
    # print(d)
    # if d.max()==0:
    #     raise Exception
    # k = npts//2
    # print(latm[k:k+1],lonm[k:k+1])
    # true=distance_ortho_sampling(lat1,lon1,lat2,lon2,latm,lonm).max()
    return distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm).max()#,0.#true





def main():
    import time
    n = 100
    low = 49
    high= 50
    lat1,lat2,lon1,lon2 = np.random.uniform(low=low,high=high,size=4)
    latm,lonm = np.random.uniform(low=low,high=high,size=(2,n))
#    latm,lonm = np.array([(lat1+lat2)/2]),np.array([(lon1+lon2)/2])
    # print(latm)
    print(lat1,lat2,lon1,lon2)
    # t0 = time.time()
    # ds = distance_ortho_sampling(lat1,lon1,lat2,lon2,latm,lonm)
    # print(time.time()-t0)
    # t0 = time.time()
    # dg = distance_ortho_pygplates(lat1,lon1,lat2,lon2,latm,lonm)
    # print(time.time()-t0)
    # print(ds-dg)
    # t0 = time.time()
    # ds = distance_loxo(lat1,lon1,lat2,lon2,latm,lonm)
    # print(time.time()-t0)
    # t0 = time.time()
    # dg = distance_loxo(lat1,lon1,lat2,lon2,latm,lonm,nptsfactor=1000)#distance_loxo_fast(lat1,lon1,lat2,lon2,latm,lonm)
    # print(time.time()-t0)
    # print(ds-dg)
    # t0 = time.time()
    # ds = distance_loxo_ortho(lat1,lon1,lat2,lon2)#,latm,lonm)
    # print(time.time()-t0)
    t0 = time.time()
    # ds=distance_loxo_ortho(lat1,lon1,lat2,lon2)
    dg,ds = distance_loxo_ortho(lat1,lon1,lat2,lon2)#distance_loxo_fast(lat1,lon1,lat2,lon2,latm,lonm)
    # print(time.time()-t0)
    print(ds-dg,ds,dg)
    

if __name__ == "__main__":
    np.random.seed(400)
    for i in range(40):
        main()

