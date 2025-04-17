import numpy as np

class ResLine:
    def __init__(self,i,j,d):#,lats,lons):#,dolmax):

        self.interval=(i,j)
        self.idmax = np.argmax(d)
        self.dmax = d[self.idmax]
    def length(self):
        i,j = self.interval
        return j-i+1
    def __repr__(self):
        return f"{self.interval} {self.dmax} "


def buildTemporalEuclide(xy,t,i,j):
    dt = t[j]-t[i]
    lam = ((t[i:j+1]-t[i]) / dt)[:,None]
    exy = xy[i] * (1-lam) + lam * xy[j] - xy[i:j+1]
    assert(exy.shape[-1]==4)
    d = np.maximum(np.linalg.norm(exy[...,2:],axis=-1),np.linalg.norm(exy[...,:2],axis=-1))#np.hypot(exy[...,0],exy[...,1])
    assert(d.shape[0]==j-i+1)
    return ResLine(i,j,d)


def buildEuclide(xy,t,i,j):
    v = xy[j]-xy[i]
    # print(dir(np.linalg))
    lam = np.sum(v[None,:]*(xy[i:j+1]-xy[i]),axis=-1)/np.dot(v,v)
    lam = np.clip(lam,a_min=0,a_max=1)
    exy = xy[i] + v * lam[:,None] - xy[i:j+1]
    assert(exy.shape[-1]==4)
    d = np.maximum(np.linalg.norm(exy[...,2:],axis=-1),np.linalg.norm(exy[...,:2],axis=-1))#np.hypot(exy[...,0],exy[...,1])
    # d = np.linalg.norm(exy,axis=-1)
    assert(d.shape[0]==j-i+1)
    return ResLine(i,j,d)


def douglas_peucker(xy,t,eps,build=buildTemporalEuclide):
    def aux(i,j):
        # print(i,j)
        assert(i<j)
        assert((xy[i]==xy[i]).all())
        assert((xy[j]==xy[j]).all())
        r = build(xy,t,i,j)
        if r.dmax < eps:# and ((nanmaxrdo * 0.9 >  nanmaxrdl) or (nanmaxrdl * 0.9  >  nanmaxrdo)):
            return [r]
        else:
            k = i + r.idmax
            # print(f"{i=} {j=} {k=} {np.isnan(lats).mean()=} {np.isnan(lons).mean()=} {r.dmax=} {r.dlmax=} {r.domax=}")
            assert (not (k==i or k==j))
            return aux(i,k) + aux(k,j)
    n = xy.shape[0]
    res = aux(0,n-1)
    mask = np.zeros(n,dtype=bool)
    # print(mask)
    for r in res:
        # print(r)
        (i,j) = r.interval
        mask[i] = True
        mask[j] = True
    # print(mask)
    return mask#[resline.reindex(indexes) for resline in res]
