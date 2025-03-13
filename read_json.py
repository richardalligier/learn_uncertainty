import json
import pandas as pd
import numpy as np
from filterclassic import FilterCstLatLon
from traffic.core import Traffic, mixins
import pyproj
# import pyproj

# predicted_pairwise, nb_aircraft_at_tcpa pas bon
# repeated position at different time

TRAJECTORIES = "trajectories"
ICAO24 = "icao24"
POINTS = "points"
FLIGHT_ID = "flight_id"
START_DEVIATION = "start_deviation"
STOP_DEVIATION = "stop_deviation"
DEVIATED_AIRCRAFT = "deviated_aircraft"
FLIGHT_PLAN = "flight_plan"

def nointerpolate(x):
    ''' identity function '''
    return x

PROJ = pyproj.Proj( proj="lcc",
                    ellps="WGS84",
                    lat_1=41.,#self.data.latitude.min(),
                    lat_2=47.,#self.data.latitude.max(),
                    lat_0=44.,#self.data.latitude.mean(),
                    lon_0=0.,#self.data.longitude.mean(),
                   )

class Point(mixins.PointMixin):
    def __init__(self,name,latitude,longitude,proj=PROJ):
        self.longitude = longitude
        self.latitude = latitude
        self.x,self.y = proj.transform([longitude],[latitude])
        self.name = name
    def __repr__(self):
        return f"({self.longitude}, {self.latitude})"




class Deviated_aircraft:
    def __init__(self,json_deviated_aircraft):
        self.start = json_deviated_aircraft[START_DEVIATION]
        self.stop = json_deviated_aircraft[STOP_DEVIATION]
        self.flight_id = np.int64(json_deviated_aircraft[FLIGHT_ID])
        self.beacons = [Point(line.name,line.latitude,line.longitude) for _,line in pd.json_normalize(json_deviated_aircraft[FLIGHT_PLAN]).iterrows()]
    def __str__(self):
        return f"deviation (start,stop): ({self.start},{self.stop})"

# class Projection:
#     def __init__(self,df):
#         projection = pyproj.Proj(
#             proj="lcc",
#             ellps="WGS84",
#             lat_1=df.latitude.min(),
#             lat_2=df.latitude.max(),
#             lat_0=df.latitude.mean(),
#             lon_0=df.longitude.mean(),
#         )
#         self.transformer = pyproj.Transformer.from_proj(
#             pyproj.Proj("epsg:4326"), projection, always_xy=True
#         )
#     def transform(self,df):
#         x, y = self.transformer.transform(df.longitude.values,df.latitude.values)
#         return df.assign(x=x,y=y)


class  Situation:
    def __init__(self,trajectories,deviated):
        self.trajectories = trajectories
        self.deviated = deviated
    @staticmethod
    def from_json(fname):
        with open(fname,'r') as f:
            fjson = json.load(f)
        ltrajs = []
        trajs = fjson[TRAJECTORIES]
        for k,v in trajs.items():
            dfpoints = pd.json_normalize(v["points"]).sort_values(["timestamp"])
            dfpoints[ICAO24]=v["icao24"]
            dfpoints[FLIGHT_ID]=np.int64(k)
            for vname in [ICAO24]:
                dfpoints[vname]=dfpoints[vname].astype("string")
            dfpoints = Traffic(dfpoints).filter(filter=FilterCstLatLon(),strategy=nointerpolate).eval(max_workers=1).data.dropna(subset=["latitude"])
            ltrajs.append(dfpoints)
        trajectories = pd.concat(ltrajs,ignore_index=True)
        trajectories = trajectories.rename(columns={"track_angle":"track"})
        trajectories = Traffic(trajectories).compute_xy(projection=PROJ).data
        # projection = Projection(trajectories)
        # trajectories = projection.transform(trajectories)
        deviated = Deviated_aircraft(fjson[DEVIATED_AIRCRAFT])
        trajectories = trajectories.query("flight_id == @deviated.flight_id")
        return Situation(trajectories,deviated)
    def cut(self,start=None,stop=None):
        if start is None:
            start = self.deviated.start
        if stop is None:
            stop = self.deviated.stop
        return Situation(self.trajectories.query("@start<=timestamp<=@stop"),self.deviated)




