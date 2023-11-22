import numpy as np
from typing import NewType

H = 168
S = 52

class ThermalUnit:

    def __init__(self, marginal_cost: float, fixed_cost: float, start_up_cost:float, P_min:float, P_max:float, name:str):
        self.marginal_cost = marginal_cost
        self.fixed_cost = fixed_cost
        self.start_up_cost = start_up_cost
        self.P_min = P_min
        self.P_max = P_max
        assert("_" not in name)
        self.name = name

    def __str__(self) -> str:
        return(self.name)

class Reservoir:

    def __init__(self, initial_level:float, P_turb:float, P_pump:float, rho:float, dir_study:str, name_area:str, name:str):
        self.initial_level = initial_level
        self.P_turb = P_turb
        self.P_pump = P_pump
        self.capacity = -rho*P_turb

        courbes_guides = np.loadtxt(dir_study+"/input/hydro/common/capacity/reservoir_"+name_area+".txt")[6:365:7,[0,2]]*self.capacity # on récupère les valeurs du dimanche
        Xmin = courbes_guides[:,0]
        Xmax = courbes_guides[:,1]
        self.Xmin = np.concatenate((Xmin,Xmin[[0]]))
        self.Xmax = np.concatenate((Xmax,Xmax[[0]]))

        self.inflow = np.loadtxt(dir_study+"/input/hydro/series/"+name_area+"/mod.txt")[6:365:7]*7/H 
        assert("_" not in name)
        self.name = name

ThermalUnits = NewType("ThermalUnits",list[ThermalUnit])
Reservoirs = NewType("Reservoirs",list[Reservoir])

class Area:

    def __init__(self, thermal_units:ThermalUnits, reservoirs:Reservoirs, dir_study:str, name_area:str, name:str):
        self.list_thermal_units = thermal_units
        self.list_reservoirs = reservoirs

        load = np.loadtxt(dir_study+"/input/load/series/load_"+name_area+".txt")[:8736]
        mc = load.shape[1]
        self.load = np.reshape(load,(S,H,mc))
        assert("_" not in name)
        self.name = name

    def __str__(self):
        return(self.name)

Areas = NewType("Areas",list[Area])
Links = NewType("Links",dict[str,dict[str,tuple[float,float]]])
class Study:

    def __init__(self, areas:Areas, pen_low:float, pen_high:float, cost_ens:float, nb_mc:int, links:Links):
        self.list_areas = areas
        self.pen_low = pen_low
        self.pen_high = pen_high
        self.cost_ens = cost_ens
        self.nb_mc = nb_mc
        self.links = links