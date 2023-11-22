import numpy as np

H = 168
S = 52

# #Thermal units and energy not served parameters (the energy not served is modeled as an expensive thermal unit with infinite capacity )
# cTh = [0.1, 0.2, 0.3, 3] # euro/MWh
# PminTh = [0.0, 0.0, 0.0, 0.0]
# PmaxTh = [42500.0, 12500.0, 7500.0, float('inf')] # MW

# # Battery
# x0 = 0.0
# PminB = -5.e4 #MW turbinage max
# PmaxB = 5.e4 # pompage max
# XminB = 0.0
# rho = 200 
# XmaxB = rho*PmaxB         #The maximum capacity changes for the different values of rho considered 
# xNsteps = 51

# courbes_guides = np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/hydro/common/capacity/reservoir_area.txt")[6:365:7,[0,2]]*XmaxB # on récupère les valeurs du dimanche
# Xmin = courbes_guides[:,0]
# Xmax = courbes_guides[:,1]
# Xmin = np.concatenate((Xmin,Xmin[[0]]))
# Xmax = np.concatenate((Xmax,Xmax[[0]]))

# NTrain = 10

# Demand = np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/load/series/load_area.txt")[:8736, :NTrain]

# LAW_s = np.reshape(Demand,(S,H,NTrain)) 
# proba_s = np.ones((S,NTrain))/NTrain

# apport =np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/hydro/series/area/mod.txt")[6:365:7,:NTrain]*7/H 

# pen_low = 3.1
# pen_high = 3.1


class ThermalUnit:

    def __init__(self, marginal_cost, P_min, P_max, name):
        self.marginal_cost = marginal_cost
        self.P_min = P_min
        self.P_max = P_max
        assert("_" not in name)
        self.name = name

    def __str__(self) -> str:
        return(self.name)

class Reservoir:

    def __init__(self, initial_level, P_turb, P_pump, rho, dir_study, name_area, name):
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

class Area:

    def __init__(self, thermal_units, reservoirs, dir_study, name_area, name):
        self.list_thermal_units = thermal_units
        self.list_reservoirs = reservoirs

        load = np.loadtxt(dir_study+"/input/load/series/load_"+name_area+".txt")[:8736]
        mc = load.shape[1]
        self.load = np.reshape(load,(S,H,mc))
        assert("_" not in name)
        self.name = name

    def __str__(self):
        return(self.name)

class Study:

    def __init__(self, areas, pen_low, pen_high, cost_ens, nb_mc, links):
        self.list_areas = areas
        self.pen_low = pen_low
        self.pen_high = pen_high
        self.cost_ens = cost_ens
        self.nb_mc = nb_mc
        self.links = links