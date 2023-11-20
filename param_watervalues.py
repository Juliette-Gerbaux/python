import numpy as np

H = 168
S = 52

#Thermal units and energy not served parameters (the energy not served is modeled as an expensive thermal unit with infinite capacity )
cTh = [0.1, 0.2, 0.3, 3] # euro/MWh
PminTh = [0.0, 0.0, 0.0, 0.0]
PmaxTh = [42500.0, 12500.0, 7500.0, float('inf')] # MW

# Battery
x0 = 0.0
PminB = -5.e4 #MW turbinage max
PmaxB = 5.e4 # pompage max
XminB = 0.0
rho = 200 
XmaxB = rho*PmaxB         #The maximum capacity changes for the different values of rho considered 
xNsteps = 51

courbes_guides = np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/hydro/common/capacity/reservoir_area.txt")[6:365:7,[0,2]]*XmaxB # on récupère les valeurs du dimanche
Xmin = courbes_guides[:,0]
Xmax = courbes_guides[:,1]
Xmin = np.concatenate((Xmin,Xmin[[0]]))
Xmax = np.concatenate((Xmax,Xmax[[0]]))

NTrain = 10

Demand = np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/load/series/load_area.txt")[:8736, :NTrain]

LAW_s = np.reshape(Demand,(S,H,NTrain)) 
proba_s = np.ones((S,NTrain))/NTrain

apport =np.loadtxt("D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase_ref/input/hydro/series/area/mod.txt")[6:365:7,:NTrain]*7/H 

pen_low = 3.1
pen_high = 3.1


X = np.linspace(XminB, XmaxB, num = xNsteps)

Ncontrols = 51
U = np.linspace(-PmaxB * H, -PminB * H, Ncontrols)