import matplotlib.pyplot as plt
import numpy as np
from param_watervalues import Reservoir
from functions_iterative import itr_control, NTrain, S

study_path = "D:/Users/gerbauxjul/Documents/6-Etudes Antares/ERAA_scandinavie"
output_path = study_path+"/user/fast_fi00"
# study_path = "D:/Users/gerbauxjul/Documents/6-Etudes Antares/OneNodeBase"
# output_path = study_path+"/user"

pen_low = 10000
pen_high = 0
pen_final = 10000


reservoir = Reservoir(5530000,1,study_path,"fi00","reservoir")
# reservoir = Reservoir(1e7,1,study_path,"area","reservoir", final_level=True)


xNsteps = 50
X = np.linspace(0, reservoir.capacity, num = xNsteps)


vb, G, itr, temps_tot, U = itr_control(reservoir, output_path, pen_low, pen_high, X,2, pen_final) 
# 6m6s cas simple pour 20 niveaux de stock
# 9m0s cas simple pour 50 niveaux de stock 4 itérations
# 86m pour 50 niveaux de stock eraa_scandinavie 10-20 min récupération pb puis 3 itérations (1 simu avec Antares prend 9 minutes > ici c'est plus long)



plt.figure()
plt.title(f"Controles évalués pour les différentes semaines")
plt.ylabel("Controle")
plt.xlabel("Itération")
for s in range(S):
    plt.plot(U[s][0][2:])
plt.show()


plt.figure()
plt.title(f"Nombre d'itérations du simplexe moyen sur toutes les semaines pour différents scénarios")
plt.ylabel("Nombre d'itérations")
plt.xlabel("Itération")
for k in range(NTrain):
    plt.plot(np.arange(len(itr)),np.mean(itr[:,:,k,0],axis=1))
plt.show()


plt.figure()
plt.title(f"Temps de résolution du simplexe total sur toutes les semaines pour différents scénarios")
plt.xlabel("Itération")
plt.ylabel("Temps de résolution")
for k in range(NTrain):
    plt.plot(np.arange(len(itr)),np.sum(itr[:,:,k,1],axis=1))
plt.show()


plt.figure()
plt.title(f"Temps de calcul total/du simplexe par itération")
plt.xlabel("Itération")
plt.ylabel("Temps de calcul")
plt.plot(temps_tot)
plt.plot(np.sum(np.sum(itr[:,:,:,1],axis=1),axis=1))
plt.show()


plt.figure()
plt.title(f"Fonctions de gain à différentes itérations")
plt.xlabel("Contrôle")
plt.ylabel("Coût")
s = 10
for k in range(len(G)):
    g = G[k][s,0,0]*np.array(U[s][0])+G[k][s,0,1]
    plt.plot(np.array(U[s][0])/reservoir.capacity*100,g,label=f"{k}")#-penalties[i:j,s]
plt.legend(title="Itération")
plt.show()