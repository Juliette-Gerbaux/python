import numpy as np
import xpress as xp
xp.controls.outputlog = 0
from time import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from param_watervalues import *

def optModel_thermic():
    model = xp.problem()
    model.controls.xslp_log = -1

    # LOAD
     
    w = [xp.var(f"load_{i}",lb = float('-inf'), ub =  float('inf')) for i in range(H)]

    model.addVariable (w) # Uncertain

    # CONTROLS
    # Thermal 1
    t1 = [xp.var(f"t1_{i}",lb = PminTh[0], ub =  PmaxTh[0]) for i in range(H)]

    model.addVariable (t1)      
    # Thermal 2
    t2 = [xp.var(f"t2_{i}",lb = PminTh[1], ub =  PmaxTh[1]) for i in range(H)]

    model.addVariable (t2)      # Second thermal unit       
    # Thermal 3
    t3 = [xp.var(f"t3_{i}",lb = PminTh[2], ub =  PmaxTh[2]) for i in range(H)]

    model.addVariable (t3)        # Third thermal unit       
    # Not served energy
    t4p = [xp.var(f"ens_{i}") for i in range(H)]

    model.addVariable (t4p)                    # Energy not served 
    t4n = [xp.var(f"spill_{i}") for i in range(H)]

    model.addVariable (t4n) # Energy excess                               

    # Energy balnce
    surplus_thermic = [xp.var(f"surplus_thermic_{h}",lb = float('-inf'), ub =  float('inf')) for h in range(H)]
    model.addVariable(surplus_thermic)
    for h in range(H):
        model.addConstraint(surplus_thermic[h] == t1[h] + t2[h] + t3[h] + t4p[h] - w[h] - t4n[h])

    cost_thermic = xp.var("cost_thermic",lb = float('-inf'), ub =  float('inf'))
    model.addVariable(cost_thermic)
    model.addConstraint(cost_thermic==xp.Sum(np.transpose(np.tile(cTh,(168,1))) * np.array([t1, t2, t3, t4p])))

    return(model)

def optModel_exact(s, V, debut):#x_s,inflow,w
    model = optModel_thermic()

    # STATE
    x_s = xp.var("x_s",lb = XminB, ub = XmaxB)
    model.addVariable (x_s)          # State at the begining of the current week

    x_s_1 = xp.var("x_s_1",lb = XminB, ub = XmaxB)
    model.addVariable (x_s_1) # State at the begining of the following week

    q = [xp.var(f"q_{i}",lb = float('-inf'), ub =  float('inf')) for i in range(H+1)]
    model.addVariable (q)               # Level of stock at each hour

    # INFLOW
    inflow = xp.var("inflow",lb = float('-inf'), ub =  float('inf'))
    model.addVariable (inflow)

    # CONTROLS
    # Battery (>0 : pompage)
    r = [xp.var(f"r_{i}",lb = PminB, ub =  PmaxB) for i in range(H)]

    model.addVariable (r)           # Charge and discharge control for the battery: r>0 means charge                          

    z = xp.var("z",lb = float('-inf'), ub =  float('inf'))

    model.addVariable (z) # Auxiliar variable to introduce the piecewise representation of the future cost
    y = xp.var("y")

    model.addVariable (y)    # Penality for violating guide curves

    # Energy balnce
    var_thermic = model.getVariable()
    surplus_thermic = [x for x in var_thermic if "surplus_thermic" in x.name]
    for h in range(H):
        model.addConstraint(surplus_thermic[h] == r[h])

    # Battery dynamics
    model.addConstraint(q[0] == x_s)                           # State of the current week equal to the initial stock of the reservoir for the week
    model.addConstraint(q[H] == x_s_1)                          # State of the following week equal to the resulting level of stock for begining of the following week

    for h in range(H):
        model.addConstraint(q[h+1] == q[h] + r[h] + inflow)               # Stock evolution equation

    # Future cost: piecewise representation of the future cost
    for i in range(xNsteps-1):
        if (V[i+1, s+1]<float('inf'))&(V[i, s+1]<float('inf')):
            model.addConstraint(z >= (V[i+1, s+1] - V[i, s+1]) / (X[i+1] - X[i]) * (x_s_1 - X[i]) + V[i, s+1])
    
    if debut: 
        if s>=1:
            model.addConstraint(y >=  -pen_low* (x_s - Xmin[s-1]))
            model.addConstraint(y >=  pen_high* (x_s - Xmax[s-1]))
        else:
            model.addConstraint(y >=  -pen_low* (x_s - Xmin[S-1]))
            model.addConstraint(y >=  pen_high* (x_s - Xmax[S-1]))
    else:
        model.addConstraint(y >=  -pen_low* (x_s_1 - Xmin[s]))
        model.addConstraint(y >=  pen_high* (x_s_1 - Xmax[s]))

    cost_thermic = [x for x in var_thermic if "cost_thermic" in x.name]
    model.setObjective(xp.Sum(cost_thermic) + z + y)
    
    return (model)


def SDP_exact(debut):
    V = np.zeros((xNsteps, S+1))
    if debut:
        pen = get_penalties(debut,S)
        for i in range(xNsteps):
            V[i,S] = pen(X[i])

    for s in range(51,-1,-1):
        model = optModel_exact(s, V, debut)
        for i in range(xNsteps): # state boucle x 
            Vx = 0
            model.chgbounds(["x_s","x_s"],['L','U'],[X[i],X[i]])
            for k in range(NTrain): # chronicle boucle w 
                model.chgbounds(["inflow","inflow"],['L','U'],[apport[s,k],apport[s,k]])
                model.chgbounds([f"load_{j}" for j in range(H)]*2,['L']*H+['U']*H,list(LAW_s[s,:,k])+list(LAW_s[s,:,k]))
                model.solve()
                Vx = Vx + model.getObjVal()
            V[i, s] = Vx/NTrain
    return (V)

def OpModel_weeklycost():
    model = optModel_thermic()

    u = xp.var("u")
    model.addVariable (u)          # total turb

    # CONTROLS
    # Battery
    r = [xp.var(f"r_{i}",lb = PminB, ub =  PmaxB) for i in range(H)]

    model.addVariable (r)           # Charge and discharge control for the battery: r>0 means charge
    
    # Energy balnce
    var_thermic = model.getVariable()
    surplus_thermic = [x for x in var_thermic if "surplus_thermic" in x.name]
    for h in range(H):
        model.addConstraint(surplus_thermic[h] == r[h])

    # Battery dynamics
    model.addConstraint(xp.Sum(r) == -u)

    cost_thermic = [x for x in var_thermic if "cost_thermic" in x.name]
    model.setObjective(xp.Sum(cost_thermic))
    
    return (model)


def WeeklyCost(u):
    model = OpModel_weeklycost()
    L = np.zeros((S,NTrain))
    model.chgbounds(["u","u"],['L','U'],[u,u])
    for k in range(NTrain):
        for s in range(S):
            model.chgbounds([f"load_{j}" for j in range(H)]*2,['L']*H+['U']*H,list(LAW_s[s,:,k])+list(LAW_s[s,:,k]))
            model.solve()
            L[s,k] = model.getObjVal()
    return(L)

def SDP_precalculated_rewards(debut, Gu):    
    V = np.zeros((xNsteps, S+1))

    if debut:
        pen = get_penalties(debut,S)
        for i in range(xNsteps):
            V[i,S] = pen(X[i])

    for s in range(S-1,-1,-1):
        V_fut = interp1d(X, V[:, s+1])
        pen = get_penalties(debut,s)

        for k in range(NTrain):
            Gs = interp1d(U, Gu[:, s, k])
            for i in range(xNsteps):
                Vu = float('-inf')

                if debut :
                    penalty = pen(X[i])
                
                for i_fut in range(xNsteps):
                    u = -X[i_fut] + X[i] + apport[s,k]*H
                    if np.min(U) <= u <= np.min(U):
                        G = Gs(u)
                        if not(debut):
                            penalty = pen(X[i_fut])
                        if (G + V[i_fut, s+1]+penalty) > Vu:
                            Vu = G + V[i_fut, s+1]+penalty

                for u in range(Ncontrols):
                    state_fut = X[i] - U[u] + apport[s,k]*H 
                    if XminB <= state_fut <= XmaxB:
                        if not(debut):
                            penalty = pen(state_fut)
                        if (Gu[u, s, k] + V_fut(state_fut)+penalty) > Vu:
                            Vu = (Gu[u, s, k] + V_fut(state_fut)+penalty)

                Umin = X[i]+ apport[s,k]*H-Xmin[s]
                if np.min(U) <= Umin <= np.max(U):
                    state_fut = X[i] - Umin + apport[s,k]*H 
                    if not(debut):
                        penalty = pen(state_fut)
                    if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umin) + V_fut(state_fut)+penalty

                Umax = X[i]+ apport[s,k]*H-Xmax[s]
                if np.min(U) <= Umax <= np.max(U):
                    state_fut = X[i] - Umax + apport[s,k]*H 
                    if not(debut):
                        penalty = pen(state_fut)
                    if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umax) + V_fut(state_fut)+penalty
            
                V[i, s] = Vu/NTrain + V[i,s]
    return V

def get_penalties(for_beginning_of_week,s):
    if for_beginning_of_week:
        if s>=1:
            pen = interp1d([XminB,Xmin[s-1],Xmax[s-1],XmaxB],[-pen_low*(Xmin[s-1]-XminB),0,0,-pen_high*(XmaxB-Xmax[s-1])])
        else:
            pen = interp1d([XminB,Xmin[S-1],Xmax[S-1],XmaxB],[-pen_low*(Xmin[S-1]-XminB),0,0,-pen_high*(XmaxB-Xmax[S-1])])
    else:
        pen = interp1d([XminB,Xmin[s],Xmax[s],XmaxB],[-pen_low*(Xmin[s]-XminB),0,0,-pen_high*(XmaxB-Xmax[s])])
    return(pen)
