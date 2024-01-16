import numpy as np
import xpress as xp
xp.controls.outputlog = 0
from scipy.interpolate import interp1d
from param_watervalues import *
import matplotlib.pyplot as plt

def optModel_thermic(study: Study):
    model = xp.problem()
    model.controls.xslp_log = -1

    total_cost = [0]*H

    imports = {}
    exports = {}
    for area in study.list_areas:
        imports[area.name] = [0]*H
        exports[area.name] = [0]*H
    for a1 in study.list_areas:
        for a2 in study.list_areas:
            if (a1.name in study.links) and (a2.name in study.links[a1.name]):
                exchange = [xp.var(f"exchange_{a1.name}_{a2.name}_{h}",lb = 0, ub =  study.links[a1.name][a2.name][0]) for h in range(H)] 
                model.addVariable(exchange)
                for h in range(H):
                    imports[a2.name][h] += exchange[h]
                    exports[a1.name][h] += exchange[h]
                    total_cost[h] += exchange[h]*study.links[a1.name][a2.name][1]



    # CONTROLS
    # Thermal
    for area in study.list_areas:

        total_prod = [0]*H

        # LOAD
        w = [xp.var(f"load_{area.name}_{i}",lb = float('-inf'), ub =  float('inf')) for i in range(H)]
        model.addVariable (w) # Uncertain

        for thermal_unit in area.list_thermal_units :
            t = [xp.var(f"t_{area.name}_{thermal_unit.name}_{i}",lb = 0, ub =  thermal_unit.P_max) for i in range(H)]
            model.addVariable (t) 
            state_on = [xp.var(f"on_{area.name}_{thermal_unit.name}_{i}",vartype=xp.binary) for i in range(H)]
            model.addVariable (state_on) 
            start_up = [xp.var(f"up_{area.name}_{thermal_unit.name}_{i}",vartype=xp.binary) for i in range(H)]
            model.addVariable (start_up)
            for h in range(H):
                total_prod[h] += t[h]
                total_cost[h] += t[h]*thermal_unit.marginal_cost 
                total_cost[h] += state_on[h]*thermal_unit.fixed_cost
                total_cost[h] += start_up[h]*thermal_unit.start_up_cost
                model.addConstraint(state_on[h]*thermal_unit.P_max>=t[h])
                model.addConstraint(state_on[h]*thermal_unit.P_min<=t[h])
                if h>=1:
                    model.addConstraint(start_up[h]>=state_on[h]-state_on[h-1])
                else :
                    model.addConstraint(start_up[h]>=state_on[h])

       
        # Not served energy
        ens = [xp.var(f"ens_{area.name}_{i}") for i in range(H)]
        model.addVariable (ens)                    # Energy not served 
        for h in range(H):
            total_cost[h] += ens[h]*study.cost_ens

        spill = [xp.var(f"spill_{area.name}_{i}") for i in range(H)]
        model.addVariable (spill) # Energy excess

        # Interco
                            

        # Energy balnce
        surplus_thermic = [xp.var(f"surplus_thermic_{area.name}_{h}",lb = float('-inf'), ub =  float('inf')) for h in range(H)]
        model.addVariable(surplus_thermic)
        for h in range(H):
            model.addConstraint(surplus_thermic[h] == total_prod[h] + ens[h] + imports[area.name][h] - w[h] - spill[h] - exports[area.name][h])

    cost_thermic = xp.var("cost_thermic",lb = float('-inf'), ub =  float('inf'))
    model.addVariable(cost_thermic)
    model.addConstraint(cost_thermic==xp.Sum(total_cost))

    return(model)

def optModel_exact(study: Study, area: Area, reservoir: Reservoir, s, V, debut, X):#x_s,inflow,w
    model = optModel_thermic(study)

    # STATE
    x_s = xp.var("x_s",lb = 0, ub = reservoir.capacity)
    model.addVariable (x_s)          # State at the begining of the current week

    x_s_1 = xp.var("x_s_1",lb = 0, ub = reservoir.capacity)
    model.addVariable (x_s_1) # State at the begining of the following week

    q = [xp.var(f"q_{i}",lb = float('-inf'), ub =  float('inf')) for i in range(H+1)]
    model.addVariable (q)               # Level of stock at each hour

    # INFLOW
    inflow = xp.var("inflow",lb = float('-inf'), ub =  float('inf'))
    model.addVariable (inflow)

    # CONTROLS
    # Battery (>0 : pompage)
    r = [xp.var(f"r_{i}",lb = reservoir.P_turb, ub =  reservoir.P_pump) for i in range(H)]

    model.addVariable (r)           # Charge and discharge control for the battery: r>0 means charge                          

    z = xp.var("z",lb = float('-inf'), ub =  float('inf'))

    model.addVariable (z) # Auxiliar variable to introduce the piecewise representation of the future cost
    y = xp.var("y")

    model.addVariable (y)    # Penality for violating guide curves

    # Energy balnce
    var_thermic = model.getVariable()
    surplus_thermic = [x for x in var_thermic if f"surplus_thermic_{area.name}_" in x.name]
    for h in range(H):
        model.addConstraint(surplus_thermic[h] == r[h])
    
    for a in study.list_areas:
        if a.name != area.name:
            surplus_thermic = [x for x in var_thermic if f"surplus_thermic_{a.name}_" in x.name]
            for h in range(H):
                model.addConstraint(surplus_thermic[h] == 0)


    # Battery dynamics
    model.addConstraint(q[0] == x_s)                           # State of the current week equal to the initial stock of the reservoir for the week
    model.addConstraint(q[H] == x_s_1)                          # State of the following week equal to the resulting level of stock for begining of the following week

    for h in range(H):
        model.addConstraint(q[h+1] == q[h] + r[h] + inflow)               # Stock evolution equation

    assert(len(X)>=2)
    # Future cost: piecewise representation of the future cost
    for i in range(len(X)-1):
        if (V[i+1, s+1]<float('inf'))&(V[i, s+1]<float('inf')):
            model.addConstraint(z >= (V[i+1, s+1] - V[i, s+1]) / (X[i+1] - X[i]) * (x_s_1 - X[i]) + V[i, s+1])
    
    if debut: 
        if s>=1:
            model.addConstraint(y >=  -study.pen_low* (x_s - reservoir.Xmin[s-1]))
            model.addConstraint(y >=  study.pen_high* (x_s - reservoir.Xmax[s-1]))
        else:
            model.addConstraint(y >=  -study.pen_low* (x_s - reservoir.Xmin[S-1]))
            model.addConstraint(y >=  study.pen_high* (x_s - reservoir.Xmax[S-1]))
    else:
        model.addConstraint(y >=  -study.pen_low* (x_s_1 - reservoir.Xmin[s]))
        model.addConstraint(y >=  study.pen_high* (x_s_1 - reservoir.Xmax[s]))

    cost_thermic = [x for x in var_thermic if "cost_thermic" in x.name]
    model.setObjective(xp.Sum(cost_thermic) + z + y)
    
    return (model)

def SDP_exact(study: Study, area:Area, reservoir:Reservoir, debut:bool, X):
    V = np.zeros((len(X), S+1))
    if debut:
        pen = get_penalties(debut,S)
        for i in range(len(X)):
            V[i,S] = pen(X[i])
    for s in range(51,-1,-1):
        model = optModel_exact(study, area, reservoir, s, V, debut, X)
        for i in range(len(X)): # state boucle x 
            Vx = 0
            model.chgbounds(["x_s","x_s"],['L','U'],[X[i],X[i]])
            for k in range(study.nb_mc): # chronicle boucle w 
                model.chgbounds(["inflow","inflow"],['L','U'],[reservoir.inflow[s,k],reservoir.inflow[s,k]])
                for a in study.list_areas:
                    model.chgbounds([f"load_{a.name}_{j}" for j in range(H)]*2,['L']*H+['U']*H,list(a.load[s,:,k])+list(a.load[s,:,k]))
                # return(model, s, X[i])
                model.solve()
                Vx = Vx + model.getObjVal()
            V[i, s] = Vx/study.nb_mc
    return (V)

def OpModel_weeklycost(study:Study, area:Area, reservoir:Reservoir):
    model = optModel_thermic(study)

    u = xp.var("u",lb = float('-inf'), ub =  float('inf'))
    model.addVariable (u)          # total turb

    # CONTROLS
    # Battery
    r = [xp.var(f"r_{i}",lb = reservoir.P_turb, ub =  reservoir.P_pump) for i in range(H)]

    model.addVariable (r)           # Charge and discharge control for the battery: r>0 means charge
    
    # Energy balnce
    var_thermic = model.getVariable()
    surplus_thermic = [x for x in var_thermic if f"surplus_thermic_{area.name}_" in x.name]
    for h in range(H):
        model.addConstraint(surplus_thermic[h] == r[h])
    for a in study.list_areas:
        if a.name != area.name:
            surplus_thermic = [x for x in var_thermic if f"surplus_thermic_{a.name}_" in x.name]
            for h in range(H):
                model.addConstraint(surplus_thermic[h] == 0)

    # Battery dynamics
    model.addConstraint(xp.Sum(r) == -u)

    cost_thermic = [x for x in var_thermic if "cost_thermic" in x.name]
    model.setObjective(xp.Sum(cost_thermic))
    
    return (model)


def WeeklyCost(study:Study, area:Area, reservoir:Reservoir, u):
    model = OpModel_weeklycost(study, area, reservoir)
    L = np.zeros((S,study.nb_mc))
    model.chgbounds(["u","u"],['L','U'],[u,u])
    for k in range(study.nb_mc):
        for s in range(S):
            for a in study.list_areas:
                model.chgbounds([f"load_{a.name}_{j}" for j in range(H)]*2,['L']*H+['U']*H,list(a.load[s,:,k])+list(a.load[s,:,k]))
            model.solve()
            L[s,k] = model.getObjVal()
    return(L)

def SDP_precalculated_rewards(study:Study, reservoir:Reservoir, debut:bool, Gu, X, U):    
    V = np.zeros((len(X), S+1))

    if debut:
        pen = get_penalties(study, reservoir,debut,S)
        for i in range(len(X)):
            V[i,S] = pen(X[i])

    for s in range(S-1,-1,-1):
        V_fut = interp1d(X, V[:, s+1])
        pen = get_penalties(study, reservoir,debut,s)

        for k in range(study.nb_mc):
            Gs = interp1d(U, Gu[:, s, k])
            for i in range(len(X)):
                Vu = float('-inf')

                if debut :
                    penalty = pen(X[i])
                
                for i_fut in range(len(X)):
                    u = -X[i_fut] + X[i] + reservoir.inflow[s,k]*H
                    if np.min(U) <= u <= np.max(U):
                        G = Gs(u)
                        if not(debut):
                            penalty = pen(X[i_fut])
                        if (G + V[i_fut, s+1]+penalty) > Vu:
                            Vu = G + V[i_fut, s+1]+penalty

                for u in range(len(U)):
                    state_fut = X[i] - U[u] + reservoir.inflow[s,k]*H 
                    if 0 <= state_fut <= reservoir.capacity:
                        if not(debut):
                            penalty = pen(state_fut)
                        if (Gu[u, s, k] + V_fut(state_fut)+penalty) > Vu:
                            Vu = (Gu[u, s, k] + V_fut(state_fut)+penalty)

                Umin = X[i]+ reservoir.inflow[s,k]*H-reservoir.Xmin[s]
                if np.min(U) <= Umin <= np.max(U):
                    state_fut = X[i] - Umin + reservoir.inflow[s,k]*H 
                    if not(debut):
                        penalty = pen(state_fut)
                    if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umin) + V_fut(state_fut)+penalty

                Umax = X[i]+ reservoir.inflow[s,k]*H-reservoir.Xmax[s]
                if np.min(U) <= Umax <= np.max(U):
                    state_fut = X[i] - Umax + reservoir.inflow[s,k]*H 
                    if not(debut):
                        penalty = pen(state_fut)
                    if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umax) + V_fut(state_fut)+penalty
            
                V[i, s] = Vu/study.nb_mc + V[i,s]
    return V

def get_penalties(study:Study, reservoir:Reservoir, for_beginning_of_week:bool,s:int):
    if for_beginning_of_week:
        if s>=1:
            pen = interp1d([0,reservoir.Xmin[s-1],reservoir.Xmax[s-1],reservoir.capacity],[-study.pen_low*(reservoir.Xmin[s-1]),0,0,-study.pen_high*(reservoir.capacity-reservoir.Xmax[s-1])])
        else:
            pen = interp1d([0,reservoir.Xmin[S-1],reservoir.Xmax[S-1],reservoir.capacity],[-study.pen_low*(reservoir.Xmin[S-1]),0,0,-study.pen_high*(reservoir.capacity-reservoir.Xmax[S-1])])
    else:
        pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-study.pen_low*(reservoir.Xmin[s]),0,0,-study.pen_high*(reservoir.capacity-reservoir.Xmax[s])])
    return(pen)

def solve_and_plot(m, study:Study, s, x):
    m.solve()
    vars = m.getVariable()
    sol = m.getSolution()

    nb_plot = 3
    fig, ax = plt.subplots(nb_plot*len(study.list_areas), sharex='all',figsize=(7,2.5*nb_plot*len(study.list_areas)))
    j = 0
    for a in study.list_areas:
        conso = [sol[i] for i in range(len(vars)) if f"load_{a.name}_" in vars[i].name]
        ax[j].plot(conso, label="Conso")
        prod_plus = {}
        for th in a.list_thermal_units:
            prod_plus[f"Prod_th_{th.name}"] = np.array([sol[i] for i in range(len(vars)) if f"t_{a.name}_{th.name}_" in vars[i].name])
        for r in a.list_reservoirs:
            prod_plus[f"Turb_r_{r.name}"] = np.array([max(-sol[i],0) for i in range(len(vars)) if f"r_" in vars[i].name])
        prod_plus[f"ENS_{a.name}"] = np.array([sol[i] for i in range(len(vars)) if f"ens_{a.name}_" in vars[i].name])
        for a2 in study.list_areas:
            if (a2.name in study.links) and (a.name in study.links[a2.name]):
                prod_plus[f"exchange_{a2.name}_{a.name}"] = np.array([sol[i] for i in range(len(vars)) if f"exchange_{a2.name}_{a.name}_" in vars[i].name])
        ax[j].stackplot(np.arange(H), prod_plus.values(),
                labels=prod_plus.keys())
        
        ax[j].legend()
        ax[j].grid(True)
        ax[j].set_title(f"{a.name}")
        # plt.show()

        # fig, ax = plt.subplots()
        ax[j+1].plot(sum([prod_plus[a] for a in prod_plus.keys()])-conso, label="Conso-prod")
        prod_minus = {}
        for r in a.list_reservoirs:
            prod_minus[f"Pump_r_{r.name}"] = np.array([max(sol[i],0) for i in range(len(vars)) if f"r_" in vars[i].name])
        prod_minus[f"spill_{a.name}"] = np.array([-sol[i] for i in range(len(vars)) if f"spill_{a.name}_" in vars[i].name])
        for a2 in study.list_areas:
            if (a.name in study.links) and (a2.name in study.links[a.name]):
                prod_minus[f"exchange_{a.name}_{a2.name}"] = np.array([sol[i] for i in range(len(vars)) if f"exchange_{a.name}_{a2.name}_" in vars[i].name])
        # assert(np.all(prod_plus-conso-prod_minus)==0) 
        ax[j+1].stackplot(np.arange(H), prod_minus.values(),
                labels=prod_minus.keys())
        ax[j+1].legend()
        ax[j+1].grid(True)
        ax[j+1].set_title(f"{a.name}")

        for th in a.list_thermal_units:
            ax[j+2].plot([sol[i]*th.P_max for i in range(len(vars)) if f"on_{a.name}_{th.name}_" in vars[i].name],label=f"On_th_{th.name}")
            ax[j+2].plot([sol[i] for i in range(len(vars)) if f"t_{a.name}_{th.name}_" in vars[i].name],label=f"Prod_th_{th.name}")
            ax[j+2].plot([sol[i]*th.P_max for i in range(len(vars)) if f"up_{a.name}_{th.name}_" in vars[i].name],label=f"Up_th_{th.name}")
        ax[j+2].legend()
        ax[j+2].grid(True)
        ax[j+2].set_title(f"{a.name}")

        j += nb_plot

    plt.show()

    for a in study.list_areas:
        for r in a.list_reservoirs:
            fig, ax = plt.subplots()
            u = np.array([sol[i] for i in range(len(vars)) if f"r_" in vars[i].name])
            ax.plot(np.cumsum(u)+x, label=f"r_{r.name}")
            ax.plot(r.Xmin[s].repeat(H))
            ax.plot(r.Xmax[s].repeat(H))
            ax.legend()
            ax.set_title(f"{a.name}")
            plt.show()