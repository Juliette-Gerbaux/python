import xpress as xp
import numpy as np
from scipy.interpolate import interp1d
from param_watervalues import Reservoir
from time import time
from random import randint, seed

xp.controls.outputlog = -1

S = 52
NTrain = 10
H = 168

def retrieve_problem(year,week,output_path,itr=1):
    model = xp.problem()
    model.read(output_path+f"/problem-{year}-{week}--optim-nb-{itr}.mps")
    return(model)

def create_weekly_problem_itr(k,s,output_path, reservoir, pen_low, pen_high,pen_final):
    model = retrieve_problem(k+1,s+1,output_path)
    model.controls.xslp_log = -1

    cst = model.getConstraint()
    binding_id = [i for i in range(len(cst)) if "WeeklyWaterAmount" in cst[i].name]

    model_copy = model.copy()

    x_s = xp.var("x_s",lb = 0, ub = reservoir.capacity)
    model_copy.addVariable (x_s)          # State at the begining of the current week

    x_s_1 = xp.var("x_s_1",lb = 0, ub = reservoir.capacity)
    model_copy.addVariable (x_s_1) # State at the begining of the following week

    U = xp.var("u",lb = -reservoir.P_pump[7*s]*reservoir.efficiency*H, ub = reservoir.P_turb[7*s]*H)
    model_copy.addVariable (U) # State at the begining of the following week

    model_copy.addConstraint(x_s_1 == x_s - U + reservoir.inflow[s,k]*H)

    y = xp.var("y")

    model_copy.addVariable (y)    # Penality for violating guide curves

    if s!=S-1:
        model_copy.addConstraint(y >=  -pen_low* (x_s_1 - reservoir.Xmin[s]))
        model_copy.addConstraint(y >=  pen_high* (x_s_1 - reservoir.Xmax[s]))
    else :
        model_copy.addConstraint(y >=  -pen_final* (x_s_1 - reservoir.Xmin[s]))
        model_copy.addConstraint(y >=  pen_final* (x_s_1 - reservoir.Xmax[s]))

    model_copy.chgmcoef(binding_id,[U],[-1])
    model_copy.chgrhs(binding_id,[0])

    z = xp.var("z",lb = float('-inf'), ub =  float('inf'))

    model_copy.addVariable (z) # Auxiliar variable to introduce the piecewise representation of the future cost

    model_copy.chgobj([y,z], [1,1])

    return([model,binding_id, model_copy,x_s,x_s_1, z, y])
    


def modify_weekly_problem_itr(model,binding_id,u):

    model.chgrhs(binding_id,[u])

    model.lpoptimize()

    if model.attributes.lpstatus==1:
        beta = model.getObjVal()
        lamb = model.getDual(binding_id)[0]
        return(beta,lamb,model.attributes.SIMPLEXITER, model.attributes.TIME)
    else :
        raise(ValueError)


def calculate_VU(reward, reservoir,X,U, pen_low, pen_high, pen_final):
    V = np.zeros((len(X), S+1, NTrain))

    for s in range(S-1,-1,-1):#
        
        if s==S-1:
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
        else :
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])

        for k in range(NTrain):
            V_fut = interp1d(X, V[:, s+1,k])
            Gs = lambda x: min([reward[i][s,k,0]*x+reward[i][s,k,1] for i in range(len(reward))])
            for i in range(len(X)):
                Vu = float('-inf')

                for i_fut in range(len(X)):
                    u = -X[i_fut] + X[i] + reservoir.inflow[s,k]*H
                    if -reservoir.P_pump[7*s]*H <= u <= reservoir.P_turb[7*s]*H:
                        G = Gs(u)
                        penalty = pen(X[i_fut])
                        if (G + V[i_fut, s+1,k]+penalty) > Vu:
                            Vu = G + V[i_fut, s+1,k]+penalty

                for u in range(len(U[s][k])):
                    state_fut = min(reservoir.capacity,X[i] - U[s][k][u] + reservoir.inflow[s,k]*H) 
                    if 0 <= state_fut :
                        penalty = pen(state_fut)
                        G = Gs(U[s][k][u])
                        if (G + V_fut(state_fut)+penalty) > Vu:
                            Vu = (G + V_fut(state_fut)+penalty)

                Umin = X[i]+ reservoir.inflow[s,k]*H-reservoir.Xmin[s]
                if -reservoir.P_pump[7*s]*H <= Umin <= reservoir.P_turb[7*s]*H:
                    state_fut = X[i] - Umin + reservoir.inflow[s,k]*H
                    penalty = pen(state_fut)
                    if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umin) + V_fut(state_fut)+penalty

                Umax = X[i]+ reservoir.inflow[s,k]*H-reservoir.Xmax[s]
                if -reservoir.P_pump[7*s]*H <= Umax <= reservoir.P_turb[7*s]*H:
                    state_fut = X[i] - Umax + reservoir.inflow[s,k]*H 
                    penalty = pen(state_fut)
                    if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
                        Vu = Gs(Umax) + V_fut(state_fut)+penalty
            
                V[i, s, k] = Vu + V[i,s,k]
        V[:,s,:] = np.repeat(np.mean(V[:,s,:],axis=1,keepdims=True),NTrain,axis=1)
    return np.mean(V,axis=2)


def compute_x(reservoir,X,U,V,reward,pen_low,pen_high, pen_final, itr):
    initial_x = [reservoir.initial_level]
    controls = np.zeros((S,NTrain))
    cout = 0
    seed(19*itr)
    
    for s in range(S):
        cout_s =0
        V_fut = interp1d(X, V[:, s+1])
        k = randint(0,NTrain-1)
        
        if s==S-1:
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
        else :
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])
        Gs = lambda x: min([reward[i][s,k,0]*x+reward[i][s,k,1] for i in range(len(reward))])

        Vu = float('-inf')

        for i_fut in range(len(X)):
            u = -X[i_fut] + initial_x[-1] + reservoir.inflow[s,k]*H
            if -reservoir.P_pump[7*s]*H <= u <= reservoir.P_turb[7*s]*H:
                G = Gs(u)
                penalty = pen(X[i_fut])
                if (G + V[i_fut, s+1]+penalty) > Vu:
                    Vu = G + V[i_fut, s+1]+penalty
                    xf = X[i_fut]
                    cout_s = G

        for u in range(len(U[s][k])):
            state_fut = min(reservoir.capacity,initial_x[-1] - U[s][k][u] + reservoir.inflow[s,k]*H) 
            if 0 <= state_fut :
                penalty = pen(state_fut)
                G = Gs(U[s][k][u])
                if (G + V_fut(state_fut)+penalty) > Vu:
                    Vu = (G + V_fut(state_fut)+penalty)
                    xf = state_fut
                    cout_s =G

        Umin = initial_x[-1]+ reservoir.inflow[s,k]*H-reservoir.Xmin[s]
        if -reservoir.P_pump[7*s]*H <= Umin <= reservoir.P_turb[7*s]*H:
            state_fut = initial_x[-1] - Umin + reservoir.inflow[s,k]*H
            penalty = pen(state_fut)
            if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
                Vu = Gs(Umin) + V_fut(state_fut)+penalty
                xf = state_fut
                cout_s = Gs(Umin)

        Umax = initial_x[-1]+ reservoir.inflow[s,k]*H-reservoir.Xmax[s]
        if -reservoir.P_pump[7*s]*H <= Umax <= reservoir.P_turb[7*s]*H:
            state_fut = initial_x[-1] - Umax + reservoir.inflow[s,k]*H 
            penalty = pen(state_fut)
            if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
                Vu = Gs(Umax) + V_fut(state_fut)+penalty
                xf = state_fut
                cout_s = Gs(Umax)

        initial_x.append(xf)
        controls[s] = -(initial_x[s+1]-initial_x[s]-reservoir.inflow[s,k]*H)
        cout += cout_s
    return(initial_x, controls, cout)


def compute_upper_bound(reservoir,list_models,X,V):
    cout = 0
    controls = np.zeros((S,NTrain))
    for k in range(NTrain):
        
        level_i = reservoir.initial_level
        for s in range(S):
            print(f"{k} {s}",end="\r")
            m = list_models[s][k]
            # Future cost: piecewise representation of the future cost
            nb_cons = m[2].attributes.rows
          

            for i in range(len(X)-1):
                if (V[i+1, s+1]<float('inf'))&(V[i, s+1]<float('inf')):
                    m[2].addConstraint(m[5] >= (V[i+1, s+1] - V[i, s+1]) / (X[i+1] - X[i]) * (m[4] - X[i]) + V[i, s+1])

            cst_initial_level = m[3] == level_i
            m[2].addConstraint(cst_initial_level)

            m[2].lpoptimize()

            if m[2].attributes.lpstatus==1:
                beta = m[2].getObjVal()
                xf = m[2].getSolution(m[4])
                z = m[2].getSolution(m[5])
                y = m[2].getSolution(m[6])
                m[2].delConstraint(range(nb_cons,m[2].attributes.rows))
                cout += beta
                controls[s,k]=-(xf-level_i-reservoir.inflow[s,k]*H)
                level_i = xf
                if s!=S-1:
                    cout += - z - y
                
            else :
                raise(ValueError)
    return(cout/NTrain, controls)
            


def itr_control(reservoir:Reservoir, output_path, pen_low, pen_high,X, N, pen_final):

    # seed(19)
    
    list_models = [[] for i in range(S)]
    for s in range(S):
        for k in range(NTrain):
            m = create_weekly_problem_itr(k=k,s=s,output_path=output_path,reservoir=reservoir,pen_low=pen_low,pen_high=pen_high,pen_final=pen_final)
            list_models[s].append(m)
    
    # k = randint(0,NTrain-1)
    # initial_x = [reservoir.initial_level]
    # controls = []
    # for s in range(S):
    #     xf = initial_x[-1] + max(reservoir.inflow[s,k]*H-reservoir.P_turb[7*s]*H,0)
    #     initial_x.append(xf)
    #     controls.append(-(initial_x[s+1]-initial_x[s]-reservoir.inflow[s,k]*H))
    V = np.zeros((len(X), S+1))

    G = [np.zeros((S+1,NTrain,2))]
    U = [[[reservoir.P_turb[7*s]*H,-reservoir.P_pump[7*s]*H] for k in range(NTrain)] for s in range(S)]
    itr_tot = []
    tot_t = []

    best_cost = np.zeros(NTrain)
    best_lb = np.zeros(NTrain)

    # upper_bound, controls = compute_upper_bound(reservoir,list_models,X,V,pen_low,pen_high) 
    i = 0
    gap = 1e3
    while gap>=1e-2 or gap<=-1e-3:
        debut = time()
        Gj = np.zeros((S+1,NTrain,2))
        current_cost = np.zeros(NTrain)
        current_itr = np.zeros((S,NTrain,2))
        # k = randint(0,NTrain-1)
        initial_x, controls, lb = compute_x(reservoir=reservoir,X=X,U=U,V=V,reward=G,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final, itr=i)
        for k in range(NTrain):
            # initial_x, controls, lb = compute_x(reservoir=reservoir,k=k,X=X,U=U,V=V,reward=G,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final)
            # best_lb[k] = max(best_lb[k],-lb)
            for s in range(S):
                print(f"{k} {s}",end="\r")
                m = list_models[s][k]
                beta,lamb,itr, t = modify_weekly_problem_itr(model=m[0],binding_id=m[1],u=controls[s][k])
                Gj[s,k,0] = -lamb
                Gj[s,k,1] = -beta + lamb*controls[s][k]
                U[s][k].append(controls[s][k])
                current_cost[k] += beta
                current_itr[s,k] = (itr,t)
            # V = calculate_VU(reward=G+[Gj],reservoir=reservoir,X=X,U=U,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final)
        G.append(Gj)
        itr_tot.append(current_itr)

        # V = calculate_VU_continu(reward=G,reservoir=reservoir,X=X,U=U,pen_low=pen_low,pen_high=pen_high)
        
        # V0_continu = max([V[0][i,0]*reservoir.initial_level+V[0][i,1] for i in range(len(V[0]))])

        V = calculate_VU(reward=G,reservoir=reservoir,X=X,U=U,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final)
        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir.initial_level)

        if np.mean(best_cost)==0:
            best_cost = current_cost
        else :
            best_cost = np.minimum(best_cost,current_cost)
        upper_bound, controls = compute_upper_bound(reservoir,list_models,X,-V) 
        gap = upper_bound+V0
        print(gap, upper_bound,-V0, np.mean(best_lb),np.mean(best_cost)) #,V0_continu)
        gap = gap/-V0
        fin = time()
        i+=1
        tot_t.append(fin-debut)
    return (V, G, np.array(itr_tot),tot_t, U)