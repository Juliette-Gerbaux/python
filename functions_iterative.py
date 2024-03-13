import xpress as xp
import numpy as np
from scipy.interpolate import interp1d
from time import time
from random import randint, seed
from data_iterative import S,H,NTrain

xp.controls.outputlog = 0
xp.controls.threads = 1
xp.controls.scaling = 0
xp.controls.presolve = 0
xp.controls.feastol = 1.e-7
xp.controls.optimalitytol = 1.e-7
xp.setOutputEnabled(False)

class Reservoir:

    def __init__(self, capacity:float, efficiency:float, dir_study:str, name_area:str, name:str, final_level:bool=True):
        
        self.capacity = capacity

        courbes_guides = np.loadtxt(dir_study+"/input/hydro/common/capacity/reservoir_"+name_area+".txt")[:,[0,2]]*self.capacity
        if (courbes_guides[0,0]==courbes_guides[0,1]):
            self.initial_level = courbes_guides[0,0]
        else :
            print("Problème avec le niveau initial")
        Xmin = courbes_guides[6:365:7,0]
        Xmax = courbes_guides[6:365:7,1]
        self.Xmin = np.concatenate((Xmin,Xmin[[0]]))
        self.Xmax = np.concatenate((Xmax,Xmax[[0]]))
        if final_level:
            self.Xmin[51] = self.initial_level
            self.Xmax[51] = self.initial_level
        

        self.inflow = np.loadtxt(dir_study+"/input/hydro/series/"+name_area+"/mod.txt")[6:365:7]*7/H 
        assert("_" not in name)
        self.name = name

        P_turb = np.loadtxt(dir_study+"/input/hydro/common/capacity/maxpower_"+name_area+".txt")[:,0]
        P_pump = np.loadtxt(dir_study+"/input/hydro/common/capacity/maxpower_"+name_area+".txt")[:,2]
        self.P_turb = P_turb
        self.P_pump = P_pump
        self.efficiency = efficiency

def retrieve_problem(year,week,output_path,itr=1):
    model = xp.problem()
    model.controls.xslp_log = -1
    model.controls.lplogstyle = 0
    model.read(output_path+f"/problem-{year}-{week}--optim-nb-{itr}.mps")
    return(model)

def create_weekly_problem_itr(k,s,output_path, reservoir, pen_low, pen_high,pen_final):
    model = retrieve_problem(k+1,s+1,output_path)
    

    cst = model.getConstraint()
    binding_id = [i for i in range(len(cst)) if "WeeklyWaterAmount" in cst[i].name]

    x_s = xp.var("x_s",lb = 0, ub = reservoir.capacity)
    model.addVariable (x_s)          # State at the begining of the current week

    x_s_1 = xp.var("x_s_1",lb = 0, ub = reservoir.capacity)
    model.addVariable (x_s_1) # State at the begining of the following week

    U = xp.var("u",lb = -reservoir.P_pump[7*s]*reservoir.efficiency*H, ub = reservoir.P_turb[7*s]*H)
    model.addVariable (U) # State at the begining of the following week

    model.addConstraint(x_s_1 <= x_s - U + reservoir.inflow[s,k]*H)

    y = xp.var("y")

    model.addVariable (y)    # Penality for violating guide curves

    if s!=S-1:
        model.addConstraint(y >=  -pen_low* (x_s_1 - reservoir.Xmin[s]))
        model.addConstraint(y >=  pen_high* (x_s_1 - reservoir.Xmax[s]))
    else :
        model.addConstraint(y >=  -pen_final* (x_s_1 - reservoir.Xmin[s]))
        model.addConstraint(y >=  pen_final* (x_s_1 - reservoir.Xmax[s]))

    z = xp.var("z",lb = float('-inf'), ub =  float('inf'))

    model.addVariable (z) # Auxiliar variable to introduce the piecewise representation of the future cost

    return([model,binding_id,U,x_s,x_s_1, z, y]) #, model_copy,x_s,x_s_1, z, y])

def modify_weekly_problem_itr(m,control,rstatus,cstatus,i,basis,control_basis):

    if (len(rstatus)!=0)&(i==0):
        m[0].loadbasis(rstatus, cstatus)

    if i>=1:
        u = np.argmin(np.abs(np.array(control_basis)-control))
        m[0].loadbasis(basis[u][0],basis[u][1])

    rbas = []
    cbas = []

    m[0].chgrhs(m[1],[control])
    debut_1 = time()
    m[0].lpoptimize()
    fin_1 = time()
    

    if m[0].attributes.lpstatus==1:
        beta = m[0].getObjVal()
        lamb = m[0].getDual(m[1])[0]
        itr = m[0].attributes.SIMPLEXITER
        t = m[0].attributes.TIME

        
        m[0].getbasis(rbas,cbas)
        
        if i==0:
            m[0].getbasis(rstatus, cstatus)
        return(beta,lamb,itr, t, rbas, cbas, rstatus,cstatus, debut_1, fin_1)
    else :
        
        raise(ValueError)
    
def calculate_VU(reward, reservoir,X,U, pen_low, pen_high, pen_final):
    V = np.zeros((len(X), S+1, NTrain))

    for s in range(S-1,-1,-1):
        
        if s==S-1:
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
        else :
            pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])

        for k in range(NTrain):
            V_fut = interp1d(X, V[:, s+1,k])
            Gs = lambda x: min([reward[s][k][i][0]*x+reward[s][k][i][1] for i in range(len(reward[s][k]))])
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

def compute_x_multi_scenario(reservoir,X,U,V,reward,pen_low,pen_high, pen_final, itr):
    initial_x = np.zeros((S+1,NTrain))
    initial_x[0] = reservoir.initial_level
    np.random.seed(19*itr)
    controls = np.zeros((S,NTrain))
    
    for s in range(S):
        
        V_fut = interp1d(X, V[:, s+1])
        for j,k_s in enumerate(np.random.permutation(range(NTrain))):
        
            if s==S-1:
                pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
            else :
                pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])
            Gs = lambda x: min([reward[s][k_s][i][0]*x+reward[s][k_s][i][1] for i in range(len(reward[s][k_s]))])

            Vu = float('-inf')

            for i_fut in range(len(X)):
                u = -X[i_fut] + initial_x[s,j] + reservoir.inflow[s,k_s]*H
                if -reservoir.P_pump[7*s]*H <= u <= reservoir.P_turb[7*s]*H:
                    G = Gs(u)
                    penalty = pen(X[i_fut])
                    if (G + V[i_fut, s+1]+penalty) > Vu:
                        Vu = G + V[i_fut, s+1]+penalty
                        xf = X[i_fut]

            for u in range(len(U[s][k_s])):
                state_fut = min(reservoir.capacity,initial_x[s,j] - U[s][k_s][u] + reservoir.inflow[s,k_s]*H) 
                if 0 <= state_fut :
                    penalty = pen(state_fut)
                    G = Gs(U[s][k_s][u])
                    if (G + V_fut(state_fut)+penalty) > Vu:
                        Vu = (G + V_fut(state_fut)+penalty)
                        xf = state_fut

            Umin = initial_x[s,j]+ reservoir.inflow[s,k_s]*H-reservoir.Xmin[s]
            if -reservoir.P_pump[7*s]*H <= Umin <= reservoir.P_turb[7*s]*H:
                state_fut = initial_x[s,j] - Umin + reservoir.inflow[s,k_s]*H
                penalty = pen(state_fut)
                if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
                    Vu = Gs(Umin) + V_fut(state_fut)+penalty
                    xf = state_fut

            Umax = initial_x[s,j]+ reservoir.inflow[s,k_s]*H-reservoir.Xmax[s]
            if -reservoir.P_pump[7*s]*H <= Umax <= reservoir.P_turb[7*s]*H:
                state_fut = initial_x[s,j] - Umax + reservoir.inflow[s,k_s]*H 
                penalty = pen(state_fut)
                if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
                    Vu = Gs(Umax) + V_fut(state_fut)+penalty
                    xf = state_fut

            initial_x[s+1,j] = xf
            controls[s,k_s] = min(-(initial_x[s+1,j]-initial_x[s,j]-reservoir.inflow[s,k_s]*H), reservoir.P_turb[7*s]*H)

    return(initial_x, controls)

def find_likely_control(reservoir,X,U,V,reward,pen_low,pen_high, pen_final, level_i, s, k):
    
    V_fut = interp1d(X, V[:, s+1])
        
        
    if s==S-1:
        pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
    else :
        pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])
    Gs = lambda x: min([reward[s][k][i][0]*x+reward[s][k][i][1] for i in range(len(reward[s][k]))])

    Vu = float('-inf')

    for i_fut in range(len(X)):
        u = -X[i_fut] + level_i + reservoir.inflow[s,k]*H
        if -reservoir.P_pump[7*s]*H <= u <= reservoir.P_turb[7*s]*H:
            G = Gs(u)
            penalty = pen(X[i_fut])
            if (G + V[i_fut, s+1]+penalty) > Vu:
                Vu = G + V[i_fut, s+1]+penalty
                control = u

    for u in range(len(U[s][k])):
        state_fut = min(reservoir.capacity,level_i - U[s][k][u] + reservoir.inflow[s,k]*H) 
        if 0 <= state_fut :
            penalty = pen(state_fut)
            G = Gs(U[s][k][u])
            if (G + V_fut(state_fut)+penalty) > Vu:
                Vu = (G + V_fut(state_fut)+penalty)
                control = U[s][k][u]

    Umin = level_i+ reservoir.inflow[s,k]*H-reservoir.Xmin[s]
    if -reservoir.P_pump[7*s]*H <= Umin <= reservoir.P_turb[7*s]*H:
        state_fut = level_i - Umin + reservoir.inflow[s,k]*H
        penalty = pen(state_fut)
        if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
            Vu = Gs(Umin) + V_fut(state_fut)+penalty
            control = Umin

    Umax = level_i+ reservoir.inflow[s,k]*H-reservoir.Xmax[s]
    if -reservoir.P_pump[7*s]*H <= Umax <= reservoir.P_turb[7*s]*H:
        state_fut = level_i - Umax + reservoir.inflow[s,k]*H 
        penalty = pen(state_fut)
        if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
            Vu = Gs(Umax) + V_fut(state_fut)+penalty
            control = Umax

    return(control)

def compute_upper_bound(reservoir,list_models,X,U,V,G,pen_low,pen_high,pen_final,control_basis,basis):
    
    current_itr = np.zeros((S,NTrain,3))
    current_basis =[[[] for k in range(NTrain)] for s in range(S)]
    
    cout = 0
    controls = np.zeros((S,NTrain))
    for k in range(NTrain):
        
        level_i = reservoir.initial_level
        for s in range(S):
            print(f"{k} {s}",end="\r")
            m = list_models[s][k]

            nb_cons = m[0].attributes.rows

            m[0].chgmcoef(m[1],[m[2]],[-1])
            m[0].chgrhs(m[1],[0])

            m[0].chgobj([m[6],m[5]], [1,1])

            likely_control = find_likely_control(reservoir,X,U,V,G,pen_low,pen_high, pen_final, level_i, s, k)
            
            u = np.argmin(np.abs(np.array(control_basis[s][k])-likely_control))
            m[0].loadbasis(basis[u][s][k][0],basis[u][s][k][1])
        
            for j in range(len(X)-1):
                if (V[j+1, s+1]<float('inf'))&(V[j, s+1]<float('inf')):
                    m[0].addConstraint(m[5] >= (-V[j+1, s+1] + V[j, s+1]) / (X[j+1] - X[j]) * (m[4] - X[j]) - V[j, s+1])

            cst_initial_level = m[3] == level_i
            m[0].addConstraint(cst_initial_level)

            rbas = []
            cbas = []

            debut_1 = time()
            m[0].lpoptimize()
            fin_1 = time()

            if m[0].attributes.lpstatus==1:

                m[0].getbasis(rbas,cbas)
                current_basis[s][k] = (rbas[:nb_cons],cbas)
                control_basis[s][k].append(m[0].getSolution(m[2]))

                beta = m[0].getObjVal()
                xf = m[0].getSolution(m[4])
                z = m[0].getSolution(m[5])
                y = m[0].getSolution(m[6])
                m[0].delConstraint(range(nb_cons,m[0].attributes.rows))
                m[0].chgmcoef(m[1],[m[2]],[0])
                
                m[0].chgobj([m[6],m[5]], [0,0])
                cout += beta
                controls[s,k]=-(xf-level_i-reservoir.inflow[s,k]*H)
                level_i = xf
                if s!=S-1:
                    cout += - z - y

                itr = m[0].attributes.SIMPLEXITER
                t = m[0].attributes.TIME

            else :
                raise(ValueError)
            current_itr[s,k] = (itr,t,fin_1-debut_1)

        upper_bound = cout/NTrain
    return(upper_bound, controls, current_itr, current_basis, control_basis)

def update_reward_approximation(reward,points,lamb,beta,new_control):

    Gs = lambda x: min([reward[i][0]*x+reward[i][1] for i in range(len(reward))])
    new_cut = lambda x:-lamb*x -beta + lamb*new_control
    new_reward = []
    new_points = [points[0]]

    if (len(points)!=len(reward)+1) :
        print(reward)
        print(points)
        raise(ValueError)
    
    for i in range(len(points)):
        if i==0:
            if (new_cut(points[i]) < Gs(points[i])) :
                new_reward.append((-lamb,-beta + lamb*new_control))
                if (new_cut(points[i+1]) >= Gs(points[i+1])) :
                    if -lamb-reward[i][0]!=0:
                        new_points.append(-(-beta + lamb*new_control-reward[i][1])/(-lamb-reward[i][0]))
                        new_reward.append(reward[i])
            elif (new_cut(points[i]) >= Gs(points[i])):
                new_reward.append(reward[i])
                if (new_cut(points[i+1]) < Gs(points[i+1])) :
                    if -lamb-reward[i][0]!=0:
                        new_points.append(-(-beta + lamb*new_control-reward[i][1])/(-lamb-reward[i][0]))
                        new_reward.append((-lamb,-beta + lamb*new_control))
        elif i==len(points)-1:
            new_points.append(points[-1])
        else :
            if (new_cut(points[i]) >= Gs(points[i])) :
                new_reward.append(reward[i])
                new_points.append(points[i])
                if (new_cut(points[i+1]) < Gs(points[i+1])) :
                    if -lamb-reward[i][0]!=0:
                        new_reward.append((-lamb,-beta + lamb*new_control))
                        new_points.append(-(-beta + lamb*new_control-reward[i][1])/(-lamb-reward[i][0]))
            elif (new_cut(points[i]) < Gs(points[i])) and (new_cut(points[i+1]) >= Gs(points[i+1])) :
                if -lamb-reward[i][0]!=0:
                    new_reward.append(reward[i])
                    new_points.append(-(-beta + lamb*new_control-reward[i][1])/(-lamb-reward[i][0]))
                
    
    return(new_reward,new_points)

def calculate_reward(controls, list_models, basis, control_basis, G, U, i):
    current_itr = np.zeros((S,NTrain,3))
    current_basis =[[[] for k in range(NTrain)] for s in range(S)]
    
    for k in range(NTrain):
        rstatus = []
        cstatus = []    
        for s in range(S):
            print(f"{k} {s}",end="\r")

            beta, lamb, itr, t, rbas, cbas, rstatus,cstatus, debut_1, fin_1 = modify_weekly_problem_itr(m=list_models[s][k],control=controls[s][k],rstatus=rstatus,cstatus=cstatus,i=i,basis=[basis[u][s][k] for u in range(len(basis))],control_basis=control_basis[s][k])

            current_basis[s][k] = (rbas,cbas)
            control_basis[s][k].append(controls[s][k])
            G[s][k],U[s][k] = update_reward_approximation(G[s][k],U[s][k],lamb,beta,controls[s][k])
            
            current_itr[s,k] = (itr,t,fin_1-debut_1)

    return(current_basis,current_itr,control_basis,G,U)

def itr_control(reservoir:Reservoir, output_path, pen_low, pen_high,X, N, pen_final, tol_gap):

    tot_t = []
    debut = time()
    
    list_models = [[] for i in range(S)]
    for s in range(S):
        for k in range(NTrain):
            m = create_weekly_problem_itr(k=k,s=s,output_path=output_path,reservoir=reservoir,pen_low=pen_low,pen_high=pen_high,pen_final=pen_final)
            list_models[s].append(m)
    
    V = np.zeros((len(X), S+1))
    G = [[[(0,0)] for k in range(NTrain)] for s in range(S)]
    U = [[[-reservoir.P_pump[7*s]*H,reservoir.P_turb[7*s]*H] for k in range(NTrain)] for s in range(S)]
    
    itr_tot = []
    basis = []
    control_basis = [[[] for k in range(NTrain)] for s in range(S)]
    controls_upper = []
    traj = []

    i = 0
    gap = 1e3
    fin = time()
    tot_t.append(fin-debut)
    while (gap>=tol_gap and gap>=0) and i <N : #and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(reservoir=reservoir,X=X,U=U,V=V,reward=G,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final, itr=i)
        traj.append(np.array(initial_x))

        current_basis,current_itr,control_basis,G,U = calculate_reward(controls=controls, list_models=list_models, basis=basis, control_basis=control_basis, G=G, U=U, i=i)
        itr_tot.append(current_itr)
        basis.append(current_basis)

        V = calculate_VU(reward=G,reservoir=reservoir,X=X,U=U,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final)
        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir.initial_level)
            
        upper_bound, controls, current_itr, current_basis, control_basis = compute_upper_bound(reservoir,list_models,X,U,V,G,pen_low,pen_high,pen_final,control_basis,basis)
        itr_tot.append(current_itr)
        basis.append(current_basis)        
        controls_upper.append(controls)
        
        gap = upper_bound+V0
        print(gap, upper_bound,-V0)
        gap = gap/-V0
        i+=1
        fin = time()
        tot_t.append(fin-debut)
    return (V, G, np.array(itr_tot),tot_t, U, control_basis, controls_upper, traj)