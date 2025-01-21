import numpy as np
import matplotlib.pyplot as plt

import xpress as xp

xp.controls.outputlog = 1
xp.controls.miprelstop = 1e-10

T = 10


def list_to_print(L):
    return "[" + ", ".join([str(round(l, 10)) for l in L]) + "]"


def plot_empilement(G, H, D):
    hours_week = range(T + 1)
    fig, ax = plt.subplots()
    ax.stackplot(
        hours_week,
        [np.append(G, 0), np.append(H, 0), np.append(D, 0)],
        labels=["G", "H", "D"],
        baseline="zero",
        alpha=0.7,
        step="post",
    )
    fig.legend()


def plot_bounds(H, bornes):
    hours_week = range(T)
    fig, ax = plt.subplots()
    ax.plot(hours_week, H, label="H", alpha=0.7)
    ax.plot(hours_week, bornes[:, 0], "--", color="tab:blue", alpha=0.7, label="p_min")
    ax.plot(hours_week, bornes[:, 1], "-.", color="tab:blue", alpha=0.7, label="p_max")
    fig.legend()


def plot_level(initial_level, capa, H, inflow):
    fig, ax2 = plt.subplots()
    level = np.concatenate(
        [np.array([initial_level]), initial_level + np.cumsum(inflow - H)]
    )
    ax2.plot(level, "tab:green", label="Level of stock")
    ax2.plot(np.zeros_like(level), "--", color="tab:green", label="level_min")
    ax2.plot(np.ones_like(level) * capa, "-.", color="tab:green", label="level_max")
    fig.legend()


def orignal_algorithm(G, H, D, P):
    HE = np.zeros(T)
    DE = np.zeros(T)

    remix = []

    for i in range(T):
        if D[i] + H[i] > 0:
            remix.append(True)
        else:
            remix.append(False)

    WH = sum(H)

    L = G + D + H

    bottom = np.min(G)
    top = np.max(L)

    ecart = 1.0
    loop = 100

    while (abs(ecart) > 0.01) and loop > 0:
        niveau = (top + bottom) * 0.5
        stock = 0.0

        for i in range(T):
            if remix[i]:
                if niveau > L[i]:
                    HEi = H[i] + D[i]
                    if HEi > P[i]:
                        HEi = P[i]
                        DE[i] = H[i] + D[i] - HEi
                    else:
                        DE[i] = 0
                else:
                    if G[i] > niveau:
                        HEi = 0
                        DE[i] = H[i] + D[i]
                    else:
                        HEi = niveau - G[i]
                        if HEi > P[i]:
                            HEi = P[i]
                        DE[i] = H[i] + D[i] - HEi
                stock += HEi
                HE[i] = HEi
            else:
                HE[i] = 0
                DE[i] = 0

        ecart = WH - stock
        if ecart > 0.0:
            bottom = niveau
        else:
            top = niveau

        loop -= 1
    print(loop, ecart, bottom, top)

    return HE, DE


def simplified_algorithm(G, H, D, P_max):
    HE = np.zeros(T)
    DE = np.zeros(T)

    remix = []

    for i in range(T):
        if D[i] + H[i] > 0:
            remix.append(True)
        else:
            remix.append(False)

    WH = sum(H)

    L = G + D + H

    bottom = np.min(G)
    top = np.max(L)

    ecart = 1.0
    loop = 100

    while (abs(ecart) > 0.01) and loop > 0:
        niveau = (top + bottom) * 0.5
        stock = 0.0

        for i in range(T):
            if remix[i]:
                if niveau > L[i]:
                    HEi = H[i] + D[i]
                else:
                    if G[i] > niveau:
                        HEi = 0
                    else:
                        HEi = niveau - G[i]

                HEi = min(HEi, P_max[i])
                stock += HEi
                HE[i] = HEi
                DE[i] = H[i] + D[i] - HEi
            else:
                HE[i] = 0
                DE[i] = 0

        ecart = WH - stock
        if ecart > 0.0:
            bottom = niveau
        else:
            top = niveau

        loop -= 1
    print(loop, ecart, bottom, top)

    return HE, DE


def simplified_algorithm_with_p_min(G, H, D, P_max, P_min):
    HE = np.zeros(T)
    DE = np.zeros(T)

    remix = []

    for i in range(T):
        if D[i] + H[i] > 0:
            remix.append(True)
        else:
            remix.append(False)

    WH = sum(H)

    L = G + D + H

    bottom = np.min(G)
    top = np.max(L)

    ecart = 1.0
    loop = 100

    while (abs(ecart) > 0.01) and loop > 0:
        niveau = (top + bottom) * 0.5
        stock = 0.0

        for i in range(T):
            if remix[i]:
                if niveau > L[i]:
                    HEi = H[i] + D[i]
                else:
                    if G[i] > niveau:
                        HEi = 0
                    else:
                        HEi = niveau - G[i]

                HEi = max(P_min[i], min(HEi, P_max[i]))
                stock += HEi
                HE[i] = HEi
                DE[i] = H[i] + D[i] - HEi
            else:
                HE[i] = 0
                DE[i] = 0

        ecart = WH - stock
        if ecart > 0.0:
            bottom = niveau
        else:
            top = niveau

        loop -= 1
    print(loop, ecart, bottom, top)

    return HE, DE


def simplified_algorithm_with_p_min_and_capa(
    G, H, D, P_max, P_min, initial_level, capa, inflow
):
    HE = np.zeros(T)
    DE = np.zeros(T)

    remix = []

    for i in range(T):
        if D[i] + H[i] > 0:
            remix.append(True)
        else:
            remix.append(False)

    WH = sum(H)

    L = G + D + H

    bottom = np.min(G)
    top = np.max(L)

    ecart = 1.0
    loop = 1000

    while (abs(ecart) > 0.01) and loop > 0:
        niveau = (top + bottom) * 0.5
        stock = 0.0
        level = initial_level

        for i in range(T):
            if remix[i]:
                if niveau > L[i]:
                    HEi = (H[i] + D[i]) * 0.5
                else:
                    if G[i] > niveau:
                        HEi = 0
                    else:
                        HEi = (niveau - G[i]) * 0.5

                if (
                    HEi < level + inflow[i] - capa
                    and level + inflow[i] - capa <= P_max[i]
                ):
                    HEi = level + inflow[i] - capa
                else:
                    HEi = HE[i]
                if HEi > level + inflow[i] and level + inflow[i] >= P_min[i]:
                    HEi = level + inflow[i]
                else:
                    HEi = HE[i]
                assert HEi >= P_min[i]
                assert HEi <= P_max[i]
                stock += HEi
                HE[i] = HEi
                DE[i] = H[i] + D[i] - HEi
            else:
                HE[i] = 0
                DE[i] = 0
            level = level + inflow[i] - HE[i]
            assert level >= 0
            assert level <= capa

        ecart = WH - stock
        if ecart > 0.0:
            bottom = niveau
        else:
            top = niveau

        loop -= 1
    print(loop, ecart, bottom, top)
    if loop == 0:
        return H, D

    return HE, DE


def new_heuristic(G, H, D, P_max, P_min, initial_level, capa, inflow):
    new_H = np.array(H, copy=True, dtype=np.float32)
    new_D = np.array(D, copy=True, dtype=np.float32)

    loop = 1000

    top = max(G + H + D + 1)

    while loop > 0:
        G_plus_H = G + new_H

        idx_pic = np.argmax(np.where(new_H > 0, G_plus_H, 0))
        idx_creux = np.argmin(np.where((new_D > 0) * (new_H < P_max), G_plus_H, top))

        if abs(G_plus_H[idx_pic] - G_plus_H[idx_creux]) <= 1e-2:
            break

        max_pic = new_H[idx_pic]
        max_creux = min(P_max[idx_creux] - new_H[idx_creux], new_D[idx_creux])

        dif_pic_creux = G_plus_H[idx_pic] - G_plus_H[idx_creux]

        delta = min(max_pic, max_creux, dif_pic_creux / 2)

        new_H[idx_pic] = new_H[idx_pic] - delta
        new_H[idx_creux] = new_H[idx_creux] + delta
        new_D[idx_pic] = H[idx_pic] + D[idx_pic] - new_H[idx_pic]
        new_D[idx_creux] = H[idx_creux] + D[idx_creux] - new_H[idx_creux]

        loop -= 1
    print(loop)
    return new_H, new_D


def new_heuristic_with_p_min(G, H, D, P_max, P_min, initial_level, capa, inflow):
    new_H = np.array(H, copy=True, dtype=np.float32)
    new_D = np.array(D, copy=True, dtype=np.float32)

    loop = 1000

    top = max(G + H + D + 1)

    while loop > 0:
        G_plus_H = G + new_H

        idx_pic = np.argmax(np.where((new_H > P_min), G_plus_H, 0))
        idx_creux = np.argmin(np.where((new_D > 0) * (new_H < P_max), G_plus_H, top))

        if abs(G_plus_H[idx_pic] - G_plus_H[idx_creux]) <= 1e-2:
            break

        max_pic = new_H[idx_pic] - P_min[idx_pic]
        max_creux = min(P_max[idx_creux] - new_H[idx_creux], new_D[idx_creux])

        dif_pic_creux = G_plus_H[idx_pic] - G_plus_H[idx_creux]

        delta = min(max_pic, max_creux, dif_pic_creux / 2)

        new_H[idx_pic] = new_H[idx_pic] - delta
        new_H[idx_creux] = new_H[idx_creux] + delta
        new_D[idx_pic] = H[idx_pic] + D[idx_pic] - new_H[idx_pic]
        new_D[idx_creux] = H[idx_creux] + D[idx_creux] - new_H[idx_creux]

        loop -= 1
    print(loop)
    return new_H, new_D


def new_heuristic_with_p_min_and_capa_random(
    G, H, D, P_max, P_min, initial_level, capa, inflow
):
    new_H = np.array(H, copy=True, dtype=np.float32)
    new_D = np.array(D, copy=True, dtype=np.float32)

    loop = 1000

    top = max(G + H + D + 1)

    while loop > 0:

        G_plus_H = G + new_H

        level = initial_level + np.cumsum(inflow - new_H)
        possible_pic = np.argwhere((new_H > P_min) * (level < capa))[:, 0]
        idx_pic = np.random.choice(
            possible_pic, p=G_plus_H[possible_pic] / sum(G_plus_H[possible_pic])
        )
        possible_creux = np.argwhere(
            (new_D > 0)
            * (new_H < P_max)
            * (level > 0)
            * (G_plus_H <= G_plus_H[idx_pic])
        )[:, 0]
        if len(possible_creux) != 0:

            idx_creux = np.random.choice(
                possible_creux,
                p=(top - G_plus_H[possible_creux])
                / sum((top - G_plus_H[possible_creux])),
            )

            if idx_creux < idx_pic:
                intermediate_level = level[idx_creux : idx_pic + 1]
            else:
                intermediate_level = level[idx_pic : idx_creux + 1]

            max_pic = min(
                new_H[idx_pic] - P_min[idx_pic], capa - max(intermediate_level)
            )
            max_creux = min(
                P_max[idx_creux] - new_H[idx_creux],
                new_D[idx_creux],
                min(intermediate_level),
            )

            dif_pic_creux = max(G_plus_H[idx_pic] - G_plus_H[idx_creux], 0)

            delta = max(min(max_pic, max_creux, dif_pic_creux / 2), 0)

            new_H[idx_pic] = new_H[idx_pic] - delta
            new_H[idx_creux] = new_H[idx_creux] + delta
            new_D[idx_pic] = H[idx_pic] + D[idx_pic] - new_H[idx_pic]
            new_D[idx_creux] = H[idx_creux] + D[idx_creux] - new_H[idx_creux]

        loop -= 1

        level = initial_level + np.cumsum(inflow - new_H)

        idx_pic = np.argmax(np.where((new_H > P_min) * (level < capa), G_plus_H, 0))
        idx_creux = np.argmin(
            np.where((new_D > 0) * (new_H < P_max) * (level > 0), G_plus_H, top)
        )

        if abs(G_plus_H[idx_pic] - G_plus_H[idx_creux]) <= 1e-2:
            break

    print(loop)
    return new_H, new_D


def optimization_problem(G, H, D, P_max, P_min, initial_level, capa, inflow):
    p = xp.problem()

    eps = 0
    M = max([G[i] + H[i] + D[i] for i in range(T)])

    HE = [xp.var(lb=0, ub=P_max[i]) for i in range(T)]
    DE = [xp.var(lb=0) for i in range(T)]
    # niveau = [xp.var(lb=0) for i in range(T+1)]
    niveau = xp.var(lb=0)
    marge_p_max = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_d = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_p_min = [xp.var(vartype=xp.binary) for i in range(T)]

    p.addVariable(HE, DE, niveau, marge_p_max, marge_d, marge_p_min)

    p.addConstraint(HE[i] + DE[i] == H[i] + D[i] for i in range(T))
    p.addConstraint(xp.Sum(HE) == sum(H))

    p.addConstraint(
        marge_p_max[i] >= (P_max[i] - HE[i] - eps) / P_max[i] for i in range(T)
    )
    p.addConstraint(marge_p_min[i] >= (HE[i]) / P_max[i] for i in range(T))
    p.addConstraint(
        marge_d[i] >= (DE[i] - eps) / (G[i] + H[i] + D[i]) for i in range(T)
    )

    p.addConstraint(
        G[i] + HE[i] >= niveau - M * (2 - marge_d[i] - marge_p_max[i]) for i in range(T)
    )
    p.addConstraint(G[i] + HE[i] <= niveau + M * (1 - marge_p_min[i]) for i in range(T))

    p.setObjective(niveau)

    p.solve()

    assert "optimal" in p.getProbStatusString()

    return p.getSolution(HE, DE, marge_p_max, marge_d, niveau)


def optimization_problem_p_min(G, H, D, P_max, P_min, initial_level, capa, inflow):
    p = xp.problem()

    eps = 0
    M = max([G[i] + H[i] + D[i] for i in range(T)])

    HE = [xp.var(lb=P_min[i], ub=P_max[i]) for i in range(T)]
    DE = [xp.var(lb=0) for i in range(T)]
    # niveau = [xp.var(lb=0) for i in range(T+1)]
    niveau = xp.var(lb=0)
    marge_p_max = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_d = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_p_min = [xp.var(vartype=xp.binary) for i in range(T)]

    p.addVariable(HE, DE, niveau, marge_p_max, marge_d, marge_p_min)

    p.addConstraint(HE[i] + DE[i] == H[i] + D[i] for i in range(T))
    p.addConstraint(xp.Sum(HE) == sum(H))

    p.addConstraint(
        marge_p_max[i] >= (P_max[i] - HE[i] - eps) / P_max[i] for i in range(T)
    )
    p.addConstraint(
        marge_p_min[i] >= (HE[i] - P_min[i]) / (P_max[i] - P_min[i])
        for i in range(T)
        if P_max[i] > P_min[i]
    )
    p.addConstraint(
        marge_d[i] >= (DE[i] - eps) / (G[i] + H[i] + D[i]) for i in range(T)
    )

    p.addConstraint(
        G[i] + HE[i] >= niveau - M * (2 - marge_d[i] - marge_p_max[i]) for i in range(T)
    )
    p.addConstraint(G[i] + HE[i] <= niveau + M * (1 - marge_p_min[i]) for i in range(T))

    p.setObjective(niveau)

    p.solve()

    assert "optimal" in p.getProbStatusString()

    return p.getSolution(HE, DE, marge_p_max, marge_d, niveau)


def optimization_problem_p_min_and_capa(
    G, H, D, P_max, P_min, initial_level, capa, inflow
):
    p = xp.problem()
    # p.controls.xslp_log = -1

    eps = 0
    M = max([G[i] + H[i] + D[i] for i in range(T)])

    HE = [xp.var(lb=P_min[i], ub=P_max[i]) for i in range(T)]
    DE = [xp.var(lb=0) for i in range(T)]
    level = [xp.var(lb=0, ub=capa) for i in range(T + 1)]
    niveau = xp.var(lb=0)
    marge_p_max = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_d = [xp.var(vartype=xp.binary) for i in range(T)]
    marge_p_min = [xp.var(vartype=xp.binary) for i in range(T)]
    # marge_0 = [xp.var(vartype=xp.binary) for i in range(T)]
    # marge_capa = [xp.var(vartype=xp.binary) for i in range(T)]

    p.addVariable(HE, DE, niveau, marge_p_max, marge_d, marge_p_min, level)

    p.addConstraint(HE[i] + DE[i] == H[i] + D[i] for i in range(T))
    p.addConstraint(xp.Sum(HE) == sum(H))

    p.addConstraint(level[i + 1] == level[i] + inflow[i] - HE[i] for i in range(T))
    p.addConstraint(level[0] == initial_level)

    p.addConstraint(
        marge_p_max[i] >= (P_max[i] - HE[i] - eps) / P_max[i] for i in range(T)
    )
    p.addConstraint(
        marge_p_min[i] >= (HE[i] - P_min[i]) / (P_max[i] - P_min[i])
        for i in range(T)
        if P_max[i] > P_min[i]
    )
    p.addConstraint(
        marge_d[i] >= (DE[i] - eps) / (G[i] + H[i] + D[i]) for i in range(T)
    )
    # p.addConstraint(marge_0[i] >= level[i]/capa for i in range(T))
    # p.addConstraint(marge_capa[i] >= (capa-level[i])/capa for i in range(T))

    p.addConstraint(
        G[i] + HE[i] >= niveau - M * (2 - marge_d[i] - marge_p_max[i]) for i in range(T)
    )
    p.addConstraint(G[i] + HE[i] <= niveau + M * (1 - marge_p_min[i]) for i in range(T))

    p.setObjective(niveau)

    p.solve()

    print(p.getProbStatusString(), end="\r")

    if "optimal" in p.getProbStatusString():
        return p.getSolution(HE, DE, marge_p_max, marge_d, niveau)
    else:
        return np.array()
