import numpy as np


def new_remix_hydro(
    G,
    H,
    D,
    P_max,
    P_min,
    initial_level,
    capa,
    inflow,
    overflow,
    pumping,
    spillage,
    dtg_mrg,
):
    new_H = np.array(H, copy=True, dtype=np.float64)
    new_D = np.array(D, copy=True, dtype=np.float64)

    loop = 1000

    G_plus_H = G + new_H

    level = initial_level + np.cumsum(inflow - overflow + pumping - new_H)

    # Prendre aussi en compte le pompage et l'overlfow (que l'on ne touche pas) dans le calcul des niveaux de stock

    eps = 1e-2

    top = max(G + H + D + 1)
    bottom = min(G + H - 1)

    filter_hours_remix = (spillage + dtg_mrg == 0) * (H + D > 0)

    while loop > 0:

        tried_creux = np.zeros_like(H)

        delta = 0

        possible_creux = np.argwhere(
            (new_D > 0) * (new_H < P_max) * (tried_creux == 0) * filter_hours_remix
        )[:, 0]

        while len(possible_creux) > 0:
            idx_creux = np.argmin(
                np.where(
                    (new_D > 0)
                    * (new_H < P_max)
                    * (tried_creux == 0)
                    * filter_hours_remix,
                    G_plus_H,
                    top,
                )
            )

            tried_pic = np.zeros_like(H)

            possible_pic = np.argwhere(
                (new_H > P_min)
                * (G_plus_H >= G_plus_H[idx_creux] + eps)
                * (tried_pic == 0)
                * filter_hours_remix
            )[:, 0]

            while len(possible_pic) > 0:

                idx_pic = np.argmax(
                    np.where(
                        (new_H > P_min)
                        * (G_plus_H >= G_plus_H[idx_creux] + eps)
                        * (tried_pic == 0)
                        * filter_hours_remix,
                        G_plus_H,
                        bottom,
                    )
                )

                if idx_creux < idx_pic:
                    intermediate_level = level[idx_creux:idx_pic]
                    max_pic = capa
                    max_creux = min(intermediate_level)
                else:
                    intermediate_level = level[idx_pic:idx_creux]
                    max_pic = capa - max(intermediate_level)
                    max_creux = capa

                max_pic = min(new_H[idx_pic] - P_min[idx_pic], max_pic)
                max_creux = min(
                    P_max[idx_creux] - new_H[idx_creux], new_D[idx_creux], max_creux
                )

                dif_pic_creux = max(G_plus_H[idx_pic] - G_plus_H[idx_creux], 0)

                delta = max(min(max_pic, max_creux, dif_pic_creux / 2), 0)

                if delta > 0:

                    new_H[idx_pic] = new_H[idx_pic] - delta
                    new_H[idx_creux] = new_H[idx_creux] + delta
                    new_D[idx_pic] = H[idx_pic] + D[idx_pic] - new_H[idx_pic]
                    new_D[idx_creux] = H[idx_creux] + D[idx_creux] - new_H[idx_creux]
                    break

                else:
                    tried_pic[idx_pic] = 1
                    possible_pic = np.argwhere(
                        (new_H > P_min)
                        * (G_plus_H >= G_plus_H[idx_creux] + eps)
                        * (tried_pic == 0)
                        * filter_hours_remix
                    )[:, 0]

            if delta > 0:
                break

            else:
                tried_creux[idx_creux] = 1
                possible_creux = np.argwhere(
                    (new_D > 0)
                    * (new_H < P_max)
                    * (tried_creux == 0)
                    * filter_hours_remix
                )[:, 0]

        loop -= 1

        G_plus_H = G + new_H

        level = initial_level + np.cumsum(inflow - overflow + pumping - new_H)

        if delta == 0:
            break

    return new_H, new_D, level
