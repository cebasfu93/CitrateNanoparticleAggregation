XTC = "T2-Nnnn-Iiii_EQfff_FIX.xtc"
TPR = "T2-Nnnn-Iiii_EQfff.tpr"
NAME = "T2-Nnnn-Iiii_EQfff_FIX"

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"NP1" : U.select_atoms("bynum 0:81"),
"NP2" : U.select_atoms("bynum 82:162"),
"NA"  : U.select_atoms("name NA"),
"CL"  : U.select_atoms("name CL")
}

props_polar = {
'ref1'      : "NP1", #The axis between the centroid of ref and ref2 define the reference pole
'ref2'      : "NP2",
'targets'    : ["NA", "CL"],
'start_ps'  : 0, #ps
'stop_ps'   : 10000, #ps
'down_lim'  : 12.42, #A
'up_lim'  : 17.35, #A
'dt'        : 40,
'bins'      : 40,
'plot'      : False,
}

def count_polar(props):
    N_frames = (props['stop_ps'] - props['start_ps'])/DT+1
    g_ref1 = sel[props['ref1']]
    g_ref2 = sel[props['ref2']]
    angles = np.linspace(0, np.pi, props['bins'])
    center_angles = center_bins(angles)*180/np.pi #rad to deg
    volumes = 4/3.*(props['up_lim']**3-props['down_lim']**3)*(angles[1:]-angles[:-1])

    all_dens = []
    for target in props['targets']:
        g_target = sel[target]
        print("Working on the following target: ", target)

        dots = np.array([])
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
                center1, center2 = g_ref1.centroid(), g_ref2.centroid()
                eje1 = center2 - center1
                eje1 = eje1/np.linalg.norm(eje1)
                eje2 = center1 - center2
                eje2 = eje2/np.linalg.norm(eje2)
                x_target1 = g_target.positions - center1
                x_target2 = g_target.positions - center2
                ndx_keep1 = np.where(np.logical_and(x_target1 <= props['up_lim'], x_target1 >= props['down_lim']))
                ndx_keep2 = np.where(np.logical_and(x_target2 <= props['up_lim'], x_target2 >= props['down_lim']))
                x_keep1 = x_target1[ndx_keep1]
                x_keep2 = x_target2[ndx_keep2]

                for x1 in x_target1:
                    dots = np.append(dots, np.dot(eje1, x1)/np.linalg.norm(x1))
                for x2 in x_target2:
                    dots = np.append(dots, np.dot(eje2, x2)/np.linalg.norm(x2))

            if ts.time >= props['stop_ps']:
                break

        popu = np.arccos(dots)
        counts, bins = np.histogram(popu, bins = angles)
        dens = counts/(N_frames*volumes)
        all_dens.append(dens)

    all_dens = np.array(all_dens).T

    if props['plot']:
        fig = plt.figure()
        ax = plt.axes()
        for d, dens in enumerate(all_dens):
            ax.plot(center_angles, dens, label=props['targets'][d])
        plt.legend()
        plt.show()

    return center_angles, all_dens

def write_polar(angles, densities, props):
    f = open(NAME+"_polar.sfu", "w")
    f.write("#Density (number / A3)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#angle (degrees)")
    for key in props['targets']:
        f.write("{:>15} ".format(key))
    f.write("\n")

    for i in range(len(angles)):
        f.write("{:<15.2f} ".format(angles[i]))
        for j in range(len(densities[0,:])):
            f.write("{:>15.5f} ".format(densities[i,j]))
        f.write("\n")
    f.close()

angs, dens = count_polar(props_polar)
write_polar(angs, dens, props_polar)
