XTC = "T2-Nnnn-Iiii_L_MDfff_FIX.xtc"
TPR = "T2-Nnnn-Iiii_L_MDfff.tpr"
NAME = XTC[:-8]

import numpy as np
from scipy.spatial.distance import cdist
from MDAnalysis import *

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt

sel = {
"NP1"       : U.select_atoms("bynum 1:126"),
"NP2"       : U.select_atoms("bynum 127:252"),
"SOL"       : U.select_atoms("resname PW PNA PCL MNP")
}

props_interfacial_density = {
'ref1'       : "NP1",
'ref2'       : "NP2",
'target'    : "SOL",
'dx'        : 1, #A, grid spacing
'edgesize'  : 30, #A, size of the square grid
}


def histogramize(props, data, np_dist_av):
    n_b1 = int(np_dist_av//props['dx'] + 1)
    n_b2 = int(props['edgesize']//props['dx'] + 1)
    b1_bins = np.linspace(-np_dist_av/2, np_dist_av/2, n_b1)
    b2_bins = np.linspace(-props['edgesize']/2, props['edgesize']/2, n_b2)

    populations = np.zeros((n_b1-1, n_b2-1, n_b2-1))
    for i, (bin1_low, bin1_high) in enumerate(zip(b1_bins[:-1], b1_bins[1:])):
        slice_pts1 = data[np.logical_and(data[:,0]>= bin1_low, data[:,0]<=bin1_high)]

        for j, (bin2_low, bin2_high) in enumerate(zip(b2_bins[:-1], b2_bins[1:])):
            slice_pts2 = slice_pts1[np.logical_and(slice_pts1[:,1]>=bin2_low, slice_pts1[:,1]<=bin2_high)]

            for k, (bin3_low, bin3_high) in enumerate(zip(b2_bins[:-1], b2_bins[1:])):
                slice_pts3 = slice_pts2[np.logical_and(slice_pts2[:,2]>=bin3_low, slice_pts2[:,2]<=bin3_high)]

                populations[i,j,k] = np.mean(slice_pts3[:,3])

    populations = np.nan_to_num(populations, 0.0)
    return populations

def interfacial_density(props):
    n_frames = len(U.trajectory)
    g_ref1 = sel[props['ref1']]
    g_ref2 = sel[props['ref2']]
    g_target = sel[props['target']]
    np_dist_av = np.mean([np.linalg.norm(g_ref1.center_of_mass() - g_ref2.center_of_mass()) for ts in U.trajectory])

    data = np.array([[0,0,0,0]])
    for ts in U.trajectory:
        b1 = g_ref2.center_of_mass() - g_ref1.center_of_mass() #cv1
        np_dist = np.linalg.norm(b1)
        b1 = b1/np_dist
        b2 = np.array([1, 0, -b1[0]/b1[2]])
        b2 = b2/np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        b3 = b3/np.linalg.norm(b3)
        Q = np.array([b1, b2, b3]).T

        np_midpoint = (g_ref1.center_of_mass() + g_ref2.center_of_mass())/2
        pts = np.linalg.solve(Q, (g_target.positions - np_midpoint).T).T
        condition_b1 = np.logical_and(pts[:,0] >= -np_dist/2, pts[:,0] <= np_dist/2)
        condition_b2 = np.logical_and(pts[:,1] >= -props['edgesize']/2, pts[:,1] <= props['edgesize']/2)
        condition_b3 = np.logical_and(pts[:,2] >= -props['edgesize']/2, pts[:,2] <= props['edgesize']/2)
        conditions = np.logical_and(condition_b1, np.logical_and(condition_b2, condition_b3))
        pts_relevant = pts[conditions,:]
        charges = g_target.charges[conditions]
        coords_charge = np.append(pts_relevant, np.array([charges]).T, axis=1)
        data = np.vstack((data, coords_charge))

    populations = histogramize(props, data, np_dist_av)
    return populations, np_dist_av

def write_populations(props, data, np_dist_av):
    n_b1 = int(np_dist_av//props['dx'] + 1)
    bins1 = np.linspace(-np_dist_av/2, np_dist_av/2, n_b1)
    bins1 = (bins1[:-1] + bins1[1:])/2
    f = open(NAME+"_interdensity.sfu", "w")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("Each slice is divided by $(CVvalue)")
    f.write("#Interfacial charge distribution (e)\n")
    for i, cv1 in zip(range(len(data[:,0,0])), bins1):
        f.write("$ CV = {:<8.3f} nm\n".format(cv1/10)) #A to nm
        for j in range(len(data[0,:,0])):
            for k in range(len(data[0,0,:])):
                f.write("{:<8.3f} ".format(data[i,j,k]))
            f.write("\n")
    f.close()

data, np_dist_av = interfacial_density(props_interfacial_density)
write_populations(props_interfacial_density, data, np_dist_av)
