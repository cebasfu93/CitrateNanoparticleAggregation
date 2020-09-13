XTC = "T1-Nnnn-Iiii_L_PRO1_FIX.xtc"
TPR = "T1-Nnnn-Iiii_L_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
from MDAnalysis.analysis.rdf import InterRDF
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"NP"       : U.select_atoms("bynum 1:126"),
"NA"       : U.select_atoms("resname PNA"),
"CL"       : U.select_atoms("resname PCL"),
"SOL"      : U.select_atoms("resname PW"),
}

props_rdf = {
'ref'       : "NP",
'targets'    : ["NA", "CL", "SOL"],
'start_ps'  : 0,
'stop_ps'   : 100000,
'r_range'   : (0, 45),
'nbins'     : 250,
'dt'        : 20
}

def rdf_mdanalysys(props):
    rdfs = {}
    for target in props['targets']:
        rdfs[target] = InterRDF(g1 = sel[props['ref']], g2 = sel[target], start = ps_frame(props['start_ps'], DT), stop = ps_frame(props['stop_ps'], DT), verbose = True, range=props['r_range'], nbins = props['nbins'])
        rdfs[target].run()
    return rdfs

def rdf_manual(props):
    rdfs = {}
    n_frames = ps_frame(props['stop_ps'], DT) - ps_frame(props['start_ps'], DT) + 1
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    dR = R[1] - R[0]
    RC = center_bins(R)
    vols = 4./3.*math.pi*(np.power(R[1:],3) - np.power(R[:-1],3))
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    for target in props['targets']:
        print("Current target: {}".format(target))
        rdf = []
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time % props['dt'] == 0:
                x_ref = g_ref.center_of_mass()
                x_target = g_target.positions
                dx = np.subtract(x_target, x_ref)
                dists = np.linalg.norm(dx, axis = 1)
                rdf.append(dists)
            elif ts.time > props['stop_ps']:
                break
        rdf = np.array(rdf).flatten()
        counts, bins = np.histogram(rdf, bins = R)
        counts = counts/n_frames
        dens_homo = np.sum(counts)/(4./3.*math.pi*props['r_range'][1]**3)
        counts = counts/dens_homo
        counts = counts/vols       

        rdfs[target] = counts*10# A-1 to nm-1

    return RC, rdfs

def write_rdf(space, rdf_dict, properties):
    f = open(NAME + "_rdf.sfu", 'w')
    values = []
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in rdf_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i] / 10))#A to nm, strange but true
        f.write("\n")
    f.close()

r, rdfs = rdf_manual(props_rdf)
write_rdf(r, rdfs, props_rdf)
"""
fig = plt.figure()
ax = plt.axes()
for target, rdf in rdfs.items():
    ax.plot(r, rdf, label = target)
plt.legend()
plt.show()
"""
