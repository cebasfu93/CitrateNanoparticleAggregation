XTC = "T1-Nnnn-Iiii_L_PRO1_FIX.xtc"
TPR = "T1-Nnnn-Iiii_L_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from Extras import *
from MDAnalysis import *
import MDAnalysis
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"NP" : U.select_atoms("bynum 1:126"),
"NA"       : U.select_atoms("resname PNA"),
"CL"       : U.select_atoms("resname PCL"),
"SOL"      : U.select_atoms("resname PW"),
}

props_charge = {
'ref'       : "NP",
'targets'   : ["NP", "NA", "CL", "SOL"], #Should add to all the system
'start_ps'  : 0,
'stop_ps'   : 100000,
'r_range'   : (0, 50),
'nbins'     : 250,
'dt'        : 20
}

def charge(props):
    charges = {}
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    RC = center_bins(R)
    
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))

    for target in props['targets']:
        n_frames = 0
        print("Current target: {}".format(target))
        Q = []
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt'] == 0:
                n_frames += 1
                dist = np.linalg.norm(np.subtract(g_target.positions, g_ref.center_of_mass()), axis = 1)
                for r1, r2 in zip(R[:-1], R[1:]):
                    q = np.sum(g_target.charges[np.logical_and(dist >= r1, dist  < r2)])
                    Q.append(q)
        Q = np.mean(np.array(Q).reshape((n_frames, len(RC))), axis = 0)
        Q = np.cumsum(Q)
        charges[target] = Q
    Q_tot = []
    for key, val in charges.items():
        Q_tot.append(val)
    Q_tot = np.array(Q_tot)
    Q_tot = np.sum(Q_tot, axis = 0)
    charges["TOTAL"] = Q_tot
    return RC, charges

def write_charge(space, charge_dict, properties):
    f = open(NAME + "_charge.sfu", 'w')
    values = []
    f.write("#Charges in (e)\n")
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in charge_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i])) #1660.539 for uma/A3 to kg/m3
        f.write("\n")
    f.close()


r, charges = charge(props_charge)
write_charge(r, charges, props_charge)

"""
fig = plt.figure()
ax = plt.axes()
for target, charge in charges.items():
    ax.plot(r/10, charge, label = target)
plt.legend()
plt.show()
"""
