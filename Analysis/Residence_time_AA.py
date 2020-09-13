XTC = "T1-Nnnn-Iiii_AA_PRO1_FIX.xtc"
TPR = "T1-Nnnn-Iiii_AA_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
from scipy.spatial.distance import cdist
from MDAnalysis import *
from Extras import *

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"NP"       : U.select_atoms("bynum 1:459"),
"NA"       : U.select_atoms("resname NA"),
}

props_residence_time = {
'ref'       : "NP",
'targets'    : ["NA"],
'from'      : 'any', #in the future also 'surf' and 'com'
'start_ps'  : 0,
'stop_ps'   : 250000,
'threshold'   : 6.0, #A from any bead of 'ref'
}

def residence_time(props):
    n_frames = ps_frame(props['stop_ps'], DT) - ps_frame(props['start_ps'], DT) + 1
    answer = []
    print(U.trajectory[-1].time)
    for target in props['targets']:
        n_atoms = sel[target].n_atoms
        res_frames = np.zeros(n_atoms)
        res_frames_av = []
        all_min_dists = np.zeros((n_atoms, n_frames))
        for t, ts in enumerate(U.trajectory):
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps']:
                print(ts.time, end='\r')
                dists = cdist(sel[target].positions, sel[props['ref']].positions)
                min_dists = np.sort(dists, axis=1)[:,0]
                all_min_dists[:,t] = min_dists <= props['threshold']
            elif ts.time > props['stop_ps']:
                break

        all_min_dists = all_min_dists.astype('int')
        on_off = all_min_dists[:,1:] - all_min_dists[:,:-1]
        current_states = np.zeros(n_atoms)
        for i in range(n_atoms):
            trans_ndx = np.nonzero(on_off[i,:])
            if list(trans_ndx[0]) == []:
                if all_min_dists[i,0]:
                    res_frames_av.append(n_frames-1)
                current_states[i] = 0
            else:
                first_transition = on_off[i,trans_ndx][0,0]
                if first_transition == -1:
                    current_states[i] = 1

        for i in range(n_frames-1):
            current_states += on_off[:,i]
            for j in range(n_atoms):
                if current_states[j] == 1:
                    res_frames[j] += 1
                elif current_states[j] == 0:
                    if res_frames[j] != 0:
                        res_frames_av.append(res_frames[j])
                        res_frames[j] = 0

        res_times = np.array(res_frames_av)*DT
        answer.append(res_times)
    return np.array(answer[0])

def write_res_times(res_times, props):
    f = open(NAME + "_restimes.sfu", 'w')
    values = []
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Residence times (ns) \n")
    for i in range(len(res_times)):
        f.write("{:<8.3f}\n".format(res_times[i] / 1000))#ps to ns
    f.close()

res_times = residence_time(props_residence_time)
write_res_times(res_times, props_residence_time)
