XTC = "T1-Nnnn-Iiii_L_PRO1_FIX.xtc"
TPR = "T1-Nnnn-Iiii_L_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from Extras import *
from scipy.optimize import curve_fit
plt.rcParams["font.family"] = "Times New Roman"

props_potential = {
'charge_file'   : NAME+"_charge.sfu",
'charge_keys'   : ["NP", "NA", "CL", "SOL"], #The name of the columns in charge_file. Order matters. Last one will be read as Total
'rdf_file'      : NAME+"_rdf.sfu",
'rdf_keys'      : ["NA", "CL", "SOL"], #The name of the columns in rdf_file. Order matters
'counter_key'   : ["NA"], #Key of the counterion's rdf
'rdf_tolerance' : 0.1 #Criterium to consider rdf a mismatch from debye-huckel fit
}

def electric_field(props):
    charge_data = read_text_file(props['charge_file'])
    space = charge_data[:,0]
    field = charge_data[:,1:] #exclude space column
    field = np.divide(field.T, np.power(space, 2)).T*sci.e*10**9 #Converts charge from e to C and one nm to m
    field = field/(4*sci.pi*sci.epsilon_0) #V/nm
    return space, field

def electric_potential(props):
    space, fields = electric_field(props) #nm, V/nm
    dr = space[1] - space[0]
    pots = fields+0
    for f, field in enumerate(fields.T):
        for i in range(len(field)):
            pots[i,f] = -np.trapz(y = field[:i], x = space[:i], dx = dr) #V
    pots = pots - pots[-1]
    return space, pots

def debye_huckel(x, A, B, C):
    return A*np.divide(np.exp(-B*x), x) + C

def electrophoretic_radius(props):
    rdf_data = read_text_file(props['rdf_file'])
    space = rdf_data[:,0]
    ndx_counter = np.where(np.array(props['rdf_keys']) == props['counter_key'])[0][0]
    rdf = rdf_data[:,ndx_counter+1] #+1 to ignore space column
    ndx_maxrdf = np.argmax(rdf)

    ndx_maxfit = int(0.95 * len(space))
    space_fit = space[ndx_maxrdf:ndx_maxfit]
    rdf_fit = rdf[ndx_maxrdf:ndx_maxfit]
    rmse = []
    for i in range(len(space_fit)):
        crop_space = space_fit[i:]
        crop_rdf = rdf_fit[i:]
        try:
            opt_params, cov = curve_fit(debye_huckel, xdata = crop_space, ydata = crop_rdf)
            error = root_mean_squared_error(crop_rdf, debye_huckel(crop_space, *opt_params))
        except:
            error = 0.1 #empirical random value
        rmse.append(error)
    rmse = np.array(rmse)
    rmse[np.isinf(rmse)] = 1
    ndx_best = local_minima(space_fit, rmse)

    f = open(NAME+"_shear.sfu", "w")
    #fig = plt.figure()
    #ax = plt.axes()
    #ax.set_ylim((min(rdf), max(rdf)))
    f.write("{:<8} {:<8} {:<8} {:<8}\n".format("Fit start", "RMSE", "Debye L", "Shear plane"))
    for ndx in ndx_best:
        opt_params, cov = curve_fit(debye_huckel, xdata = space_fit[ndx:], ydata = rdf_fit[ndx:])
       # ax.plot(space, debye_huckel(space, *opt_params), '--', label = "{:.2f}".format(1/opt_params[1]))
        diff = np.abs(rdf - debye_huckel(space, *opt_params))
        ndx_shear = np.where(diff > props['rdf_tolerance'])[0][-1]
        f.write("{:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}\n".format(space_fit[ndx], rmse[ndx], 1/opt_params[1], space[ndx_shear]))
    #ax.plot(space, rdf, '-k')
    #plt.legend()
    #plt.show()
    f.close()

def write_field(space, field_array, properties):
    f = open(NAME + "_field.sfu", 'w')
    f.write("#Electric field in (V/nm)\n")
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key in properties['charge_keys']:
        f.write("{:<9} ".format(key))
    f.write("TOTAL   ")
    f.write("\n")

    for i in range(len(space)):
        f.write("{:<9.4f} ".format(space[i])) #space is already imported as nm
        for j in range(len(field_array[0,:])):
            f.write("{:>9.4f} ".format(field_array[i,j]))
        f.write("\n")
    f.close()

def write_potential(space, pot_array, properties):
    f = open(NAME + "_pot.sfu", 'w')
    f.write("#Electric potential in (V)\n")
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key in properties['charge_keys']:
        f.write("{:<9} ".format(key))
    f.write("TOTAL   ")
    f.write("\n")

    for i in range(len(space)):
        f.write("{:<9.4f} ".format(space[i])) #space is already imported as nm
        for j in range(len(pot_array[0,:])):
            f.write("{:>9.4f} ".format(pot_array[i,j]))
        f.write("\n")
    f.close()

r, fields = electric_field(props_potential)
write_field(r, fields, props_potential)
r, potentials = electric_potential(props_potential)
write_potential(r, potentials, props_potential)
electrophoretic_radius(props_potential)

"""fig = plt.figure()
ax = plt.axes()
for f, field in enumerate(fields.T[:-1]):
    ax.plot(r, field, label = props_potential['charge_keys'][f])
ax.plot(r, fields[:,-1], label = "Total")
plt.legend()
plt.show()"""
"""
fig = plt.figure()
ax = plt.axes()
for f, potential in enumerate(potentials.T[:-1]):
    ax.plot(r, potential, label = props_potential['charge_keys'][f])
ax.plot(r, potentials[:,-1], label = "Total")
plt.legend()
plt.show()
fig = plt.figure()
ax = plt.axes()
ax.plot(r, potentials[:,-1], label = "Total")
plt.legend()
plt.show()"""
