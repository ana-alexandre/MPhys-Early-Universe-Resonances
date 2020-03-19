#ploting the results for many fermions
#plotting different variables against k
#and calculating errors more carefully

import os,sys
import numpy as np
import matplotlib.pyplot as plt

##change the font size for pyplot
font = {'family' : 'normal',
        'size'   : 30}

plt.rc('font', **font)

plt.rcParams.update({'lines.markeredgewidth': 0})

######################################
#          READ 1
#######################################
#define all the constants needed
mX = 100  #mass of fermion
gphi0 = 1000   #constant
m = 1 #frequency of inflaton oscillations (working in units of m = 1)
kmin = 5.1  #first momentum to calculate
kmax = 42.5 #m*gphi0 + 2*kmin  #number of particles created is calculated up  to a maximum value
osc_max = 7
osc_i = 0
osc_f = 6
#########################################


root = 'many_fermions_kmin'+str(kmin)+'_kmax'+str(kmax)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_no_osc'+str(osc_f - osc_i)+'_'


#now first import all the simulated data
k_eval = np.loadtxt(root + "k.txt",dtype=np.complex_)
n = np.loadtxt(root + "n.txt",dtype=np.complex_)
n_std = np.loadtxt(root + "n_std.txt",dtype=np.complex_)
Erat = np.loadtxt(root + "Erat.txt",dtype=np.complex_)
Erat_std = np.loadtxt(root + "Erat_std.txt",dtype=np.complex_)
Erat_f_std = np.loadtxt(root + "Erat_f_std.txt",dtype=np.complex_)
Erat_f = np.loadtxt(root + "Erat_f.txt",dtype=np.complex_)
summ = np.loadtxt(root + "summ.txt",dtype=np.complex_)
summ_std = np.loadtxt(root + "summ_std.txt",dtype=np.complex_)
summ_f = np.loadtxt(root + "summ_f.txt",dtype=np.complex_)
summ_f_std = np.loadtxt(root + "summ_f_std.txt",dtype=np.complex_)

#now use Erat and summ to get error on n
Erat_err = 3*Erat_f_std/np.sqrt(2)
where = len(n) - len(Erat_f)
Erat_sigma = 4 * np.sqrt(n[where:]) * (summ_f - n[where:]) * Erat_err

summ_sigma = 3*summ_f_std/np.sqrt(2)

n_err = np.maximum(Erat_sigma,summ_sigma)

######################################
#            READ 2
#######################################
#define all the constants needed
mX = 100  #mass of fermion
gphi0 = 1000   #constant
m = 1 #frequency of inflaton oscillations (working in units of m = 1)
kmin = 1  #first momentum to calculate
kmax = 80 #m*gphi0 + 2*kmin  #number of particles created is calculated up  to a maximum value
tosc_st = np.pi/m * (1/2 + 0)
tosc = np.pi/m * (1/2 + 6)
#########################################


root = 'many_fermions_kmin'+str(kmin)+'_kmax'+str(kmax)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_tosc'+str(tosc)+'_'+'_tosc_st'+str(tosc_st)+'_'


#now first import all the simulated data
k_eval2 = np.loadtxt(root + "k.txt",dtype=np.complex_)
n2 = np.loadtxt(root + "n.txt",dtype=np.complex_)
n_std2 = np.loadtxt(root + "n_std.txt",dtype=np.complex_)
Erat2 = np.loadtxt(root + "Erat.txt",dtype=np.complex_)
Erat_std2 = np.loadtxt(root + "Erat_std.txt",dtype=np.complex_)
Erat_f_std2 = np.loadtxt(root + "Erat_f_std.txt",dtype=np.complex_)
Erat_f2 = np.loadtxt(root + "Erat_f.txt",dtype=np.complex_)
summ2 = np.loadtxt(root + "summ.txt",dtype=np.complex_)
summ_std2 = np.loadtxt(root + "summ_std.txt",dtype=np.complex_)
summ_f2 = np.loadtxt(root + "summ_f.txt",dtype=np.complex_)
summ_f_std2 = np.loadtxt(root + "summ_f_std.txt",dtype=np.complex_)

#now use Erat and summ to get error on n
Erat_err2 = 3*Erat_f_std2/np.sqrt(2)
where2 = len(n2) - len(Erat_f2)
Erat_sigma2 = 4 * np.sqrt(n2[where2:]) * (summ_f2 - n2[where2:]) * Erat_err2

summ_sigma2 = 3*summ_f_std2/np.sqrt(2)

n_err2 = np.maximum(Erat_sigma2,summ_sigma2)

######################################
#          READ 3
#######################################
#define all the constants needed
mX = 100  #mass of fermion
gphi0 = 1000   #constant
m = 1 #frequency of inflaton oscillations (working in units of m = 1)
kmin = 1  #first momentum to calculate
kmax = 80 #m*gphi0 + 2*kmin  #number of particles created is calculated up  to a maximum value
osc_i = 0
osc_f = 12
#########################################


root = 'many_fermions_kmin'+str(kmin)+'_kmax'+str(kmax)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_no_osc'+str(osc_f - osc_i)+'_'


#now first import all the simulated data
k_eval3 = np.loadtxt(root + "k.txt",dtype=np.complex_)
n3 = np.loadtxt(root + "n.txt",dtype=np.complex_)
n_std3 = np.loadtxt(root + "n_std.txt",dtype=np.complex_)
Erat3 = np.loadtxt(root + "Erat.txt",dtype=np.complex_)
Erat_std3 = np.loadtxt(root + "Erat_std.txt",dtype=np.complex_)
Erat_f_std3 = np.loadtxt(root + "Erat_f_std.txt",dtype=np.complex_)
Erat_f3 = np.loadtxt(root + "Erat_f.txt",dtype=np.complex_)
summ3 = np.loadtxt(root + "summ.txt",dtype=np.complex_)
summ_std3 = np.loadtxt(root + "summ_std.txt",dtype=np.complex_)
summ_f3 = np.loadtxt(root + "summ_f.txt",dtype=np.complex_)
summ_f_std3 = np.loadtxt(root + "summ_f_std.txt",dtype=np.complex_)

#now use Erat and summ to get error on n
Erat_err3 = 3*Erat_f_std3/np.sqrt(2)
where3 = len(n3) - len(Erat_f3)
Erat_sigma3 = 4 * np.sqrt(n3[where3:]) * (summ_f3 - n3[where3:]) * Erat_err3

summ_sigma3 = 3*summ_f_std3/np.sqrt(2)

n_err3 = np.maximum(Erat_sigma3,summ_sigma3)


######################################
#          READ 4
#######################################
#define all the constants needed
mX = 100  #mass of fermion
gphi0 = 1000   #constant
m = 1 #frequency of inflaton oscillations (working in units of m = 1)
kmin = 1.125  #first momentum to calculate
kmax = 50.125 #m*gphi0 + 2*kmin  #number of particles created is calculated up  to a maximum value
osc_i = 0
osc_f = 12
#########################################


root = 'many_fermions_kmin'+str(kmin)+'_kmax'+str(kmax)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_no_osc'+str(osc_f - osc_i)+'_'


#now first import all the simulated data
k_eval4 = np.loadtxt(root + "k.txt",dtype=np.complex_)
n4 = np.loadtxt(root + "n.txt",dtype=np.complex_)
n_std4 = np.loadtxt(root + "n_std.txt",dtype=np.complex_)
Erat4 = np.loadtxt(root + "Erat.txt",dtype=np.complex_)
Erat_std4 = np.loadtxt(root + "Erat_std.txt",dtype=np.complex_)
Erat_f_std4 = np.loadtxt(root + "Erat_f_std.txt",dtype=np.complex_)
Erat_f4 = np.loadtxt(root + "Erat_f.txt",dtype=np.complex_)
summ4 = np.loadtxt(root + "summ.txt",dtype=np.complex_)
summ_std4 = np.loadtxt(root + "summ_std.txt",dtype=np.complex_)
summ_f4 = np.loadtxt(root + "summ_f.txt",dtype=np.complex_)
summ_f_std4 = np.loadtxt(root + "summ_f_std.txt",dtype=np.complex_)

#now use Erat and summ to get error on n
Erat_err4 = 3*Erat_f_std4/np.sqrt(2)
where4 = len(n4) - len(Erat_f4)
Erat_sigma4 = 4 * np.sqrt(n4[where4:]) * (summ_f4 - n4[where4:]) * Erat_err4

summ_sigma4 = 3*summ_f_std4/np.sqrt(2)

n_err4 = np.maximum(Erat_sigma4,summ_sigma4)






#define the root to save the files to
root = 'many_fermions_compiled_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_no_osc'+str(osc_f - osc_i)+'_'

#########################################################
#                        PLOTS
#########################################################

#plot of final n with error bars
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(k_eval, n, yerr = 100*n_err, fmt = 'o', color='olive', capsize=0.3, linestyle = '-', linewidth = 0.8, label = '6 oscillations',elinewidth = 0.4)
plt.errorbar(k_eval2, n2, yerr = 100*n_err2, fmt = 'o', color='olive', capsize=0.3, linestyle = 'none', label = None,elinewidth = 0.4)
plt.errorbar(k_eval3, n3, yerr = 100*n_err3, fmt = '<', color='darkblue', capsize=0.3, linestyle = '-', linewidth = 0.8, label = '12 oscillations',elinewidth = 0.4)
plt.errorbar(k_eval4, n4, yerr = 100*n_err4, fmt = '<', color='darkblue', capsize=0.3, linestyle = 'none', label = None,elinewidth = 0.4)
plt.xlabel(r'$k$')
plt.ylabel(r'$|\beta|^{2}$')
plt.legend()
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'n.png')
plt.show()













