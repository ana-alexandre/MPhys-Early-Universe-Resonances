#new code to study the creations of fermions

#I will start just by solving the SHO equation for u+ and u-
#and slowly add on the rest of the calculations
# to really make sure everything is working

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#######################################
#define all the constants needed
mX = 100  #mass of fermion
gphi0 = 1000   #constant
m = 1 #frequency of inflaton oscillations (working in units of m = 1)
kmin = 1  #first momentum to calculate
kmax = 80 #m*gphi0 + 2*kmin  #number of particles created is calculated up  to a maximum value
osc_max = 7
osc_i = 0
osc_f = 6
tmax = np.pi/m * (1/2 + osc_max)   #large time, equivalent to the +/- infinity limit
tosc_st = np.pi/m * (1/2 + osc_i)  
tosc = np.pi/m * (1/2 + osc_f)  #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/mX**2  #step size adjusted to the largest frequency in the system
atol = 1e-8  #absolute tolerance
rtol = 1e-6  #relative tolerance
no_t_pts = tmax*10
no_k_pts = (kmax - kmin)*5
Mpl = 1e6  #Planck mass
#########################################

root = 'many_fermions_kmin'+str(kmin)+'_kmax'+str(kmax)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_no_osc'+str(osc_f - osc_i)+'_'

#########################################
###NOTE#################################
#I use t for time in this script but I am actually refering to conformal time eta
########################################

#define the scale factor a of the universe
def a(t):
    if t <= tmax:
        return 1  #for now we'll keep it constant
    else:
        print('ERROR: a is not defined for t=',t)

#define the a dot function
def dt_a(t):
    if t <= tmax:
        return 0  #for now we'll keep it constant
    else:
        print('ERROR: a\' is not defined for t=',t)

#define the inflaton field phi in time
def M(t):
    if (t >= tosc and t <= tmax) or t <= tosc_st:
        return mX
    elif t < tosc and t > tosc_st:
        return mX + gphi0 * np.cos(m*t)
    else:
        print(r'ERROR: $\phi$ is not defined for $t=$',t)

#define the omega dot function
def dt_M(t):
    if (t >= tosc and t <= tmax) or t <= tosc_st:
        return 0
    elif t < tosc and t > tosc_st:
        return - gphi0*m*np.sin(m*t)
    else:
        print(r'ERROR: $\dot \phi$ is not defined for $t=$',t)


#define the frequency omega in time
def om(k,t):
    if t <= tmax:
        return np.sqrt(k**2 + a(t)**2 * M(t)**2)
    else:
        print(r'ERROR: $\phi$ is not defined for $t=$',t)


#function to calculate derivatives for up
def d2up(t,up2):
    up = up2[0]
    dup = up2[1]
    ddup = -(om(k,t)**2 +np.complex(0,M(t)*dt_a(t) + dt_M(t)*a(t)))*up
    return np.array([dup,ddup])

#function to calculate derivatives for um
def d2um(t,um2):
    um = um2[0]
    dum = um2[1]
    ddum = -(om(k,t)**2 +np.complex(0,-M(t)*dt_a(t) - dt_M(t)*a(t)))*um
    return np.array([dum,ddum])


####################################
#create an array with all the points where I would like to save something
t_eval = np.linspace(0,tmax,num=no_t_pts)
#create an array with all the k values I want to loop over
k_eval = np.linspace(kmin,kmax,num=no_k_pts)

#now create arrays to store important variables
Erat_means = []
Erat_stds = []
Erat_f_means = []
Erat_f_stds = []
n_f_means = []
n_f_stds = []
ests_amp_n = []
summ_means = []
summ_stds = []
summ_f_means = []
summ_f_stds = []
##################################a



#calculate the scale factor and the inflaton field
#outside the loop since they are not k dependent
a_fct = []
M_fct = []
om_no_k = []
for j in range(len(t_eval)):
    t = t_eval[j]
    #calculate omega, a and m
    a_fct.append(a(t))
    M_fct.append(M(t))
    
a_fct = np.asarray(a_fct)
M_fct = np.asarray(M_fct)



#################################################
####    loop begins here    #####################
#################################################

for i in range(len(k_eval)):  #loop over k values

    #get k value
    k = k_eval[i]
   
    #increase atol only if k is high
    if k > gphi0:
        atol = 1e-10
 
    #calculate the frequency
    #using the already calculated a and M fcts
    om_fct = np.sqrt(k**2 + a_fct**2 * M_fct**2)

    #set initial conditions for each time it loops (k dependent)
    up0 = np.complex(np.sqrt(1 - a(0)*M(0)/om(k,0)),0)
    um0 = np.complex(np.sqrt(1 + a(0)*M(0)/om(k,0)),0)
    dup0 = np.complex(0,k*um0 - a(0)*M(0)*up0)
    dum0 = np.complex(0,k*up0 + a(0)*M(0)*um0)

    print('Hello World1')

    ##################################
    #calculate solutions using RK45 method
    #with variable step size
    sol_p = solve_ivp(d2up, [0,tmax], np.array([up0,dup0]), 
                      t_eval=t_eval, 
                      method='RK45',
                      max_step=h_max,
                      atol=atol,rtol=rtol)
   
    sol_m = solve_ivp(d2um, [0,tmax], np.array([um0,dum0]), 
                      t_eval=t_eval,
                      method='RK45',
                      max_step=h_max,
                      atol = atol, rtol=rtol)
    ###################################

    #the time and functions
    tps = sol_p.t
    tms = sol_m.t
    ups = sol_p.y
    ums = sol_m.y

    #transpose the final matrix to get the values of phi and dphi
    [up,dup]=ups
    [um,dum]=ums

    #calculate complex conjugate of up and um
    up_star = np.conj(up)
    um_star = np.conj(um)

    #now calculate E(k,t) and F(k,t) in the Hamiltonian
    E = k*(up_star*um).real + a_fct*M_fct*(1 - up_star * up)
    F = k * (up**2 - um**2)/2 + a_fct*M_fct*up*um
    F_star = np.conj(F)

    #find where the inflaton stops oscillating
    where = np.argwhere(tps > tosc)[0][0]

    #check algth by seeing if E^2+F^2=omega^2
    Esum = E**2 + F*F_star
    Erat = Esum/(om_fct**2)

    Erat_means.append(np.mean(Erat))
    Erat_stds.append(np.std(Erat))

    Erat_f = Erat[where:]
    Erat_f_means.append(np.mean(Erat_f))
    Erat_f_stds.append(np.std(Erat_f))

    #calculate Bogoliubov coefficients
    beta2 = F*F_star / (2*om_fct*(E + om_fct))
    alpha2 = (om_fct + E)/(2*om_fct)

    #calculate the average of the final beta squared
    n_f = beta2[where:]
    n_f_means.append(np.mean(n_f))
    n_f_stds.append(np.std(n_f))
    ests_amp_n.append(3*np.std(n_f)/np.sqrt(2))

    #calculate the difference between alpha squared and beta squared
    #which should equal 1
    summ = alpha2 + beta2
    summ_means.append(np.mean(summ))
    summ_stds.append(np.std(summ))

    summ_f = summ[where:]
    summ_f_means.append(np.mean(summ_f))
    summ_f_stds.append(np.std(summ_f))

    print(i,': k: ',k)
    i = int(i)
    print(': \t<E2 + F2/omega> = ', Erat_means[i], ' +/- ', Erat_stds[i],'\tFinal <E2 + F2/omega> = ', Erat_f_means[i], ' +/- ', Erat_f_stds[i])
    print ('\n Final <n>: ', n_f_means[i], ' +/- ', n_f_stds[i],'\t',r'Amplitude of final n oscillations: ',ests_amp_n[i])
    print('\n<alpha2 + beta2> = ', summ_means[i], ' +/- ', summ_stds[i],'\tFinal <alpha2 + beta2> = ', summ_f_means[i], ' +/- ', summ_f_stds[i])


#########################################################
#                          PLOTS
#########################################################

#plot of final n with error bars
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(k_eval, n_f_means, yerr = ests_amp_n, fmt = 'o', color='olive', capsize=0.5, linestyle = 'none')
plt.xlabel(r'$k$')
plt.ylabel(r'$|\beta|^{2}$')
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'n.png')
plt.show()

log_n = np.log(np.asarray(n_f_means))
log_n_err = np.asarray(ests_amp_n)/np.asarray(n_f_means)

#plot of final log(n) with error bars
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(k_eval, log_n, yerr = log_n_err, fmt = 'o', color='olive', capsize=0.5, linestyle = 'none')
plt.xlabel(r'$k$')
plt.ylabel(r'$\log(|\beta|^{2})$')
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'logn.png')
plt.show()

#plot of Erat and summ with error bars
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(k_eval, Erat_means, yerr = Erat_stds, fmt = 'o', color='olive', capsize=0.5, linestyle = 'none')
plt.errorbar(k_eval, summ_means, yerr = summ_stds, fmt = '^', color='darkblue', capsize=0.5, linestyle = 'none')
plt.xlabel(r'$k$')
plt.legend([r'$\frac{E_k^2 + |F_k|^2}{\omega_k^2}$',r'$|\alpha|^{2} + |\beta|^{2}$'])
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'Erat_summ.png')
plt.show()

#calculate errors for Erat and summ
Erat_err = 3*np.asarray(Erat_f_stds)/np.sqrt(2)
summ_err = 3*np.asarray(summ_f_stds)/np.sqrt(2)

#plot of final Erat and summ with error bars
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(k_eval, Erat_f_means, yerr = Erat_err, fmt = 'o', color='olive', capsize=0.5, linestyle = 'none')
plt.errorbar(k_eval, summ_f_means, yerr = summ_err, fmt = '^', color='darkblue', capsize=0.5, linestyle = 'none')
plt.xlabel(r'$k$')
plt.legend([r'$\frac{E_k^2 + |F_k|^2}{\omega_k^2}$',r'$|\alpha|^{2} + |\beta|^{2}$'])
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'final_Erat_summ.png')
plt.show()


################################################
#save all the data to .txt files
np.savetxt(root + "k.txt",k_eval, delimiter=",")
np.savetxt(root + "n.txt",n_f_means, delimiter=",")
np.savetxt(root + "n_std.txt",n_f_stds, delimiter=",")
np.savetxt(root + "Erat.txt",Erat_means, delimiter=",")
np.savetxt(root + "Erat_std.txt",Erat_stds, delimiter=",")
np.savetxt(root + "Erat_f.txt",Erat_f_means, delimiter=",")
np.savetxt(root + "Erat_f_std.txt",Erat_f_stds, delimiter=",")
np.savetxt(root + "summ.txt",summ_means, delimiter=",")
np.savetxt(root + "summ_std.txt",summ_stds, delimiter=",")
np.savetxt(root + "summ_f.txt",summ_f_means, delimiter=",")
np.savetxt(root + "summ_f_std.txt",summ_f_stds, delimiter=",")







