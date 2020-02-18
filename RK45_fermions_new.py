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
k = 100  #momentum of fourier mode
mX = 10  #mass of fermion
gphi0 = 100   #constant
m = 10 #frequency of inflaton oscillations
tmax = 40*2*np.pi/np.sqrt(k**2+mX**2)   #large time, equivalent to the +/- infinity limit
tosc_st = np.pi/m * (1/2 + 1)  
tosc = np.pi/m * (1/2 + 4)  #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/(100*max([k,mX,m]))  #step size adjusted to the largest frequency in the system
no_points = tmax/h_max
#########################################

root = 'fermion_k'+str(k)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_m'+str(m)+'_tosc'+str(tosc)+'_'+'_tosc_st'+str(tosc_st)+'_'


#define the scale factor a of the universe
def a(t):
    if t <= tmax:
        return 1  #for now we'll keep it constant
    else:
        print('ERROR: a is not defined for t=',t)

#define the a dot function
def dt_a(t):
    if t <= tmax:
        return 0
    else:
        print('ERROR: ',r'$\dot a$','is not defined for t=',t)


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

#plot the frequency
om_fct = []
a_fct = []
M_fct = []
t_eval = np.linspace(0,tmax,no_points)
for j in range(len(t_eval)):
    i = t_eval[j]
    #calculate omega, a and m
    om_fct.append(om(k,i))
    a_fct.append(a(i))
    M_fct.append(M(i))

om_fct = np.asarray(om_fct)
a_fct = np.asarray(a_fct)
M_fct = np.asarray(M_fct)

#plot M over t_eval
fig = plt.figure()

plt.plot(t_eval,M_fct,color='teal', linewidth=1)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$M$')
#fig.savefig(root+'m.png')
plt.show()

#plot omega over t_eval
fig = plt.figure()

plt.plot(t_eval,om_fct,color='teal', linewidth=1)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\omega$')
#fig.savefig(root+'m.png')
plt.show()


#function to calculate derivatives for up
def d2up(t,up2):
    up = up2[0]
    dup = up2[1]
    ddup = -om(k,t) * up  #######################
    return np.array([dup,ddup])

#function to calculate derivatives for um
def d2um(t,um2):
    um = um2[0]
    dum = um2[1]
    ddum = -om(k,t) * um ########################
    return np.array([dum,ddum])

#create an array with all the points where I would like to save something
#t_eval = np.linspace(0,tmax,no_points)

#set initial conditions
up0 = np.complex(np.sqrt(1 - a(0)*M(0)/om(k,0)),0)
um0 = np.complex(np.sqrt(1 + a(0)*M(0)/om(k,0)),0)
dup0 = np.complex(0,k*um0 - a(0)*M(0)*up0)
dum0 = np.complex(0,k*up0 + a(0)*M(0)*um0)

#calculate solutions using RK45 method
sol_p = solve_ivp(d2up, [0,tmax], np.array([up0,dup0]), t_eval=t_eval, method='RK45',max_step=h_max) ############# tmax
sol_m = solve_ivp(d2um, [0,tmax], np.array([um0,dum0]), t_eval=t_eval, method='RK45',max_step=h_max)  ############# tmax

#the time and functions
tps = sol_p.t
tms = sol_m.t
ups = sol_p.y
ums = sol_m.y

#transpose the final matrix to get the values of phi and dphi
[up,dup]=ups
[um,dum]=ums

#now plot the solutions
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,up, color='olive', linewidth=4)
plt.plot(tms,um, color='darkblue', linewidth=4)
#########TEST############
#plt.plot(t_eval,up_ex,'--', color='brown', linewidth=4)
#plt.plot(t_eval,um_ex,'--', color='teal', linewidth=4)
#plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)',r'Re($u^{ex}_{+} (k)$)', r'Re($u^{ex}_{-} (k)$)'],loc='upper right')
########################
#plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
#fig.savefig(root+'re_up_um.png')
plt.show()


#calculate complex conjugate of up and um
up_star = np.conj(up)
um_star = np.conj(um)

















