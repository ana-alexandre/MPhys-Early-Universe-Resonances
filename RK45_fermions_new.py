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
tmax = 20*2*np.pi/np.sqrt(k**2+mX**2)   #large time, equivalent to the +/- infinity limit
tosc_st = np.pi/m * (1/2 + 0)  
tosc = np.pi/m * (1/2 + 2)  #om2 oscillates between -tosc and +tosc (otherwise it is constant)
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
    ddup = -(om(k,t)**2 +np.complex(0,M(t)*dt_a(t) + dt_M(t)*a(t)))*up
    return np.array([dup,ddup])

#function to calculate derivatives for um
def d2um(t,um2):
    um = um2[0]
    dum = um2[1]
    ddum = -(om(k,t)**2 +np.complex(0,-M(t)*dt_a(t) - dt_M(t)*a(t)))*um
    return np.array([dum,ddum])

#create an array with all the points where I would like to save something
#t_eval = np.linspace(0,tmax,no_points)

#set initial conditions
up0 = np.complex(np.sqrt(1 - a(0)*M(0)/om(k,0)),0)
um0 = np.complex(np.sqrt(1 + a(0)*M(0)/om(k,0)),0)
dup0 = np.complex(0,k*um0 - a(0)*M(0)*up0)
dum0 = np.complex(0,k*up0 + a(0)*M(0)*um0)

#calculate solutions using RK45 method
sol_p = solve_ivp(d2up, [0,tmax], np.array([up0,dup0]), t_eval=t_eval, method='RK45',max_step=h_max)
sol_m = solve_ivp(d2um, [0,tmax], np.array([um0,dum0]), t_eval=t_eval, method='RK45',max_step=h_max)

#the time and functions
tps = sol_p.t
tms = sol_m.t
ups = sol_p.y
ums = sol_m.y

#transpose the final matrix to get the values of phi and dphi
[up,dup]=ups
[um,dum]=ums

###################################################
#########TEST######################################
#exact solutions for up,um,E,F,alpha and beta
up_ex = []
um_ex = []
for l in range(len(t_eval)):
    ti = t_eval[l]
    up_ex.append(np.sqrt(1 - mX/om(k,0))*np.exp(np.complex(0,om(k,0)*ti)))
    um_ex.append(np.sqrt(1 + mX/om(k,0))*np.exp(np.complex(0,om(k,0)*ti)))

up_ex = np.asarray(up_ex)
um_ex = np.asarray(um_ex)

######################################################
######################################################


#now plot the solutions
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,up, color='olive', linewidth=4)
plt.plot(tms,um, color='darkblue', linewidth=4)
#########TEST############
plt.plot(t_eval,up_ex,'--', color='brown', linewidth=4)
plt.plot(t_eval,um_ex,'--', color='teal', linewidth=4)
plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)',r'Re($u^{ex}_{+} (k)$)', r'Re($u^{ex}_{-} (k)$)'],loc='upper right')
########################
#plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_up_um_test.png')
plt.show()


#calculate complex conjugate of up and um
up_star = np.conj(up)
um_star = np.conj(um)


#now calculate E(k,t) and F(k,t) in the Hamiltonian
E = k*(up_star*um).real + a_fct*M_fct*(1 - up_star * up)
F = k * (up**2 - um**2)/2 + a_fct*M_fct*up*um
F_star = np.conj(F)


#check algth by seeing if E^2+F^2=omega^2
Esum = E**2 + F*F_star
Erat = Esum/(om_fct**2)

Erat_mean = np.mean(Erat)
Erat_std = np.std(Erat)

print('\n<E2 + F2/omega> = ', Erat_mean)
print('\nE2 + F2/omega^2 = ',Erat)


#calculate Bogoliubov coefficients
beta2 = F*F_star / (2*om_fct*(E + om_fct))
alpha2 = (om_fct + E)/(2*om_fct)

log_n = np.log(beta2)

print('\nbeta^2 = ',beta2)


#calculate the difference between alpha squared and beta squared
#which should equal 1
summ = alpha2 + beta2
summ_mean = np.mean(summ)
summ_std = np.std(summ)

print('\n<alpha2 + beta2> = ', summ_mean, ' +/- ', summ_std)
print('\nalpha2 + beta2 = ', summ)


#########################################################
#                          PLOTS
#########################################################

#now plot the solutions
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,up, color='olive', linewidth=4)
plt.plot(tms,um, color='darkblue', linewidth=4)
plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_up_um_test.png')
plt.show()

#plot up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,dup, color='olive', linewidth=4)
plt.plot(tms,dum,'--', color='darkblue', linewidth=4)
plt.legend([r'Re($\dot{u}_{+} (k)$)', r'Re($\dot{u}_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_dum_dup.png')
plt.show()

#plot imaginary part of up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,up.imag, color='olive', linewidth=4)
plt.plot(tms,um.imag,'--', color='darkblue', linewidth=4)
plt.legend([r'Im($u_{+} (k)$)', r'Im(${u}_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'im_up_dup.png')
plt.show()

#plot um and dum against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,dup.imag, color='olive', linewidth=4)
plt.plot(tms,dum.imag,'--', color='darkblue', linewidth=4)
plt.legend([r'Im($\dot{u}_{+} (k)$)', r'Im($\dot{u}_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'im_um_dum.png')
plt.show()

#plot omega
fig = plt.figure()

plt.plot(tps,Erat,color='teal', linewidth=2)
#plt.xlim(0,tosc_st)
plt.xlabel('t')
plt.ylabel(r'$\frac{E_k^2 + |F_k|^2}{\omega_k^2}$')
fig.savefig(root+'Erat.png')
plt.show()

#plot E and omega against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,E, color='darkblue', linewidth=4)
plt.plot(tps,F, color='teal', linewidth=4)
plt.plot(tps,om_fct,'--', color='olive', linewidth=4)
plt.legend([r'$E$',r'Re$(F)$', r'$\omega$'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'E_F_om.png')
plt.show()

#calculate the average of the final
where = np.argwhere(tps > tosc)[0][0]
n_f = beta2[where:]
n_f_mean = np.mean(n_f)
n_f_std = np.std(n_f)
print (r'Final $|\beta|^{2}$: ', n_f_mean, ' +/- ', n_f_std)


#plot the squares of beta against t
fig = plt.figure()

plt.plot(tps,beta2, color='teal', linewidth=2)
plt.xlabel('$\eta$')
plt.ylabel(r'$|\beta|^{2}$')
#plt.ylim(-1,1)
fig.savefig(root+'sq_beta.png')
plt.show()


#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(tps,log_n, color='darkblue', linewidth=4)
plt.xlabel(r'$\eta (m^{-1})$')
plt.ylabel(r'$\ln(N_k)$')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(0,23)
plt.tight_layout()
fig.savefig(root+'log_n_osc.png')
plt.show()

#########################
#save all the data to .txt files
np.savetxt(root + "time.txt",tps, delimiter=",")
np.savetxt(root + "up.txt",up, delimiter=",")
np.savetxt(root + "dup.txt",dup, delimiter=",")
np.savetxt(root + "um.txt",up, delimiter=",")
np.savetxt(root + "dum.txt",dup, delimiter=",")








