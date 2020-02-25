#:Creation of Fermionic Fourier modes
#solves the differential equations of motion
#using the rk5(4) method
#calculates Bogoliubov coefficients
#averages over many k values

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import arange

##change the font size for pyplot
font = {'family' : 'normal',
        'size'   : 15}

plt.rc('font', **font)


#######################################
#define all the constants needed
k = 100  #momentum of fourier mode
mX = 10  #mass of fermion
gphi0 = 100   #constant
Om = 1 #frequency of inflaton oscillations
tmax = 60   #large time, equivalent to the +/- infinity limit
tosc_st = 6*np.pi/Om
tosc = 12*np.pi/Om   #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/(10*Om)  #step size
number_points = 8*tmax
#########################################

#########################################
###NOTE#################################
#I use t for time in this script but I am actually refering to conformal time eta
########################################


root = 'fermion_k'+str(k)+'_mX'+str(mX)+'_gphi0'+str(gphi0)+'_Om'+str(Om)+'_tosc'+str(tosc)+'_'


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
def m(t):
    if (t >= tosc and t <= tmax) or t <= tosc_st:
        return mX
    elif t < tosc and t > tosc_st:
        return mX + gphi0 * np.cos(Om*t)
    else:
        print(r'ERROR: $\phi$ is not defined for $t=$',t)

#define the omega dot function
def dt_m(t):
    if (t >= tosc and t <= tmax) or t <= tosc_st:
        return 0
    elif t < tosc and t > tosc_st:
        return - gphi0*Om*np.sin(Om*t)
    else:
        print(r'ERROR: $\dot \phi$ is not defined for $t=$',t)


#define the frequency omega in time
def om(k,t):
    if t <= tmax:
        return np.sqrt(k**2 + a(t)**2 * m(t)**2)
    else:
        print(r'ERROR: $\phi$ is not defined for $t=$',t)


#plot the frequency
om_fct = []
a_fct = []
m_fct = []
t_eval = np.linspace(0,tmax,number_points)
for j in range(len(t_eval)):
    i = t_eval[j]
    #calculate omega, a and m
    om_fct.append(om(k,i))
    a_fct.append(a(i))
    m_fct.append(m(i))

om_fct = np.asarray(om_fct)
a_fct = np.asarray(a_fct)
m_fct = np.asarray(m_fct)


###################################################
#########TEST######################################
#exact solutions for up,um,E,F,alpha and beta
up_ex = []
um_ex = []
up_ex_star = []
um_ex_star = []
for l in range(len(t_eval)):
    ti = t_eval[l]
    up_ex.append(np.sqrt(1 - mX/om(k,ti))*np.exp(np.complex(0,om(k,ti)*ti)))
    um_ex.append(np.sqrt(1 + mX/om(k,ti))*np.exp(np.complex(0,om(k,ti)*ti)))

up_ex = np.asarray(up_ex)
um_ex = np.asarray(um_ex)
up_ex_star = np.conj(up_ex)
um_ex_star = np.conj(um_ex)

E_ex = k*(up_ex_star*um_ex).real + mX*(1 - up_ex_star * up_ex)
F_ex = k * (up_ex**2 - um_ex**2)/2 + mX*up_ex*um_ex
F_ex_star = np.conj(F_ex)

#beta2_ex = (om(k,0) - E_ex)/(2*om(k,0)) 
beta2_ex = F_ex*F_ex_star / (2*om(k,0)*(E_ex + om(k,0)))
#alpha2_ex = beta2 * ((E_ex + om(k,0))**2)/(F_ex*F_ex_star)
alpha2_ex = (om(k,0) + E_ex)/(2*om(k,0))

######################################################
######################################################


#plot omega
fig = plt.figure()

plt.plot(t_eval,om_fct,color='teal', linewidth=1)
plt.xlabel('t')
plt.ylabel(r'$\omega$')
fig.savefig(root+'om.png')
plt.show()

#plot m
fig = plt.figure()

plt.plot(t_eval,m_fct,color='teal', linewidth=1)
plt.xlabel('t')
plt.ylabel(r'$m$')
fig.savefig(root+'m.png')
plt.show()

##########################
#set initial conditions
up0 = np.complex(np.sqrt(1 - a(0)*m(0)/om(k,0)),0)
um0 = np.complex(np.sqrt(1 + a(0)*m(0)/om(k,0)),0)
dup0 = np.complex(0,k*um0 - a(0)*m(0)*up0)
dum0 = np.complex(0,k*up0 + a(0)*m(0)*um0)
###########################

print('ICs: um: ', up0,', ',dup0, '\t up:', um0,', ',dum0)

#function to calculate derivatives for up
def d2up(t,up2):
    up = up2[0]
    dup = up2[1]
    ddup = -(om(k,t)**2 +np.complex(0,m(t)*dt_a(t) + dt_m(t)*a(t)))*up
    return np.array([dup,ddup])

#function to calculate derivatives for um
def d2um(t,um2):
    um = um2[0]
    dum = um2[1]
    ddum = -(om(k,t)**2 +np.complex(0,-m(t)*dt_a(t) - dt_m(t)*a(t)))*um
    return np.array([dum,ddum])


##############################################
##################TEST########################
#plot the frequency
rhs_p = []
rhs_m = []
for j in range(len(t_eval)):
    t = t_eval[j]
    rhs_p.append(-(om(k,t)**2 +np.complex(0,m(t)*dt_a(t) + dt_m(t)*a(t))))
    rhs_m.append(-(om(k,t)**2 +np.complex(0,-m(t)*dt_a(t) - dt_m(t)*a(t))))

rhs_p = np.asarray(rhs_p)
rhs_m = np.asarray(rhs_m)
##############################################
##############################################


#assign initial values
up_ini = np.array([up0,dup0])  # starting value for up and dtupi
ups = np.array([up_ini])     # and all solution points
um_ini = np.array([um0,dum0])  # starting value for um and dtumi
ums = np.array([um_ini])     # and all solution points
#betas2 = np.array([0])
#alphas = np.array([1])


#calculate up and um separately with rk4 method
sol_p = solve_ivp(d2up, [0, tmax], up_ini, method='RK45',t_eval=t_eval, max_step = h_max)
sol_m = solve_ivp(d2um, [0, tmax], um_ini, method='RK45',t_eval=t_eval, max_step = h_max)

#the time and functions
ts = sol_p.t
ups = sol_p.y
ums = sol_m.y

#transpose the final matrix to get the values of phi and dphi
[up,dup]=ups
[um,dum]=ums

#calculate complex conjugate of up and um
up_star = np.conj(up)
um_star = np.conj(um)

#now calculate E(k,t) and F(k,t) in the Hamiltonian
E = k*(up_star*um).real + a_fct*m_fct*(1 - up_star * up)
F = k * (up**2 - um**2)/2 + a_fct*m_fct*up*um
F_star = np.conj(F)

print('\nE/omega (eta0) = ', E[0]/om_fct[0], '\tF(eta0) = ', F[0])

print('\nE/omega (eta) = ', E/om_fct, '\tF(eta) = ', F)

############TEST#################
print('\nE/omega (eta) = ', E_ex/om(k,0), '\tF_ex(eta) = ', F_ex)
################################


#check algth by seeing if E^2+F^2=omega^2
Esum = E**2 + F*F_star
Erat = Esum/(om_fct**2)

Erat_mean = np.mean(Erat)
Erat_std = np.std(Erat)

print('\nE2 + F2/omega^2 (eta0) = ',Erat[0],'\t<E2 + F2/omega> = ', Erat_mean)
print('\nE2 + F2/omega^2 = ',Erat)


############TEST#################
Esum_ex = E_ex**2 + F_ex*F_ex_star
Erat_ex = Esum_ex/(om(k,0)**2)

print('\nE2 + F2/omega^2 ex = ',Erat_ex)

################################


#calculate Bogoliubov coefficients
#beta2 = (om_fct - E)/(2*om_fct) 
beta2 = F*F_star / (2*om_fct*(E + om_fct))
#alpha2 = beta2 * ((E + om_fct)**2)/(F*F_star)
alpha2 = (om_fct + E)/(2*om_fct)

log_n = np.log(beta2)

#alpha2[0] = np.complex(1,0)

#print('\nalpha2 = ',alpha2,'\nbeta2 = ',beta2)


#calculate the difference between alpha squared and beta squared
#which should equal 1
summ = alpha2 + beta2
summ_mean = np.mean(summ)
summ_std = np.std(summ)

print('\nalpha2 + beta2 = ', summ)
print('\nalpha2 + beta2 (eta0) = ',summ[0],'\t<alpha2 + beta2> = ', summ_mean, ' +/- ', summ_std)


########TEST############
summ_ex = alpha2_ex + beta2_ex
print('\nalpha2 + beta2 (exact) = ', summ_ex)
#########################


##########################################
################ TEST ####################
#plot up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,rhs_p*up, color='darkblue', linewidth=4)
plt.plot(ts,rhs_m*um, color='red', linewidth=4)
plt.plot(ts,-om_fct**2*up,'--',color='olive', linewidth=4)
plt.plot(ts,-om_fct**2*um,'--', color='orange', linewidth=4)
plt.legend([r'$\partial_{\eta}^2 u_{+}$', r'$\partial_{\eta}^2 u_{-}$',r'$\omega_k^2 u_{+}$',r'$\omega_k^2 u_{-}$'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'test_rhs.png')
plt.show()

##########################################
##########################################

#plot up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,up, color='olive', linewidth=4)
plt.plot(ts,dup, color='darkblue', linewidth=4)
plt.legend([r'Re($u_{+} (k)$)', r'Re($\dot{u}_{+} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_up_dup.png')
plt.show()

#plot um and dum against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,um, color='olive', linewidth=4)
plt.plot(ts,dum,'--', color='darkblue', linewidth=4)
plt.legend([r'Re($u_{-} (k)$)', r'Re($\dot{u}_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_um_dum.png')
plt.show()

#plot up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,up, color='olive', linewidth=4)
plt.plot(ts,um, color='darkblue', linewidth=4)
#########TEST############
plt.plot(t_eval,up_ex,'--', color='brown', linewidth=4)
plt.plot(t_eval,um_ex,'--', color='teal', linewidth=4)
plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)',r'Re($u^{ex}_{+} (k)$)', r'Re($u^{ex}_{-} (k)$)'],loc='upper right')
########################
#plt.legend([r'Re($u_{+} (k)$)', r'Re($u_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_up_um.png')
plt.show()

#plot up and dup against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,dup, color='olive', linewidth=4)
plt.plot(ts,dum,'--', color='darkblue', linewidth=4)
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

plt.plot(ts,up.imag, color='olive', linewidth=4)
plt.plot(ts,dup.imag,'--', color='darkblue', linewidth=4)
plt.legend([r'Im($u_{+} (k)$)', r'Im($\dot{u}_{+} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'im_up_dup.png')
plt.show()

#plot um and dum against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,um.imag, color='olive', linewidth=4)
plt.plot(ts,dum.imag,'--', color='darkblue', linewidth=4)
plt.legend([r'Im($u_{-} (k)$)', r'Im($\dot{u}_{-} (k)$)'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'im_um_dum.png')
plt.show()

#plot omega
fig = plt.figure()

plt.plot(ts,Erat,color='teal', linewidth=2)
#########TEST############
plt.plot(ts,Erat_ex,'--', color='olive', linewidth=4)
########################
plt.xlim(0,tosc_st)
plt.xlabel('t')
plt.ylabel(r'$\frac{E_k^2 + |F_k|^2}{\omega_k^2}$')
fig.savefig(root+'Erat.png')
plt.show()

#plot E and omega against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,E, color='darkblue', linewidth=4)
plt.plot(ts,F, color='teal', linewidth=4)
plt.plot(ts,om_fct, color='olive', linewidth=4)
#########TEST############
plt.plot(ts,E_ex,'--', color='brown', linewidth=4)
plt.plot(ts,F_ex,'--', color='orange', linewidth=4)
plt.legend([r'$E$',r'Re$(F)$', r'$\omega$',r'$E^{ex}$',r'Re$(F^{ex})$'],loc='upper right')
########################
#plt.legend([r'$E$',r'Re$(F)$', r'$\omega$'],loc='upper right')
plt.xlabel(r'$\eta (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
plt.xlim(0,tosc_st)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'E_F_om.png')
plt.show()

#calculate the average of the final
where = np.argwhere(ts > tosc)[0][0]
n_f = beta2[where:]
n_f_mean = np.mean(n_f)
n_f_std = np.std(n_f)
print (r'Final $|\beta|^{2}$: ', n_f_mean, ' +/- ', n_f_std)


#plot the squares of beta against t
fig = plt.figure()

plt.plot(ts,beta2, color='teal', linewidth=2)
#########TEST############
plt.plot(ts,beta2_ex,'--', color='olive', linewidth=4)
##########################
#plt.scatter(t_plt,n, color='olive', linewidth=0.1)
plt.xlabel('$\eta$')
plt.ylabel(r'$|\beta|^{2}$')
#plt.ylim(-1,1)
fig.savefig(root+'sq_beta.png')
plt.show()

#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(ts,log_n, color='darkblue', linewidth=4)
#plt.scatter(t_plt,log_n, color='olive', linewidth=0.1)
plt.xlabel(r'$\eta (m^{-1})$')
plt.ylabel(r'$\ln(N_k)$')
#plt.xlim(-1,2*tosc+10)
#plt.ylim(0,23)
plt.tight_layout()
fig.savefig(root+'log_n_osc.png')
plt.show()

#########################
#save all the data to .txt files
np.savetxt(root + "time.txt",ts, delimiter=",")
np.savetxt(root + "up.txt",up, delimiter=",")
np.savetxt(root + "dup.txt",dup, delimiter=",")
np.savetxt(root + "um.txt",up, delimiter=",")
np.savetxt(root + "dum.txt",dup, delimiter=",")









