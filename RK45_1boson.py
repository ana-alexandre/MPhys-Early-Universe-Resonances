#:Creation of one Bosonic Fourier mode
#solves the harmonic oscillator equation
#with a time dependent frequency, omega
#using the rk5(4) method

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##change the font size for pyplot
font = {'family' : 'normal',
        'size'   : 15}

plt.rc('font', **font)




#######################
#define all the constants needed
k = 0  #momentum of fourier mode
m = 0  #mass of boson
f = np.sqrt(800)  #constant
Om = 1  #frequency of oscillations of omega^2
tmax = 200   #large time, equivalent to the +/- infinity limit
#tosc = tmax
tosc = 15*np.pi/Om   #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/(10*Om)  #step size
#######################


root = 'boson_k'+str(k)+'_m'+str(m)+'_f'+str(f)+'_Om'+str(Om)+'_tosc'+str(tosc)+'_'

#define the omega function
def om(k,t):
    if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
        om = np.sqrt(k**2 + (m + f * np.cos(Om*tosc))**2)   #omega is constant
        return om
    elif t > - tosc and t < tosc:
        om = np.sqrt(k**2 + (m + f * np.cos(Om*t))**2)  #omega is oscillating
        return om
    else:
        print('ERROR: Omega is not defined for the value of t of ',t)

#define the omega dot function
def dt_om(k,t):
    if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
        #omega is constant -->> omega dot = 0
        return 0
    elif t > - tosc and t < tosc:
        dt_om = (- Om * np.sin(Om*t) * (f**2 * np.cos(Om*t) + m * f)) / om(k,t)  #time derivative of omega oscilations
        return dt_om
    else:
        print('ERROR: Omega is not defined for the value of t of ',t)




#plot the frequency
om_fct = []
om_dot = []
limit = []
ts = []
t = -tmax
for i in range(2*tmax):
    #calculate omega
    om_fct.append(om(k,t))

    #calculate omega dot
    om_dot.append(dt_om(k,t))

    #save limit value
    limit.append(2 * np.sqrt(2))

    #save time values
    ts.append(t)
    t += 1


#now calculate the adiabatic condition
#which equals |omega dot/omega squared|
om_adiab = np.abs(om_dot / np.square(om_fct))

#plot omega
fig = plt.figure()

plt.plot(ts,om_fct,color='teal', linewidth=1)
plt.xlabel('t')
plt.ylabel(r'$\omega$')
fig.savefig(root+'om2.png')
plt.show()

ts = np.asarray(ts) + tosc

#plot the adiabatic condition
fig = plt.figure()

plt.plot(ts,om_adiab,color='teal', linewidth=1)
#plt.plot(ts,limit,color='darkblue', linewidth=1)
plt.xlabel('t')
plt.ylabel(r'$|\frac{\dot \omega}{\omega^2}|$')
plt.xlim(75,2*tosc)
plt.ylim(0,3)
fig.savefig(root+'om_adiabatic.png')
plt.show()

ts = np.asarray(ts) - tosc

##########################
#set initial conditions
phi0 = np.exp(np.complex(0,-om(k,-tmax)*tmax))/np.sqrt(2*om(k,-tmax))  #initial condition for phi+
dphi0 = phi0 * np.complex(0,om(k,-tmax))  #initial condition for dtphi+
###########################


print('Initial conditions: ', phi0,', ',dphi0)

#function to calculate derivatives for phi
def d2phi(t,phi2):
    #print('\nPhi: ',phi,'\nShape: ',phi.shape,'\nPhi[0].shape ',phi[0].shape,'\tPhi[1].shape: ',phi[1].shape)
    phi = phi2[0]
    dphi = phi2[1]
    return np.array([dphi,-(om(k,t)**2)*phi])


#assign initial values
phi_ini = np.array([phi0,dphi0])  # starting value for phi and dtphi
#t = -tmax   # starting value for t
#ts=np.array([t])     # to store all times
phis=np.array([phi_ini])     # and all solution points
betas = np.array([0])
betas2 = np.array([0])
alphas2 = np.array([1])


#calculate phi with rk4 method
sol = solve_ivp(d2phi, [-tmax, tmax], phi_ini, method='RK45', max_step = h_max)

#the time and functions
ts = sol.t
phis = sol.y

#transpose the final matrix to get the values of phi and dphi
[phi,dphi]=phis

#print('\n phi(t): ',phi[:50], '\n dphi(t): ', dphi[:50])

#get real and imaginary parts of phi and dphi
phi_real = phi.real
dphi_real = dphi.real
phi_imag = phi.imag
dphi_imag = dphi.imag

#get the complex conjugate of phi and dphi
phi_star = np.conj(phi)
dphi_star = np.conj(dphi)


#calculate the square of phi and dphi
#ie phi x phi*
phi2 = phi * phi_star
dphi2 = dphi * dphi_star


#now calculate the Bogoliubov coefficients, beta* and alpha 
#ignore complex exponential factor, since we are only interested in the square of the alpha and beta
beta_star = []
alpha = []
for i in range(ts.shape[0]):
    beta_star_temp = np.sqrt(om(k,ts[i])/2)*(phi[i] - dphi[i]/np.complex(0,om(k,ts[i])))
    alpha_temp = np.sqrt(om(k,ts[i])/2)*(phi[i] + dphi[i]/np.complex(0,om(k,ts[i])))
    beta_star.append(beta_star_temp)
    alpha.append(alpha_temp)

#now calculate alpha and beta squared by taking the product with the complex conjugate
beta = np.conj(beta_star)
alpha_star = np.conj(alpha)
n = beta_star * beta
log_n = np.log(n)
alpha2 = alpha_star * alpha

#calculate the difference between alpha squared and beta squared
#which should equal 1
dif = alpha2 - n
dif_mean = np.mean(dif)
dif_std = np.std(dif)

print('\nalpha2 - beta2 = ', dif_mean, ' +/- ', dif_std)


#shift the time scale so that oscilations start at t=0
t_plt = ts +tosc

#plot phi and dphi against t
fig = plt.figure()

plt.plot(t_plt,phi, color='olive', linewidth=1)
plt.plot(t_plt,dphi,'--', color='darkblue', linewidth=1)
plt.legend([r'Re($\varphi$)', r'Re($\partial_{t} \varphi$)'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'re_phi_dphi.png')
plt.show()

#plot phi and dphi against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(t_plt,phi, color='olive', linewidth=4)
plt.plot(t_plt,dphi,'--', color='darkblue', linewidth=4)
plt.legend([r'Re($\varphi_k$)', r'Re($\dot \varphi_k$)'],loc='upper right')
plt.xlabel(r'$t (m^{-1})$')
#plt.ylabel(r'Re($\varphi_k$)')
plt.xlim(-1,2*tosc+10)
#plt.ylim(-35000,35000)
plt.tight_layout()
fig.savefig(root+'re_phi_dphi_osc.png')
plt.show()


#plot the imaginary part of phi and dphi against t
#fig = plt.figure()

#plt.plot(ts,phi_imag, color='olive', linewidth=1)
#plt.plot(ts,dphi_imag,'--', color='darkblue', linewidth=1)
#plt.legend([r'Im($\varphi$)', r'Im($\partial_{t} \varphi$)'],loc='upper right')
#plt.xlabel('t')
#fig.savefig(root+'im_phi_dphi.png')
#plt.show()

#plot the squares of phi and dphi against t
#fig = plt.figure()

#plt.plot(ts,phi2, color='olive', linewidth=1)
#plt.plot(ts,dphi2,'--', color='darkblue', linewidth=1)
#plt.legend([r'$|\varphi |^{2}$', r'$|\partial_{t} \varphi |^{2}$'],loc='upper right')
#plt.xlabel('t')
#fig.savefig(root+'sq_phi_dphi.png')
#plt.show()

#calculate the average of the final
where = np.argwhere(ts > tosc)[0][0]
n_f = n[where:]
n_f_mean = np.mean(n_f)
n_f_std = np.std(n_f)
print (r'Final $|\beta|^{2}$: ', n_f_mean, ' +/- ', n_f_std)

#plot the squares of beta against t
fig = plt.figure()

plt.plot(t_plt,n, color='teal', linewidth=2)
#plt.scatter(t_plt,n, color='olive', linewidth=0.1)
plt.xlabel('t')
plt.ylabel(r'$|\beta|^{2}$')
#plt.ylim(-1,1)
fig.savefig(root+'sq_beta.png')
plt.show()

#plot the squares of beta against t
fig = plt.figure()

plt.plot(t_plt,log_n, color='teal', linewidth=2)
#plt.scatter(t_plt,log_n, color='olive', linewidth=0.1)
plt.xlabel('t')
plt.ylabel(r'$\ln(|\beta|^{2})$')
fig.savefig(root+'log_n.png')
plt.show()

#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.plot(t_plt,log_n, color='darkblue', linewidth=4)
#plt.scatter(t_plt,log_n, color='olive', linewidth=0.1)
plt.xlabel(r'$t (m^{-1})$')
plt.ylabel(r'$\ln(N_k)$')
plt.xlim(-1,2*tosc+10)
#plt.ylim(0,23)
plt.tight_layout()
fig.savefig(root+'log_n_osc.png')
plt.show()

#########################
#save all the data to .txt files
np.savetxt(root + "time.txt",ts, delimiter=",")
np.savetxt(root + "phi.txt",phi, delimiter=",")
np.savetxt(root + "dphi.txt",dphi, delimiter=",")




