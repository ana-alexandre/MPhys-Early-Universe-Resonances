##########TEST###############
#look at the SHO
#with constant frequency different than 1
#compare the numerical solutions to the analytical ones
#calculated from mathematica

#Creation of one Bosonic Fourier mode
#solves the harmonic oscillator equation
#with a time dependent frequency, omega
#using the rk5(4) method


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#######################
#define all the constants needed
k = 1  #momentum of fourier mode
m = 0  #mass of boson
f = 0  #constant
Om = 1   #frequency of oscillations of omega^2
tmax = 100   #large time, equivalent to the +/- infinity limit
tosc = 8*np.pi/Om   #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/(10*Om)  #step size
#######################


root = 'boson_k'+str(k)+'_m'+str(m)+'_f'+str(f)+'_Om'+str(Om)+'_tosc'+str(tosc)+'_'

#define the omega^2 function
#def om2(k,t):
 #   if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
  #      om2 = k**2 + (m + f * np.cos(Om*tosc))**2   #omega is constant
   #     return om2
   # elif t > - tosc and t < tosc:
   #     om2 = k**2 + (m + f * np.cos(Om*t))**2  #omega is oscillating
   #     return om2
   # else:
   #     print('ERROR: Omega is not defined for the value of t of ',t)


#####################
###########TEST#################
#define the omega^2 function
def om2(k,t):
    #om2 = k**2 + (m + f * np.cos(Om*tosc))**2   #omega is constant
    om2 = 1.2
    return om2
#############################

#define the analytical functions for phi and dphi
def analytical(t):
    phi = np.complex(-0.61478,-0.163962)*np.cos(1.2*t) + np.complex(0.196754,-0.512317)*np.sin(1.2 t)
    dphi = np.complex(0.236105,-0.61478)*np.cos(1.2*t) + np.complex(0.737736,0.196754)*np.sin(1.2 t)
    return phi,dphi

#plot the frequency
om_sq = []
ts = []
t = -tmax 
for i in range(2*tmax):
    om_sq.append(om2(k,t))
    ts.append(t)
    t += 1

fig = plt.figure()

plt.plot(ts,om_sq,color='teal', linewidth=2)
plt.xlabel('t')
plt.ylabel('$\omega^{2}$')
fig.savefig(root+'om2.png')
plt.show()


##########################
#set initial conditions
phi0 = 1/np.sqrt(2*om2(k,-tmax))  #initial condition for phi+
dphi0 = phi0 * np.complex(0,om2(k,-tmax))  #initial condition for dtphi+
###########################


print('Initial conditions: ', phi0,', ',dphi0)

#function to calculate derivatives for phi
def d2phi(t,phi2):
    #print('\nPhi: ',phi,'\nShape: ',phi.shape,'\nPhi[0].shape ',phi[0].shape,'\tPhi[1].shape: ',phi[1].shape)
    phi = phi2[0]
    dphi = phi2[1]
    return np.array([dphi,-om2(k,t)*phi])


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
n_phis = sol.y

print('time shape: ',ts.shape)

#transpose the final matrix to get the values of phi and dphi
[n_phi,n_dphi]=n_phis

print('\n phi(t): ',n_phi[:50], '\n dphi(t): ', n_dphi[:50])

#get real and imaginary parts of phi and dphi
n_phi_real = n_phi.real
n_dphi_real = n_dphi.real
n_phi_imag = n_phi.imag
n_dphi_imag = n_dphi.imag

#get the complex conjugate of phi and dphi
n_phi_star = np.conj(n_phi)
n_dphi_star = np.conj(n_dphi)


#calculate the square of phi and dphi
#ie phi x phi*
n_phi2 = n_phi * n_phi_star
n_dphi2 = n_dphi * n_dphi_star

#############################
#now get the analytical expression from mathematica
#and calculate phi, dphi, alpha and beta
a_phi = []
a_dphi = []

for t in ts:
    aphi,adphi = analytical(t)
    a_phi.append(phi)
    a_dphi.append(dphi)

#now calculate the Bogoliubov coefficients, beta* and alpha 
#ignore complex exponential factor, since we are only interested in the square of the alpha and beta
n_beta_star = np.sqrt(om2(k,ts)/2)*(n_phi - n_dphi/np.complex(0,om2(k,ts)))
n_alpha = np.sqrt(om2(k,ts)/2)*(n_phi + n_dphi/np.complex(0,om2(k,ts)))

a_beta_star = np.sqrt(om2(k,ts)/2)*(a_phi - a_dphi/np.complex(0,om2(k,ts)))
a_alpha = np.sqrt(om2(k,ts)/2)*(a_phi + a_dphi/np.complex(0,om2(k,ts)))


#now calculate alpha and beta squared by taking the product with the complex conjugate
n_beta = np.conj(n_beta_star)
n_alpha_star = np.conj(n_alpha)
n_beta2 = n_beta_star * n_beta
n_alpha2 = n_alpha_star * n_alpha

a_beta = np.conj(a_beta_star)
a_alpha_star = np.conj(a_alpha)
a_beta2 = a_beta_star * a_beta
a_alpha2 = a_alpha_star * a_alpha


#calculate the difference between alpha squared and beta squared
#which should equal 1
n_dif = n_alpha2 - n_beta2
n_dif_mean = np.mean(n_dif)
n_dif_std = np.std(n_dif)

a_dif = a_alpha2 - a_beta2
a_dif_mean = np.mean(a_dif)
a_dif_std = np.std(a_dif)

print('\nnumerical alpha2 - beta2 = ', n_dif_mean, ' ± ', n_dif_std)
print('\nanalytical alpha2 - beta2 = ', a_dif_mean, ' ± ', a_dif_std)

#plot phi and dphi against t
fig = plt.figure()

plt.plot(ts,n_phi, color='olive', linewidth=1)
plt.plot(ts,n_dphi,'--', color='darkblue', linewidth=1)
plt.plot(ts,a_phi, color='purple', linewidth=1)
plt.plot(ts,a_dphi,'--', color='darkred', linewidth=1)
plt.legend([r'Num Re($\varphi$)', r'Num Re($\partial_{t} \varphi$)',r'Ana Re($\varphi$)', r'Ana Re($\partial_{t} \varphi$)'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'re_phi_dphi.png')
plt.show()

#plot the imaginary part of phi and dphi against t
fig = plt.figure()

plt.plot(ts,n_phi_imag, color='olive', linewidth=1)
plt.plot(ts,n_dphi_imag,'--', color='darkblue', linewidth=1)
plt.plot(ts,a_phi_imag, color='purple', linewidth=1)
plt.plot(ts,a_dphi_imag,'--', color='darkred', linewidth=1)
plt.legend([r'Num Im($\varphi$)', r'Num Im($\partial_{t} \varphi$)',r'Ana Im($\varphi$)', r'Ana Im($\partial_{t} \varphi$)'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'im_phi_dphi.png')
plt.show()

print (r'Final numerical  $|\beta|^{2}$: ', n_beta2[n_beta2.shape[0]-1])
print (r'Final analytical $|\beta|^{2}$: ', a_beta2[a_beta2.shape[0]-1])

#plot the squares of beta against t
fig = plt.figure()

plt.plot(ts,n_beta2, color='teal', linewidth=1)
plt.scatter(ts,n_beta2, color='darkblue', linewidth=0.5)
plt.plot(ts,a_beta2, color='olive', linewidth=1)
plt.scatter(ts,a_beta2, color='darkolivegreen', linewidth=0.5)
plt.xlabel('t')
plt.ylabel(r'$|\beta|^{2}$')
fig.savefig(root+'sq_beta.png')
plt.show()







