#Creation of one Bosonic Fourier mode
#solves the harmonic oscillator equation
#with a time dependent frequency, omega
#using the rk4 method


import numpy as np
import matplotlib.pyplot as plt


#define all the constants needed
k = 0.05  #momentum of fourier mode
m = 1  #mass of boson
f = 0.5  #constant
Om = 1   #frequency of oscillations of omega^2
tmax = 100   #large time, equivalent to the +/- infinity limit
tosc = 3*np.pi/Om   #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h=1  #step size

root = 'boson_k'+str(k)+'_m'+str(m)+'_f'+str(f)+'_Om'+str(Om)+'_tosc'+str(tosc)+'_'

#define the omega^2 function
def om2(k,t):
    if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
        om2 = k**2 + (m + f * np.cos(Om*tosc))**2   #omega is constant
        return om2
    elif t > - tosc and t < tosc:
        om2 = k**2 + (m + f * np.cos(Om*t))**2  #omega is oscillating
        return om2
    else:
        print('ERROR: Omega is not defined for the value of t, of ',t)

#plot the frequency
om_sq = []
time = []
t = -tmax 
#for i in range(2*tmax):
 #   om_sq.append(om2(k,t))
 #   time.append(t)
 #   t += 1

fig = plt.figure()

#plt.plot(time,om_sq,color='teal', linewidth=2)
#plt.xlabel('t')
#plt.ylabel('$\omega^{2}$')
#fig.savefig(root+'om2.png')
#plt.show()

#set initial conditions
phi0 = 1/np.sqrt(2*om2(k,-tmax))  #initial condition for phi+
dphi0 = phi0 * np.complex(0,om2(k,-tmax))  #initial condition for dtphi+

print('Initial conditions: ', phi0,', ',dphi0)

#function to calculate derivatives for phi
def d2phi(phi2,t,k):
    #print('\nPhi: ',phi,'\nShape: ',phi.shape,'\nPhi[0].shape ',phi[0].shape,'\tPhi[1].shape: ',phi[1].shape)
    phi = phi2[0]
    dphi = phi2[1]
    return np.array([dphi,-om2(k,t)*phi])


#rk4 does a single RK4 step
def rk4(phi,k,t,h):
    k1 = d2phi(phi,t,k)
    k2 = d2phi(phi+h/2*k1,t+h/2,k)
    k3 = d2phi(phi+h/2*k2,t+h/2,k)
    k4 = d2phi(phi+h*k3,t+h,k)
    phi = phi+h*(k1+2*k2+2*k3+k4)/6
    t = t + h
    return (t,phi)


#define 
phi=np.array([phi0,dphi0])  # starting value for phi and dtphi
t = -tmax   # starting value for t
ts=np.array([t])     # to store all times
phis=np.array([phi])     # and all solution points


#calculate phi with rk4 method
for i in range(int(2*tmax/h)):  # take enough steps (or so)
    #print('\nPhi: ',phi,'\nShape: ',phi.shape,'\nPhi[0].shape ',phi[0].shape,'\tPhi[1].shape: ',phi[1].shape)
    
    (t,phi)=rk4(phi,k,t,h)
    if i % 10 == 0:
        print(t,': \t phi(t): ',phi[0], '\t dphi(t): ',phi[1])
    ts=np.append(ts,t)
    phis=np.concatenate((phis,np.array([phi])))


#transpose the final matrix to get the values of phi and dphi
[phi,dphi]=phis.transpose()

#get real and imaginary parts of phi and dphi
phi_real = phi.real
dphi_real = dphi.real
phi_imag = phi.imag
dphi_imag = dphi.imag

#get the complex conjugate of phi and dphi
phi_star = np.conj(phi)
dphi_star = np.conj(dphi)

####test#####
print(phi[0], ' and ', phi_star[0])
phi0_sq = phi[0] * phi_star[0]
print('\n',phi0_sq)

#calculate the square of phi and dphi
#ie phi x phi*
phi2 = phi * phi_star
dphi2 = dphi * dphi_star

#plot phi and dphi against t
fig = plt.figure()

plt.plot(ts,phi, color='olive', linewidth=1)
plt.plot(ts,dphi,'--', color='darkblue', linewidth=1)
plt.legend([r'Re($\varphi$)', r'Re($\partial_{t} \varphi$)'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'re_phi_dphi.png')
plt.show()

#plot the imaginary part of phi and dphi against t
fig = plt.figure()

plt.plot(ts,phi_imag, color='olive', linewidth=1)
plt.plot(ts,dphi_imag,'--', color='darkblue', linewidth=1)
plt.legend([r'Im($\varphi$)', r'Im($\partial_{t} \varphi$)'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'im_phi_dphi.png')
plt.show()

#plot the squares of phi and dphi against t
fig = plt.figure()

plt.plot(ts,phi2, color='olive', linewidth=1)
plt.plot(ts,dphi2,'--', color='darkblue', linewidth=1)
plt.legend([r'$|\varphi |^{2}$', r'$|\partial_{t} \varphi |^{2}$'],loc='upper right')
plt.xlabel('t')
fig.savefig(root+'sq_phi_dphi.png')
plt.show()


#############
#now calculate the Bogoliubov coefficients
beta_star = []
for i in range(2*tmax + 1):
#    beta_star.append(np.sqrt(2 * np.sqrt(om2(k,tmax)))/2 * np.exp(np.complex(0,om2(k,tmax)*ts[i]))*(phi[i] - dphi[i]/np.complex(0,om2(k,tmax))))
    beta_star.append(np.exp(np.complex(0,om2(k,tmax)*ts[i]))*(phi[i] - dphi[i]/np.complex(0,om2(k,tmax))))
beta = np.conj(beta_star)
beta2 = beta_star * beta

print (r'Final $|\beta|^{2}$: ', beta2[2*tmax])


#plot the squares of beta against t
fig = plt.figure()

plt.plot(ts,beta2, color='teal', linewidth=1)
plt.scatter(ts,beta2, color='darkblue', linewidth=1)
plt.xlabel('t')
plt.ylabel(r'$|\beta|^{2}$')
fig.savefig(root+'sq_beta.png')
plt.show()







