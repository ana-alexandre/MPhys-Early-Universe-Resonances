#Creation of one Bosonic Fourier mode
#solves the harmonic oscillator equation
#with a time dependent frequency, omega
#using the rk5(4) method

#goes over several modes
#allowing to plot quantities like f against N



import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


##change the font size for pyplot
font = {'family' : 'normal',
        'size'   : 45}

plt.rc('font', **font)


#######################
#define all the constants needed
k = 0  #momentum of fourier mode
m = 0  #mass of boson
#f = 20*np.sqrt(2)  #constant
Om = 1  #frequency of oscillations of omega^2
tmax = 200   #large time, equivalent to the +/- infinity limit
#tosc = tmax
tosc = 20*np.pi/Om   #om2 oscillates between -tosc and +tosc (otherwise it is constant)
h_max = 1/(10*Om)  #step size
#######################


root = 'boson_k'+str(k)+'_m'+str(m)+'_Om'+str(Om)+'_tosc'+str(tosc)+'_'

#define the omega function
def om(k,t,f):
    if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
        om = np.sqrt(k**2 + (m + f * np.cos(Om*tosc))**2)   #omega is constant
        return om
    elif t > - tosc and t < tosc:
        om = np.sqrt(k**2 + (m + f * np.cos(Om*t))**2)  #omega is oscillating
        return om
    else:
        print('ERROR: Omega is not defined for the value of t of ',t)

#define the omega dot function
def dt_om(k,t,f):
    if (t >= - tmax and t < - tosc) or (t > tosc and t <= tmax):
        #omega is constant -->> omega dot = 0
        return 0
    elif t > - tosc and t < tosc:
        dt_om = (- Om * np.sin(Om*t) * (f**2 * np.cos(Om*t) + m * f)) / om(k,t,f)  #time derivative of omega oscilations
        return dt_om
    else:
        print('ERROR: Omega is not defined for the value of t of ',t)


#function to calculate derivatives for phi
def d2phi(t,phi2):
    #print('\nPhi: ',phi,'\nShape: ',phi.shape,'\nPhi[0].shape ',phi[0].shape,'\tPhi[1].shape: ',phi[1].shape)
    phi = phi2[0]
    dphi = phi2[1]
    return np.array([dphi,-(om(k,t,f)**2)*phi])


#create empty arrays to store what's to be plotted
n_final = []
n_std = []
fs = []

##########################
#loop starts here

for f in np.arange(0.1,7,0.01):

    #set initial conditions
    phi0 = np.exp(np.complex(0,-om(k,-tmax,f)*tmax))/np.sqrt(2*om(k,-tmax,f))  #initial condition for phi+
    dphi0 = phi0 * np.complex(0,om(k,-tmax,f))  #initial condition for dtphi+
    ###########################

    #print('Initial conditions: ', phi0,', ',dphi0)

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
        beta_star_temp = np.sqrt(om(k,ts[i],f)/2)*(phi[i] - dphi[i]/np.complex(0,om(k,ts[i],f)))
        alpha_temp = np.sqrt(om(k,ts[i],f)/2)*(phi[i] + dphi[i]/np.complex(0,om(k,ts[i],f)))
        beta_star.append(beta_star_temp)
        alpha.append(alpha_temp)


    #now calculate alpha and beta squared by taking the product with the complex conjugate
    beta = np.conj(beta_star)
    alpha_star = np.conj(alpha)
    n = beta_star * beta
    alpha2 = alpha_star * alpha

    #calculate the difference between alpha squared and beta squared
    #which should equal 1
    dif = alpha2 - n
    dif_mean = np.mean(dif)
    dif_std = np.std(dif)

    print('\n',f,': \t alpha2 - beta2 = ', dif_mean, ' +/- ', dif_std)

    #calculate the average of the final
    where = np.argwhere(ts > tosc)[0][0]
    n_f = n[where:]
    n_f_mean = np.mean(n_f)
    n_f_std = np.std(n_f)
    print ('\n',f,': \t ',r'Final $|\beta|^{2}$: ', n_f_mean, ' +/- ', n_f_std)


    #now append values to save
    n_final.append(n_f_mean)
    n_std.append(n_f_std)
    fs.append(f)


#calculate log on final n
log_n_final = np.log(n_final)
log_n_error = np.asarray(n_std) / np.asarray(n_final)

#calculate q = f^2/(4m^2)
q = np.asarray(fs)**2/(4*Om**2)

#3-sigma errors
n_3std = 3 * np.asarray(n_std)
log_n_3error = 3 * log_n_error

####################
###### PLOTS #######

#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(fs, n_final, yerr = n_3std, fmt = 'o',  color='darkblue', capsize=0.5, linestyle = 'none')
#plt.scatter(fs,n_final, color='olive', linewidth=0.1)
plt.xlabel(r'$g\phi_0 (m)$')
plt.ylabel(r'$N_k$')
#plt.xlim(0,2*tosc)
#plt.ylim(40,52)
plt.tight_layout()
fig.savefig(root+'n_against_f.png')
plt.show()

#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(q, n_final, yerr = n_3std, fmt = 'o',  color='darkblue', capsize=0.5, linestyle = 'none')
#plt.scatter(fs,n_final, color='olive', linewidth=0.1)
plt.xlabel(r'$q$')
plt.ylabel(r'$N_k$')
#plt.xlim(0,2*tosc)
#plt.ylim(40,52)
plt.tight_layout()
fig.savefig(root+'n_against_q.png')
plt.show()


#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(fs, log_n_final, yerr = log_n_3error, fmt = 'o', color='darkblue', capsize=0.5, linestyle = 'none')
#plt.scatter(fs,n_final, color='olive', linewidth=0.1)
plt.xlabel(r'$g\phi_0 (m)$')
plt.ylabel(r'$ln(N_k)$')
#plt.xlim(75,2*tosc)
#plt.ylim(40,52)
plt.tight_layout()
fig.savefig(root+'log_n_against_f.png')
plt.show()

#plot the squares of beta against t
fig = plt.figure(figsize=(19.20,10.80))

plt.errorbar(q, log_n_final, yerr = log_n_3error, fmt = 'o', color='darkblue', capsize=0.5, linestyle = 'none')
#plt.scatter(fs,n_final, color='olive', linewidth=0.1)
plt.xlabel(r'$q$')
plt.ylabel(r'$ln(N_k)$')
#plt.xlim(75,2*tosc)
#plt.ylim(40,52)
plt.tight_layout()
fig.savefig(root+'log_n_against_q.png')
plt.show()



#########################
#save all the data to .txt files
np.savetxt(root + "fs.txt",fs, delimiter=",")
np.savetxt(root + "n_final.txt",n_final, delimiter=",")
np.savetxt(root + "n_error.txt",n_std, delimiter=",")






