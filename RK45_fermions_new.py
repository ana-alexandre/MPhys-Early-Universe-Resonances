#new code to study the creations of fermions

#I will start just by solving the SHO equation for u+ and u-
#and slowly add on the rest of the calculations
# to really make sure everything is working




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#function to calculate derivatives for up
def d2up(t,up2):
    up = up2[0]
    dup = up2[1]
    ddup = -(100**2 + 10**2) * up  #######################
    return np.array([dup,ddup])

#function to calculate derivatives for um
def d2um(t,um2):
    um = um2[0]
    dum = um2[1]
    ddum = - (100**2 + 10**2) * um ########################
    return np.array([dum,ddum])


#set initial conditions
up0 = np.complex(np.sqrt(1 - 10/np.sqrt(100**2 + 10**2)),0) ###########################
um0 = np.complex(np.sqrt(1 + 10/np.sqrt(100**2 + 10**2)),0) ##########################
dup0 = np.complex(0,100*um0 - 10*up0)  ########################
dum0 = np.complex(0,100*up0 + 10*um0)  ########################


#calculate solutions using RK45 method
sol_p = solve_ivp(d2up, [0,0.2 * np.pi], np.array([up0,dup0]), method='RK45',max_step=0.001) ############# tmax
sol_m = solve_ivp(d2um, [0, 0.2 * np.pi], np.array([um0,dum0]), method='RK45',max_step=0.001)  ############# tmax

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



