#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
# import bemder.controlsair as ctrl
# from bemder import sources
# from bemder import receivers
from bemder.bem_api_new import ExteriorBEM
# from bemder import helpers as hh
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl


#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 200,500,2) #Defines frequencies of analysis
AC.freq = [1000]
# AC.freq = [200,229,250, 400,450,500, 1000]
#% Load mesh

filename_ref = 'plane_Z.msh'
grid_ref = bempp.api.import_grid('Mshs/'+filename_ref)


zsd = porous.delany(20000,0.04,AC.freq)

mud1 = np.complex128(1/(zsd))
muz = np.zeros_like(mud1)

mu = {}

mu[1] = mud1
mu[2] = muz
mu[3] = muz
#Defining Sources and Receivers

S = sources.Source("plane",coord=[1,0,0])
# S.arc_sources(5,360,[90,270],axis = "z",random=True)
# S.plot()

R = receivers.Receiver()#(coord=[1,0,0])
R.arc_receivers(1,360,[0,360],axis = "y")
# R.coord

#% Defining grid plot properties 
plane = 'y'
d = 0

grid_size = [2,2]

n_grid_pts = 600






#%% Solve BEM
# s1 = ExteriorBEM(grid_ref,AG,AP,S,R)
s2 = ExteriorBEM(grid_ref,AC,AP,S,R,mu)

# p1 = s1.hard_bemsolve()

boundSol = s2.impedance_bemsolve()
#%%
s2.bem_save("Z_break")
#%%
s2 = ExteriorBEM(grid_ref,AC,AP,S,R,mu)
boundSol = s2.bem_load("Z_break")

#pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u,ax=True)
#%%
# pt1,ps1 = s1.point_evaluate(p1,p1,R)
pt2, ps2 = s2.point_evaluate(boundSol,R)

# pT = s1.combined_grid_evaluate(p1,u2,0, plane,d,grid_size,n_grid_pts)
pT = s2.combined_grid_evaluate(boundSol,0, plane,d,grid_size,n_grid_pts)


#%%

plt.plot(AC.freq,20*np.log10(np.abs(pt2[:,:])/2e-5))

     #%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/plane_Z.csv', sep=",", header=None)
# data.columns = ["freq","Lp"]
data.columns = ["Lp"]


# plt.plot((data.freq), data.Lp+3)
plt.polar((R.theta), data.Lp+2)

# plt.plot(AC.freq,20*np.log10(np.abs(pt2[:,:])/2e-5))

# plt.ylim([80,95])
# plt.polar(theta, np.abs(pt_T).reshape(len(theta),1))
# plt.polar(theta, 20*np.log10(np.abs(ps1)/2e-5).reshape(len(theta),1))
# plt.polar(R.theta, 20*np.log10(np.abs(pt1)/2e-5).reshape(len(R.theta),1))
plt.polar(R.theta, 20*np.log10(np.abs(pt2)/2e-5).reshape(len(R.theta),1))


# plt.ylim([min(20*np.log10(np.abs(pt2)/2e-5).reshape(len(R.theta),1))-1,max(20*np.log10(np.abs(pt2)/2e-5).reshape(len(R.theta),1))+1])
plt.ylim([70,100])
plt.show()

# plt.polar(theta, np.abs(ps1).reshape(len(theta),1))
# plt.polar(theta,np.abs(ps2).reshape(len(theta),1))

#%% Diffusion Coef

T = (np.sum(np.abs(ps1))**2 - np.sum(np.abs(ps1)**2))/((len(ps1.T)-1)*np.sum(np.abs(ps1))**2)

S = 1 - (np.abs(np.sum(ps1*np.conj(ps2)))**2/(np.sum(np.abs(ps1)**2)*np.sum(np.abs(ps2)**2)))


