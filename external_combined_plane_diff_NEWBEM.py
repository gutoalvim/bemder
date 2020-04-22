#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
import bemder.receivers as receivers
import bemder.controlsair as ctrl
from bemder import sources
from bemder import receivers
from bemder.bem_api_new import ExteriorBEM
from matplotlib import style
style.use("seaborn-talk")
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AG = ctrl.AlgControls(AP.c0, 1000,1000,500) #Defines frequencies of analysis

#% Load mesh

filename = 'QRD_1D_refine.msh'
filename_ref = 'QRD_1D_ref_refine.msh'
grid = bempp.api.import_grid('Mshs/'+filename)
grid_ref = bempp.api.import_grid('Mshs/'+filename_ref)

#Defining Sources and Receivers

S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()

R = receivers.Receiver()
R.arc_receivers(5,360,[-90,90],axis = "z")
R.coord

#% Defining grid plot properties 
plane = 'z'
d = 0

grid_size = [2,2]

n_grid_pts = 600


# np.savetxt("arch_5m_z.csv",R.coord)


#%% Solve BEM
s1 = ExteriorBEM(grid,AG,AP,S)
s2 = ExteriorBEM(grid_ref,AG,AP,S)

p1 = s1.hard_bemsolve()

p2 = s2.hard_bemsolve()

#pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u,ax=True)
#%%
pt1,ps1 = s1.point_evaluate(p1,R=R)
pt2, ps2 = s2.point_evaluate(p2,R=R)

# pT = s1.combined_grid_evaluate_n(0,plane,d,grid_size,n_grid_pts,p1)
# pT = s1.combined_grid_evaluate(p1,ps1,0,plane,d,grid_size,n_grid_pts)

# pT = s2.combined_grid_evaluate(p2,ps2,0,plane,d,grid_size,n_grid_pts)


# %% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/QRD_Array_Lp.csv', sep=",", header=None)
data1 = pd.read_csv('Data/QRD_Array_Lp_s.csv', sep=",", header=None)

data.columns = ["Lp"]
data1.columns = ["Lp"]


# plt.polar((R.theta), data1.Lp+3)

# plt.ylim([80,95])
# plt.polar(theta, np.abs(pt_T).reshape(len(theta),1))
# plt.polar(R.theta, 20*np.log10(np.abs(ps1)/2e-5).reshape(len(theta),1))
i = 0
Lp1 = 20*np.log10(np.abs(ps1[i,:])/2e-5).reshape(len(R.theta),1)-max(20*np.log10(np.abs(ps1[i,:])/2e-5).reshape(len(R.theta),1))
# Lp2 = 20*np.log10(np.abs(ps2[i,:])/2e-5).reshape(len(R.theta),1)-max(20*np.log10(np.abs(ps2[i,:])/2e-5).reshape(len(R.theta),1))

plt.polar(R.theta, Lp1,linewidth=3,label="QRD")
plt.polar((R.theta), data1.Lp-max(data1.Lp),linewidth=2,label="Validation")

# plt.polar(R.theta, Lp2,linewidth=3,label="Plane")

plt.title("Scattered Pressure For Array of QRD 1D Diffuser vs Ref. Plane \n")
plt.legend(loc="lower left")

# plt.ylim([min(20*np.log10(np.abs(ps2)/2e-5).reshape(len(R.theta),1))-5,max(20*np.log10(np.abs(ps2)/2e-5).reshape(len(R.theta),1))+1])
# plt.ylim([-25,1])
# plt.savefig("f.png",dpi=600)
plt.show()

# plt.polar(theta, np.abs(ps1).reshape(len(theta),1))
# plt.polar(theta,np.abs(ps2).reshape(len(theta),1))


#%% Diffusion Coef

T = (np.sum(np.abs(ps1[i,:]))**2 - np.sum(np.abs(ps1[i,:])**2))/((len(ps1.T))*np.sum(np.abs(ps1[i,:])**2))
T_ref = (np.sum(np.abs(ps2[i,:]))**2 - np.sum(np.abs(ps2[i,:])**2))/((len(ps2.T))*np.sum(np.abs(ps2[i,:])**2))

Tf = (T - T_ref)/(1-T_ref)


s= 1 - (np.abs(np.sum(ps1[i,:]*np.conj(ps2[i,:])))**2/(np.sum(np.abs(ps1[i,:])**2)*np.sum(np.abs(ps2[i,:])**2)))

#%%
import pickle
with open("Diffuser - 1k - bP", "wb") as f:
    pickle.dump(p1, f)
