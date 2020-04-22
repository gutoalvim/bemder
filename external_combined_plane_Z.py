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
from bemder.exterior_api_new import ExteriorBEM

# Download data set from plotly repo

bempp.api.GMSH_PATH
#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density




# fig = plt.figure()
# ax = fig.gca(projection="3d")
#% Load mesh
filename = 'plane_Z.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(5,360,[90,270],axis = "z",random=True)
# S.plot()

R = receivers.Receiver()
R.arc_receivers(5,360,[0,360],axis = "z")
R.coord


muh = 0.001*np.ones_like(f_range)
zsd1 = porous.delany(20000,0.04,f_range)
# zsd2 = porous.delany(10000,0.04,f_range)

mud1 = np.complex128(rho0*c0/(np.conj(zsd1)))
mud2 = np.zeros_like(mud1)


mu = {}

mu[1] = mud1
mu[2] = mud2
mu[3] = mud2


mu_fi = np.array([mu[i][0] for i in mu.keys()])

#% Defining grid plot properties 
plane = 'xy'
d = 0

grid_size = [2,2]

n_grid_pts = 600



space = bempp.api.function_space(grid, "P", 1)


#%% Solve BEM

s2 = ExteriorBEM(space,f_range,r0,q,mu)

p2, u2 = s2.combined_direct_bemsolve_r()

#pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u,ax=True)
#%%
pt_T = s2.combined_point_evaluate_r(points,p2, u2)

pT = s2.combined_grid_evaluate_r(0,plane,d,grid_size,n_grid_pts,p2,u2)


#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/plane_Z.csv', sep=",", header=None)
data.columns = ["Lp"]

plt.polar((theta), data.Lp)
plt.ylim([80,95])
# plt.polar(theta, np.abs(pt_T).reshape(len(theta),1))
plt.polar(theta, -3+20*np.log10(np.abs(pt_T)/2e-5).reshape(len(theta),1))

# plt.ylim([50,100])
plt.show()



