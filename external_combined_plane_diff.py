#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
import bemder.sources as sources
from bemder.exterior_api_NEWBEM import ExteriorBEM



# Download data set from plotly repo

bempp.api.GMSH_PATH
#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density




# fig = plt.figure()
# ax = fig.gca(projection="3d")
#% Load mesh

filename = 'qrd_2d_5.msh'
filename_ref = 'plane_ref.msh'
grid = bempp.api.import_grid('Mshs/'+filename)
grid_ref = bempp.api.import_grid('Mshs/'+filename_ref)


# ac = ax.scatter3D(v[0,:],v[1,:],v[2,:])
# pc = ax.plot3D(fc[0,:],fc[1,:],fc[2,:])
# fig.tight_layout()

# plt.show()

# n = grid.number_of_vertices
# d = grid.geometry().dim()



#Defining frequencies of analysis 

f1= 1000
f2 = 1000
df = 10
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance

points = {}
theta = np.linspace(0, 2*np.pi, 360)
d =0
for i in range(len(theta)):
    thetai = theta[i]
    radius = 20
    # compute x1 and x2
    x1 = d+radius*np.cos(thetai)
    x2 = d +radius*np.sin(thetai)
    x3 = d
    points[i] = np.array([x1, x3, x2])
    
pts= np.array([points[i] for i in points.keys()])

    


muh = 0.001*np.ones_like(f_range)
zsd1 = porous.delany(5000,0.1,f_range)
zsd2 = porous.delany(10000,0.04,f_range)

mud1 = np.complex128(rho0*c0/np.conj(zsd1))
mud2 = np.complex128(rho0*c0/np.conj(zsd2))
space = bempp.api.function_space(grid, "DP", 0)


mu = {}

mu[1] = muh
mu[2] = mud2

mu_fi = np.array([mu[i][0] for i in mu.keys()])

r0 = {}
r0[0] =  np.array([1,0,0])
# r0[1] =  np.array([0,0,-0.3])
# 
q = {}
q[0] = 1
# q[1] = 1

#% Defining grid plot properties 
plane = 'xy'
d = 0

grid_size = [2,2]

n_grid_pts = 600



space1 = bempp.api.function_space(grid, "DP", 0)
space2 = bempp.api.function_space(grid_ref, "DP", 0)



#%% Solve BEM
s1 = ExteriorBEM(space1,f_range,r0,q,mu)
s2 = ExteriorBEM(space2,f_range,r0,q,mu)

p1 = s1.combined_direct_bemsolve_n()

p2 = s2.combined_direct_bemsolve_n()

#pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u,ax=True)
#%%
ps1 = s1.scattered_point_evaluate_n(points,p1)
ps2 = s2.scattered_point_evaluate_n(points,p2)

pT = s1.combined_grid_evaluate_n(0,plane,d,grid_size,n_grid_pts,p1)
pT = s2.combined_grid_evaluate_n(0,plane,d,grid_size,n_grid_pts,p2)


#%% Plot Comparison between Bempp and Validation
# data = pd.read_csv('Data/soft_sphere_Lp.csv', sep=",", header=None)
# data.columns = ["Lp"]

# plt.polar((theta), data.Lp)
# plt.ylim([80,95])
# plt.polar(theta, np.abs(pt_T).reshape(len(theta),1))
plt.polar(theta, 20*np.log10(np.abs(ps1)/2e-5).reshape(len(theta),1))
plt.polar(theta, 20*np.log10(np.abs(ps2)/2e-5).reshape(len(theta),1))


# plt.ylim([50,100])
plt.show()

plt.polar(theta, np.abs(ps1).reshape(len(theta),1))
plt.polar(theta,np.abs(ps2).reshape(len(theta),1))


#%% Diffusion Coef

T = (np.sum(np.abs(ps1))**2 - np.sum(np.abs(ps1)**2))/((len(ps1.T)-1)*np.sum(np.abs(ps1))**2)

S = 1 - (np.abs(np.sum(ps1*np.conj(ps2)))**2/(np.sum(np.abs(ps1)**2)*np.sum(np.abs(ps2)**2)))


