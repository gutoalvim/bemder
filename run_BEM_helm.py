#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.interior_api import RoomBEM

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density


#% Load mesh
filename = 'helm.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#Defining frequencies of analysis 
f1= 183
f2 = 183
df = 2
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)
mum = np.complex(rho0*c0/(500*1e-6))*np.ones_like(f_range)
U = np.ones_like(f_range)


mu = {}

mu[1] = muh
mu[2] = mum
mu[3] = muh

v = {}
v[1] = 0*U
v[2] = 0*U
v[3] = U


points = {}
points[0] = np.array([0,2,2])
#points[1] = np.array([0.6,0.2,-0.15])


r0 = {}
r0[0] =  np.array([-0.5,4,1.2])
#r0[1] = np.array([1.4,-0.7,-0.35])

q = {}
q[0] = 0
#q[1] = 1

#% Defining grid plot properties 
plane = 'xz'
d = 0

grid_size = [0.6,0.4]

n_grid_pts = 250



space = bempp.api.function_space(grid, "DP", 0)


#%% Solve BEM

s1 = RoomBEM(space,f_range,r0,q,mu,v)

p,u = s1.bemsolve()

pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u)

#%% Plot Comparison between Bempp and Validation

