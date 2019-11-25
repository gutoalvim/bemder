#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.exterior_api import ExteriorBEM

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density


#% Load mesh
filename = 'hats_hrtf.msh'
grid = bempp.api.import_grid('Mshs/'+filename)
space = bempp.api.function_space(grid, "DP", 0)
#Defining frequencies of analysis 
f1= 250
f2 = 250
df = 10
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)


mu = {}

mu[1] = muh
mu[2] = muh
mu[3] = muh

v = {}

v[1] = np.zeros_like(f_range)
v[2] = np.zeros_like(f_range)
v[3] = np.ones_like(f_range)



points = {}


theta = np.linspace(0, 2*np.pi, 3600)
d = 0.028
for i in range(len(theta)):
    thetai = theta[i]
    radius = np.sqrt(1)
    # compute x1 and x2
    x1 = radius*np.cos(thetai)
    x2 = radius*np.sin(thetai)
    x3 = d
    points[i] = np.array([x1, x2+0.081, x3])

r0 = {}
r0[0] =  np.array([0,-1.2,0])
#r0[1] =  np.array([4.5,-1.2,1])



q = {}
q[0] = 0
#q[1] = 2

#%%
#% Defining grid plot properties 
plane = 'xz'
d = 0.1

grid_size = [1.6,-1.2]

n_grid_pts = 500






#%% Solve BEM

s1 = ExteriorBEM(space,f_range,r0,q,mu,v)

p,u = s1.velocity_bemsolve()

#%%
#pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u)

pt_T = s1.point_evaluate(points,p,u)

#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/hrtf_250.csv', sep=",", header=None)
data.columns = ["theta","spl"]

plt.polar(data.theta+np.pi/2, data.spl)
plt.polar((theta), 20*np.log10(np.abs(pt_T)/2e-5).reshape(len(theta),1))
plt.ylim([40,50])
plt.legend(['validarion_r1','bempp_r1'])
plt.savefig('exterior_eric_r1.png',dpi=500)
plt.show()



