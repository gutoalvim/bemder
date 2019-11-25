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
filename = 'hats.msh'
grid = bempp.api.import_grid('Mshs/'+filename)
space = bempp.api.function_space(grid, "DP", 0)
#Defining frequencies of analysis 
f1= 1000
f2 = 1000
df = 10
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)


mu = {}

mu[1] = muh




points = {}
points[0] = np.array([0,0,0.25])
points[1] = np.array([0.1,-0.1,0.15])
points[2] = np.array([-0.1,0.1,0.1])




r0 = {}
r0[0] =  np.array([0,-1.2,0])
#r0[1] =  np.array([4.5,-1.2,1])



q = {}
q[0] = 2
#q[1] = 2

#%%
#% Defining grid plot properties 
plane = 'yz'
d = 0.1

grid_size = [1.6,-1.2]

n_grid_pts = 500

#%% Solve BEM

s1 = ExteriorBEM(space,f_range,r0,q,mu,c0)

p,u = s1.bemsolve()

#%%
pT = s1.grid_evaluate(0,plane,d,grid_size,n_grid_pts,p,u,savename='banana')

#pt_T = s1.point_evaluate(points,p,u)

#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/exterior_eric_close.csv', sep=",", header=None)
data.columns = ["freq","r1","r2","r3"]

plt.semilogx(data.freq,data.r1,linewidth=1)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,0])/2e-5),linewidth=2)
plt.legend(['validarion_r1','bempp_r1'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r1.png',dpi=500)
plt.show()

plt.semilogx(data.freq,data.r2,linewidth=2)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,1])/2e-5),linewidth=2)
plt.legend(['validation_r2','bempp_r2'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r2.png',dpi=500)
plt.show()

plt.semilogx(data.freq,data.r3,linewidth=3)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,2])/2e-5),linewidth=2)

plt.legend(['validarion_r3','bempp_r3'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r3.png',dpi=500)
plt.show()



