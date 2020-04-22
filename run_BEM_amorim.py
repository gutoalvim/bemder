#%% Import Packages and define simulation

import matplotlib.pyplot as plt
from matplotlib import style
style.use("seaborn-talk")
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.interior_api import RoomBEM

bempp.api.PLOT_BACKEND = "gmsh"

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density

#% Load mesh and import
filename = 'amorim_mesh.msh'
grid = bempp.api.import_grid('Mshs/'+filename)
space = bempp.api.function_space(grid, "DP", 0)

#Defining frequencies of analysis 
f1= 20
f2 = 200
df = 1
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.ones_like(f_range)*0.02


#Atribute admittance to every domain index
mu = {}
mu[1] = muh


#Receiver coords
points = {}
points[0] = np.array([1.505,0.5,1.14])
#points[1] = np.array([0.6,0.2,-0.15])

#Source coords
r0 = {}
r0[0] =  np.array([2,-0.3,1.1])
r0[1] = np.array([0.95,-0.3,1.1])

#Source Strength
q = {}
q[0] = 1
q[1] = 1

#% Defining grid plot properties 
plane = 'xyc'
d = 1.14
grid_size = [3,4]
n_grid_pts = 250

#%% Solve BEM
s1 = RoomBEM(space,f_range,r0,q,mu,c0)
p,u = s1.bemsolve()
#%% Plot Pressure Field
gplot = s1.grid_evaluate(80,plane,d,grid_size,n_grid_pts,p,u,savename='johnsons')

#%% Calculate pressure for Receivers (Points)
pT = s1.point_evaluate(points,p,u)

#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/amorim_undamped.csv', sep=",", header=None)
data.columns = ["freq","spl"]

err = np.abs((np.array([data.spl]).reshape(len(pT),1) - 20*np.log10(np.abs(pT)/2e-5)).mean(axis=1))

plt.plot(f_range, 20*np.log10(np.abs(pT)/2e-5))
#plt.plot(f_range, np.real(pT))
plt.plot(data.freq,data.spl-40)
plt.legend(['bempp','validation'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('topology_r0_r1_cmplx_SPL.png', dpi=500)
plt.show()


plt.plot(f_range, err)

plt.legend(['bempp','validation'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Error [dB]')
plt.savefig('topology_r0_r1_err.png', dpi=500)
plt.show()

