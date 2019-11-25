#%% Import Packages and define simulation
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.room_api import RoomBEM
import matplotlib.pyplot as plt
from matplotlib import style
style.use("seaborn-talk")

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density


#% Load mesh
filename = 'my_room_comsol.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#Defining frequencies of analysis 
f1= 20
f2 = 100
df = 10
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)
zsd1 = porous.delany(5000,0.1,f_range)
zsd2 = porous.delany(10000,0.2,f_range)
zsd3 = porous.delany(15000,0.3,f_range)
mud1 = np.complex128(rho0*c0/np.conj(zsd1))
mud2 = np.complex128(rho0*c0/np.conj(zsd2))
mud3 = np.complex128(rho0*c0/np.conj(zsd3))

mu = {}

mu[1] = np.zeros_like(mud2)
#mu[2] = mud2
#mu[3] = mud3


points = {}
points[0] = np.array([0.6,0,-0.15])
points[1] = np.array([0.6,0.2,-0.15])


r0 = {}
r0[0] =  np.array([1.4,0.7,-0.35])
r0[1] = np.array([1.4,-0.7,-0.35])

q = {}
q[0] = 1
q[1] = 1


#% Defining grid plot properties 
plane = 'xy'
d = -0.15

grid_size = [3.7,3]

n_grid_pts = 250



space = bempp.api.function_space(grid, "DP", 0)


#%% Solve BEM

s1 = RoomBEM(space,f_range,r0,q,mu,c0)

p,u = s1.bemsolve()
#%%
pT = s1.grid_evaluate(0,'xy',d,grid_size,n_grid_pts,p,u)

pTT = s1.point_evaluate(points,p,u)

#%%

plt.plot(f_range,20*np.log10(abs(pTT)/2e-5))
plt.legend(['bempp','bempp2'])
plt.xlabel('freq [Hz]')
plt.ylabel('spl[dB]')
