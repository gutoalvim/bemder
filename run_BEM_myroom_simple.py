#%% Import Packages and define simulation
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.interior_api import RoomBEM
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use("seaborn-talk")

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density


#% Load mesh
filename = 'my_room_simple.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#Defining frequencies of analysis 
f1= 20
f2 = 150
df = 2
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)


mu = {}

mu[1] = muh
#mu[2] = mud2
#mu[3] = mud3


points = {}
#points[0] = np.array([0.6,0,-0.15])
#points[1] = np.array([0.6,0.2,-0.15])

points[0] = np.array([0.6,0,-0.15])
points[1] = np.array([0.3,0.5,-0.15])
points[2] = np.array([-1,-0.5,0.3])



r0 = {}
r0[0] =  np.array([1.4,0.7,-0.35])
r0[1] = np.array([1.4,-0.7,-0.35])

# r0[0] =  np.array([0.62,1,-0.35])
# r0[1] = np.array([-0.62,1,-0.35])

q = {}
q[0] = 1
q[1] = 1


#% Defining grid plot properties 
plane = 'xy'
d = -0.2

grid_size = [3.7,3]

n_grid_pts = 250



space = bempp.api.function_space(grid, "DP", 0)


#%% Solve BEM

s1 = RoomBEM(space,f_range,r0,q,mu,c0)
import time
then = time.time()
p,u = s1.bemsolve()
now = time.time()
print("It took: ", now-then, " seconds")
#%%
pT = s1.grid_evaluate(26,'xy',d,grid_size,n_grid_pts,p,u,savename='damn')
#%%
pTT = s1.point_evaluate(points,p,u)

#%%

data = pd.read_csv('Data/my_room_simple.csv', sep=",", header=None)
data.columns = ["freq","r3","r2","r1"]

darg = pd.read_csv('Data/my_room_simple_arg.csv', sep=",", header=None)
darg.columns = ["freq","r3","r2","r1"]


plt.plot(f_range,20*np.log10(abs(pTT[:,0])/2e-5))
plt.plot(data.freq,data.r1)
plt.legend(['bempp','validation'])
plt.xlabel('freq [Hz]')
plt.ylabel('spl[dB]')
plt.savefig('splr1.png', dpi=500)

plt.show()

plt.plot(f_range,np.abs(np.angle(pTT[:,0])))
plt.plot(darg.freq,np.abs(darg.r1))
plt.legend(['bempp','validation'])
plt.xlabel('freq [Hz]')
plt.ylabel('spl[dB]')
plt.savefig('absphaser1.png', dpi=500)

plt.show()

plt.plot(data.freq,data.r3)
plt.plot(f_range,20*np.log10(abs(pTT[:,2])/2e-5))
plt.legend(['bempp','validation'])
plt.xlabel('freq [Hz]')
plt.ylabel('spl[dB]')
plt.savefig('splr2.png', dpi=500)

plt.show()

plt.plot(f_range,np.abs(np.angle(pTT[:,2])))
plt.plot(darg.freq,np.abs(darg.r3))
plt.legend(['bempp','validation'])
plt.xlabel('freq [Hz]')
plt.ylabel('spl[dB]')
plt.savefig('absphaser2.png', dpi=500)

plt.show()


