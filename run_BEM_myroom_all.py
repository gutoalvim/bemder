#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.room_api import RoomBEM

#% Defining constants
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density


#% Load mesh
filename = 'myroom_all.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#Defining frequencies of analysis 
f1= 20
f2 = 150
df = 2
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
muh = np.zeros_like(f_range)
muD = np.ones_like(f_range)
zsd1 = porous.delany(10000,0.05,f_range)
zsd2 = porous.delany(10000,0.1,f_range)
mud1 = np.complex128(rho0*c0/np.conj(zsd1))
mud2 = np.complex128(rho0*c0/np.conj(zsd2))

mu = {}

mu[1] = muh
mu[2] = mud1
mu[3] = mud2
mu[4] = muD


points = {}
points[0] = np.array([0,1,-0.7])
#points[1] = np.array([-0.7,1,-1.4])


r0 = {}
r0[0] =  np.array([0.7,1,-1.4])
r0[1] = np.array([-0.7,1,-1.4])

q = {}
q[0] = 1
q[1] = 1

#% Defining grid plot properties 
plane = 'xz'
d = 0

grid_size = [3.5,4]

n_grid_pts = 250



space = bempp.api.function_space(grid, "DP", 0)


#%% Solve BEM

s1 = RoomBEM(space,f_range,r0,q,mu,c0)

p,u = s1.bemsolve()

pT = s1.point_evaluate(points,p,u)

#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/my_room_all.csv', sep=",", header=None)
data.columns = ["freq","spl","arg"]

mse = np.sqrt(((np.array(20*np.log10(np.abs(pT)/2e-5)) - np.array(data.spl[0:66]).reshape(66,1))**2).mean(axis=0))

plt.plot(f_range, 20*np.log10(np.abs(pT)/2e-5))
#plt.plot(f_range, np.real(pT))
plt.plot(data.freq,data.spl)
plt.title('FRF Magnitide|Sala irregular|Admitância cmplx')
plt.legend(['bempp','validation'])
plt.xlabel('Frequência [Hz]')
plt.ylabel('NPS [dB]')
plt.savefig('my_room_all_SPL.png', dpi=500)
plt.show()

plt.plot(f_range, np.angle(pT))
#plt.plot(f_range, np.real(pT))
plt.plot(data.freq,data.arg)
plt.title('FRF Fase|Sala irregular|Admitância cmplx')
plt.legend(['bempp','validation'])
plt.xlabel('Frequência [Hz]')
plt.ylabel('Fase [rad]')
plt.savefig('my_room_all_phase.png', dpi=500)
plt.show()

