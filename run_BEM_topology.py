#%% Import Packages and define simulation

import matplotlib.pyplot as plt
from matplotlib import style
style.use("seaborn-talk")
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.bem_api_new import RoomBEM
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl


filename = 'topology.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 20,150,2) #Defines frequencies of analysis
# AC.freq = [1000]


#Defining Surface admittance
muh = np.zeros_like(AC.freq)
zsd1 = porous.delany(5000,0.1,AC.freq)
zsd2 = porous.delany(10000,0.2,AC.freq)
zsd3 = porous.delany(15000,0.3,AC.freq)
mud1 = np.complex128(AP.rho0*AP.c0/np.conj(zsd1))
mud2 = np.complex128(AP.rho0*AP.c0/np.conj(zsd2))
mud3 = np.complex128(AP.rho0*AP.c0/np.conj(zsd3))


#Atribute admittance to every domain index
mu = {}
mu[1] = mud1
mu[2] = mud2
mu[3] = mud3

#Receiver coords
R = receivers.Receiver(coord=[0.5,1.5,1.2])
# points[0] = np.array([0.5,1.5,1.2])
#points[1] = np.array([0.6,0.2,-0.15])

#Source coords
S =sources.Source("spherical",coord=[-0.5,4,1.2])

#% Defining grid plot properties 
plane = 'z'
d = 1.2
grid_size = [6,6]
n_grid_pts = 250

#%% Solve BEM
s1 = RoomBEM(grid,AC,AP,S,R,mu)
boundD = s1.bemsolve()
#%% Plot Pressure Field
gplot = s1.combined_grid_evaluate(boundD,0,plane,d,grid_size,n_grid_pts)
#%% Calculate pressure for Receivers (Points)
pT,pS = s1.point_evaluate(R,boundD)


#%% Plot Comparison between Bempp and Validation
data = pd.read_csv('Data/topology_cmplx_r1.csv', sep=",", header=None)
data.columns = ["freq","spl","arg"]

# err = np.abs((np.array([data.spl]).reshape(len(pT),1) - 20*np.log10(np.abs(pT)/2e-5)).mean(axis=1))

plt.plot(AC.freq, 20*np.log10(np.abs(pT)/2e-5))
#plt.plot(f_range, np.real(pT))
plt.plot(data.freq,data.spl)
plt.legend(['bempp','validation'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('topology_r0_r1_cmplx_SPL.png', dpi=500)
plt.show()

# plt.plot(f_range, np.angle(pT))
# #plt.plot(f_range, np.real(pT))
# plt.plot(data.freq,data.arg)
# plt.legend(['bempp','validation'])
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Phase [rad]')
# plt.savefig('topology_r0_r1_cmplx_phase.png', dpi=500)
# plt.show()

# plt.plot(f_range, err)

# plt.legend(['bempp','validation'])
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Error [dB]')
# plt.savefig('topology_r0_r1_err.png', dpi=500)
# plt.show()

