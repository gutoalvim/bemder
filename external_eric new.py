#%% Import Packages and define simulation

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
import bemder.porous as porous
from bemder.exterior_api_new import ExteriorBEM

#bempp.api.NumbaDeprecationWarning(highlighting=False)
#bempp.api.NumbaPerformanceWarning(highlighting=False)
#bempp.api.NumbaPendingDeprecationWarning(highlighting=False)
#% Defining constants

c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density





#% Load mesh
filename = 'eric_z.msh'
grid = bempp.api.import_grid('Mshs/'+filename)

#Defining frequencies of analysis 
f1= 1000
f2 = 1000
df = 10
f_range = np.arange(f1,f2+df,df)

#Defining Surface admittance
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

points = {}
# points[0] = np.array([0,0,0.1])
# points[1] = np.array([0,0,0.15])
# points[2] = np.array([0.5,0.5,0.2])
points[3] = np.array([-0.25,0,0.1])



r0 = {}
r0[0] =  np.array([0,0,0.3])
r0[1] =  np.array([0,0,-0.3])

q = {}
q[0] = 1
q[1] = 1

mu_fi = np.array([mu[i] for i in mu.keys()])
r0_fi = np.array([r0[i] for i in r0.keys()])
q_fi = np.array([q[i] for i in q.keys()])

dd =mu_fi[1,1]
#% Defining grid plot properties 
plane = 'xz_c'
d = 0.02

grid_size = [1.5,1.5]
#grid_size = [3.1,3.1]


n_grid_pts = 600

ylimit = [50,120]



#%% Solve BEM

s1 = ExteriorBEM(space,f_range,r0,q,mu)

p,u = s1.helmholtz_bemsolve()
#%%

pT = s1.grid_evaluate(len(f_range)-1,plane,d,grid_size,n_grid_pts,p,u,ylimit)
#%%
pt_T = s1.point_evaluate(points,p,u)



#%% Plot Comparison between Bempp and Validation
from matplotlib import style
style.use("seaborn-talk")

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)

data = pd.read_csv('Data/exterior_eric_beam.csv', sep=",", header=None)
data.columns = ["freq","r1","r2","r3","r4"]
#%%
plt.semilogx(data.freq,data.r1,linewidth=2,linestyle='--',color='tab:blue')

plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,0])/2e-5),linewidth=3,color='tab:blue')

plt.semilogx(data.freq,data.r2,linewidth=2,color='tab:orange')

plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,1])/2e-5),linewidth=3,linestyle='--',color='tab:orange')
plt.grid(axis='y')
plt.legend(['Validarion - R1','Bempp - R1','Validation - R2','Bempp - R2'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r1_r2_gg.png',dpi=500)
plt.xlim([200,1000])
plt.show()
#%%
plt.semilogx(data.freq,data.r1,linewidth=2)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,0])/2e-5),linewidth=3)
plt.legend(['Validarion - R1','Bempp - R1'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r1_r2_gg.png',dpi=500)
plt.xlim([200,1000])
plt.show()

plt.semilogx(data.freq,data.r2,linewidth=2)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,1])/2e-5),linewidth=3)
plt.legend(['Validation - R2','Bempp - R2'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r2-GG.png',dpi=500)
plt.xlim([200,1000])
plt.show()

plt.semilogx(data.freq,data.r3,linewidth=2)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,2])/2e-5),linewidth=3)

plt.legend(['validarion_r3','bempp_r3'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r3.png',dpi=500)
plt.xlim([200,1000])

plt.show()

plt.semilogx(data.freq,data.r4,linewidth=2)
plt.semilogx(f_range, 20*np.log10(np.abs(pt_T[:,3])/2e-5),linewidth=3)

plt.legend(['validarion_r4','bempp_r4'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.savefig('exterior_eric_r4.png',dpi=500)
plt.xlim([200,1000])

plt.show()


