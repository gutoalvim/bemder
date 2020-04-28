# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:34:25 2020

@author: gutoa
"""


import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
from bemder.bem_api_new import ExteriorBEM
import bemder.plot as bplt
import bemder.helpers as hh
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl
from bemder.bem_api_new import bem_load
from matplotlib import style
style.use("seaborn-talk")
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 200,2500,500) #Defines frequencies of analysis

AC.freq = [500,1000,1500,2000,2500]
#% Load mesh


sc_ref = 'sc_ref.msh'
sc_array = 'sc_array.msh'


sc_ref = bempp.api.import_grid('Mshs/Diffusers/SC_Hassan/'+sc_ref)
sc_array = bempp.api.import_grid('Mshs/Diffusers/SC_Hassan/'+sc_array)


#Defining Sources and Receivers

S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()


R = receivers.Receiver()
R.arc_receivers(20,90,[-90,90],axis = "z")
R.coord


#%%

SC_ARRAY = ExteriorBEM(sc_array,AC,AP,S,R)
SC_ARRAY_REF = ExteriorBEM(sc_ref,AC,AP,S,R)
#%%

bsca = SC_ARRAY.impedance_bemsolve()
SC_ARRAY.bem_save("Mshs/Diffusers/SC_Hassan/SC_ARRAY")
bscar = SC_ARRAY_REF.impedance_bemsolve()
SC_ARRAY_REF.bem_save("Mshs/Diffusers/SC_Hassan/SC_ARRAY_REF")

#%%
# QRD = ExteriorBEM(qrd,AC,AP,S,R)

# bql = QRD.bem_load("QRD")

# bq_pt,bq_ps = QRD.point_evaluate(boundD=bql,R=R) 

#%%


SC_ARRAY,bsca = bem_load("Mshs/Diffusers/SC_Hassan/SC_ARRAY")
SC_ARRAY_REF,bscar = bem_load("Mshs/Diffusers/SC_Hassan/SC_ARRAY_REF")

#%%

bsca_pt, bsca_ps = SC_ARRAY.point_evaluate(bsca,R)
bscar_pt, bscar_ps = SC_ARRAY_REF.point_evaluate(bscar,R)

#%%


#%%
bplt.polar_plot(R.theta,bq_ps,normalize=True, transformation="dB",title = "QRD Rigid SPL - mic_20m - r0_200m")
bplt.polar_plot(R.theta,bqr_ps,normalize=True, transformation="dB",title = "QRD Rigid SPL - mic_20m - r0_200m")

#%%
data = pd.read_csv('Data/sc_Lps.csv', sep=",", header=None)
data.columns = ["Lp"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(R.theta, data.Lp+3 - max(data.Lp+3),label="Validation")
ax.set_thetamax(90)
ax.set_thetamin(-90)


plt.polar(R.theta, 20*np.log10(np.abs(bsca_ps[4,:])/2e-5).reshape(len(R.theta),1)-max(20*np.log10(np.abs(bsca_ps[4,:])/2e-5).reshape(len(R.theta),1)),color='tab:orange',label="Bempp")
plt.legend(loc="top left")
# plt.polar(R.theta, data.Lp+3,marker='x')
plt.savefig("SC_Validation.png",dpi=500)

plt.show()
# plt.ylim([-35,1])
err = 20*np.log10(np.abs(bsca_ps[4,:])/2e-5) - (data.Lp+3)

plt.plot(180*R.theta/np.pi,err)

#%%

Tfq = hh.diffusion_coef(AC.freq, bsca_ps,bscar_ps,plot=False)

sfq = hh.scattering_coef(AC.freq, bsca_ps,bscar_ps,plot=False)


# Treflex = [0,0,0.69,0.6,0.22,0.27,0.8,0.65,0.53,0.45,0.5]
# Tqca = [0.01,0.07,0.12,0.07,0.39]
# Tfqa = hh.diffusion_coef(AC.freq, bqa_ps,bqra_ps,plot=False)
Tred = [0.02,0.05,0.15,0.02,0.1]
fig, ax = plt.subplots()
ax.set_ylim(0,1)
ax.semilogx(AC.freq,Tfq)
ax.semilogx(AC.freq,Tred)
ax.set_xticklabels(AC.freq)
ax.set_xticks(AC.freq)


# ax.plot(AC.freq,Treflex)
plt.legend(('Bempp ','Reflex'))
plt.title('Semi Cylinder- Normal Incidence Diffusion Coefficient')
plt.savefig('sc_Tf_comp.png',dpi=500)

plt.show()

sred = [0.03,0.2,0.93,0.15,0.98]
fig, ax = plt.subplots()
ax.set_ylim(0,1)
ax.semilogx(AC.freq,sfq)
ax.semilogx(AC.freq,sred)

ax.set_xticklabels(AC.freq)
ax.set_xticks(AC.freq)
# ax.plot(AC.freq,Treflex)
plt.legend(('Bempp ','Reflex'))
plt.title('Semi Cylinder- Normal Incidence Scattering Coefficient')
plt.savefig('sc_array_s_comp.png',dpi=500)
plt.show()