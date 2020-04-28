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
style.use("seaborn-paper")

def zh(d):
    Y = 1j*AP.rho0*AP.c0/np.tan(2*np.pi*np.array(AC.freq,dtype=np.float)*d/AP.c0)
    return (1/Y)
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 200,2500,500) #Defines frequencies of analysis

AC.freq = [250,315,400,500,630,800,1000,1250,1600,2000,2500]
#% Load mesh

qrd = 'QRD_1D.msh'
qrd_ref = 'QRD_1D_Ref.msh'
qrd_z = 'QRD_Z.msh'


qrd = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd)
qrd_ref = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd_ref)
qrd_Z = bempp.api.import_grid('Mshs/Diffusers/Impedance/'+qrd_z)

q1 = 0.027
q3 = 0.08
q4 = 0.106
q5 = 0.133
q7 = 0.186

mu = {}
mu[1] = zh(q1)
mu[3] = zh(q3)
mu[4] = zh(q4)
mu[5] = zh(q5)
mu[7] = zh(q7)
mu[9] = np.zeros_like(AC.freq)

#Defining Sources and Receivers

S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()


R = receivers.Receiver()
R.arc_receivers(20,90,[-90,90],axis = "z")
R.coord


#%%

QRD = ExteriorBEM(qrd,AC,AP,S,R)
QRD_REF = ExteriorBEM(qrd_ref,AC,AP,S,R)
QRD_Z= ExteriorBEM(qrd_Z,AC,AP,S,R)
# QRD_ARRAY_REF = ExteriorBEM(qrd_array_ref,AC,AP,S,R)

# SC = ExteriorBEM(sc,AC,AP,S,R)
# SC_REF = ExteriorBEM(sc_ref,AC,AP,S,R)
# SC_ARRAY = ExteriorBEM(sc_array,AC,AP,S,R)
# SC_ARRAY_REF = ExteriorBEM(sc_array_ref,AC,AP,S,R)

#%%
bqz = QRD_Z.impedance_bemsolve()
QRD_Z.bem_save('Mshs/Diffusers/Impedance/QRD_Z_sol')
#%%

bq = QRD.impedance_bemsolve()
QRD.bem_save("Mshs/Diffusers/QRD_Study/QRD_more")
bqr = QRD_REF.impedance_bemsolve()
QRD_REF.bem_save("Mshs/Diffusers/QRD_Study/QRD_REF_more")
# bqa = QRD_ARRAY.impedance_bemsolve()
# QRD_ARRAY.bem_save("Mshs/Diffusers/QRD_Study/QRD_ARRAY")
# bqar = QRD_ARRAY_REF.impedance_bemsolve()
# QRD_ARRAY_REF.bem_save("Mshs/Diffusers/QRD_Study/QRD_ARRAY_REF")

# bsc = SC.impedance_bemsolve()
# SC.bem_save("Mshs/Diffusers/Semi_Cylinder/SC")
# bscr = SC_REF.impedance_bemsolve()
# SC_REF.bem_save("Mshs/Diffusers/Semi_Cylinder/SC_REF")
# bsca = SC_ARRAY.impedance_bemsolve()
# SC_ARRAY.bem_save("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY")
# bscar = SC_ARRAY_REF.impedance_bemsolve()
# SC_ARRAY_REF.bem_save("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY_REF")

#%%
# QRD = ExteriorBEM(qrd,AC,AP,S,R)

# bql = QRD.bem_load("QRD")

# bq_pt,bq_ps = QRD.point_evaluate(boundD=bql,R=R) 

#%%
QRD,bq = bem_load("Mshs/Diffusers/QRD_Study/QRD")
QRD_REF,bqr = bem_load("Mshs/Diffusers/QRD_Study/QRD_REF")
QRZ,bqz = bem_load('Mshs/Diffusers/Impedance/QRD_Z_sol')
QRD_ARRAY,bqa = bem_load("Mshs/Diffusers/QRD_Study/QRD_ARRAY")
QRD_ARRAY_REF, bqar = bem_load("Mshs/Diffusers/QRD_Study/QRD_ARRAY_REF")

# SC,bsc = bem_load("Mshs/Diffusers/Semi_Cylinder/SC")
# SC_REF,bscr = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_REF")
# SC_ARRAY,bsca = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY")
# SC_ARRAY_REF,bscar = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY_REF")

#%%

bq_pt,bq_ps = QRD.point_evaluate(bq,R) 
bqr_pt, bqr_ps = QRD_REF.point_evaluate(bqr,R) 
bqz_pt,bqz_ps = QRD_Z.point_evaluate(bqz,R)
bqa_pt, bqa_ps = QRD_ARRAY.point_evaluate(bqa,R) 
bqra_pt, bqra_ps = QRD_ARRAY_REF.point_evaluate(bqar,R)

#%%
i = 0

plt.polar(R.theta, 20*np.log10(np.abs(bq_ps[i,:])/2e-5).reshape(len(R.theta),1))
# plt.polar(R.theta, data_1D.Lp+3)

plt.show()

# plt.polar(R.theta, 20*np.log10(np.abs(bqr_ps)/2e-5).reshape(len(R.theta),1))
# plt.polar(R.theta, data_ref.Lp+3)

# plt.show()

plt.polar(R.theta, 20*np.log10(np.abs(bqz_ps[i,:])/2e-5).reshape(len(R.theta),1))

plt.show()

#%%

# plt.polar(R.theta, 20*np.log10(np.abs(bq_ps[i,:])/2e-5).reshape(len(R.theta),1))


plt.polar(R.theta, 20*np.log10(np.abs(bq_ps[i,:])/2e-5).reshape(len(R.theta),1))

# plt.show()

# plt.polar(R.theta, 20*np.log10(np.abs(bqa_ps[i,:])/2e-5).reshape(len(R.theta),1))


plt.polar(R.theta, 20*np.log10(np.abs(bqa_ps[i,:])/2e-5).reshape(len(R.theta),1))

plt.show()
#%%
i = 
plt.polar(R.theta, 20*np.log10(np.abs(bsc_ps[i,:])/2e-5).reshape(len(R.theta),1))


# plt.polar(R.theta, 20*np.log10(np.abs(bscr_ps[i,:])/2e-5).reshape(len(R.theta),1))
# 
# plt.show()

plt.polar(R.theta, 20*np.log10(np.abs(bsca_ps[i,:])/2e-5).reshape(len(R.theta),1))


# plt.polar(R.theta, 20*np.log10(np.abs(bscar_ps[i,:])/2e-5).reshape(len(R.theta),1))

plt.show()

#%%
bplt.polar_plot(R.theta,bq_ps,normalize=True, transformation="dB",title = "QRD Rigid SPL - mic_20m - r0_200m")
bplt.polar_plot(R.theta,bqr_ps,normalize=True, transformation="dB",title = "QRD Rigid SPL - mic_20m - r0_200m")

#%%

Tfq = hh.diffusion_coef(AC.freq, bq_ps,bqr_ps,plot=False)
Tfqz = hh.diffusion_coef(AC.freq, bqz_ps,bqr_ps,plot=False)

sfq = hh.scattering_coef(AC.freq, bq_ps,bqr_ps,plot=False)
sfqz = hh.scattering_coef(AC.freq, bqz_ps,bqr_ps,plot=False)


Treflex = [0,0,0.69,0.6,0.22,0.27,0.8,0.65,0.53,0.45,0.5]
Tqca = [0.01,0.07,0.12,0.07,0.39]
# Tfqa = hh.diffusion_coef(AC.freq, bqa_ps,bqra_ps,plot=False)

fig, ax = plt.subplots()
ax.set_ylim(0,1)
ax.semilogx(AC.freq,Tfq)
ax.semilogx(AC.freq,Tfqz)
ax.set_xticklabels(AC.freq)
ax.set_xticks(AC.freq)


# ax.plot(AC.freq,Treflex)
plt.legend(('QRD 1D - 1 Period - Drawn Geometry','QRD 1D - 1 Period - Impedance Representation'))
plt.title('QRD 1D - Normal Incidence Diffusion Coefficient')
plt.savefig('QRD_1D_array_Tf_comp.png',dpi=500)

plt.show()


fig, ax = plt.subplots()
ax.set_ylim(0,1)
ax.semilogx(AC.freq,sfq)
ax.semilogx(AC.freq,sfqz)
ax.set_xticklabels(AC.freq)
ax.set_xticks(AC.freq)
# ax.plot(AC.freq,Treflex)
plt.legend(('QRD 1D - 1 Period - Drawn Geometry','QRD 1D - 1 Period - Impedance Representation'))
plt.title('QRD 1D - Normal Incidence Scattering Coefficient')
plt.savefig('QRD_1D_array_s_comp.png',dpi=500)
plt.show()


# fig, ax = plt.subplots()
# ax.set_ylim(0,1)
# ax.semilogx(freq,Tfq)
# ax.semilogx(AC.freq,Tfqa)
# ax.set_xticklabels(AC.freq)
# ax.set_xticks(AC.freq)
# # ax.plot(AC.freq,Treflex)
# plt.legend(('QRD 1D - 1 Period - Drawn Geometry','QRD 1D - 3 Periods - Drawn Geometry'))
# plt.title('QRD 1D - Normal Incidence Diffusion Coefficient')
# plt.show()

#%%

