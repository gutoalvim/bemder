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

qrd = 'QRD_1D.msh'
qrd_ref = 'QRD_1D_Ref.msh'
qrd_array = 'QRD_1D_Array.msh'
qrd_array_ref = 'QRD_1D_Array_Ref.msh'

qrd = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd)
qrd_ref = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd_ref)
qrd_array = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd_array)
qrd_array_ref = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd_array_ref)

sc = 'sc.msh'
sc_ref = 'sc_ref.msh'
sc_array = 'sc_array.msh'
sc_array_ref = 'sc_array_ref.msh'

sc = bempp.api.import_grid('Mshs/Diffusers/Semi_Cylinder/'+sc)
sc_ref = bempp.api.import_grid('Mshs/Diffusers/Semi_Cylinder/'+sc_ref)
sc_array = bempp.api.import_grid('Mshs/Diffusers/Semi_Cylinder/'+sc_array)
sc_array_ref = bempp.api.import_grid('Mshs/Diffusers/Semi_Cylinder/'+sc_array_ref)


#Defining Sources and Receivers

S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()


R = receivers.Receiver()
R.arc_receivers(20,90,[-90,90],axis = "z")
R.coord


#%%

# QRD = ExteriorBEM(qrd,AC,AP,S,R)
# QRD_REF = ExteriorBEM(qrd_ref,AC,AP,S,R)
# QRD_ARRAY = ExteriorBEM(qrd_array,AC,AP,S,R)
# QRD_ARRAY_REF = ExteriorBEM(qrd_array_ref,AC,AP,S,R)

# SC = ExteriorBEM(sc,AC,AP,S,R)
# SC_REF = ExteriorBEM(sc_ref,AC,AP,S,R)
# SC_ARRAY = ExteriorBEM(sc_array,AC,AP,S,R)
# SC_ARRAY_REF = ExteriorBEM(sc_array_ref,AC,AP,S,R)
#%%

# bq = QRD.impedance_bemsolve()
# QRD.bem_save("Mshs/Diffusers/QRD_Study/QRD")
# bqr = QRD_REF.impedance_bemsolve()
# QRD_REF.bem_save("Mshs/Diffusers/QRD_Study/QRD_REF")
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
QRD_ARRAY,bqa = bem_load("Mshs/Diffusers/QRD_Study/QRD_ARRAY")
QRD_ARRAY_REF, bqar = bem_load("Mshs/Diffusers/QRD_Study/QRD_ARRAY_REF")

SC,bsc = bem_load("Mshs/Diffusers/Semi_Cylinder/SC")
SC_REF,bscr = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_REF")
SC_ARRAY,bsca = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY")
SC_ARRAY_REF,bscar = bem_load("Mshs/Diffusers/Semi_Cylinder/SC_ARRAY_REF")

#%%

bq_pt,bq_ps = QRD.point_evaluate(bq,R) 
bqr_pt, bqr_ps = QRD_REF.point_evaluate(bqr,R) 
bqa_pt, bqa_ps = QRD_ARRAY.point_evaluate(bqa,R) 
bqra_pt, bqra_ps = QRD_ARRAY_REF.point_evaluate(bqar,R)



bsc_pt, bsc_ps = SC.point_evaluate(bsc,R)
bscr_pt, bscr_ps = SC_REF.point_evaluate(bscr,R)
bsca_pt, bsca_ps = SC_ARRAY.point_evaluate(bsca,R)
bscar_pt, bscar_ps = SC_ARRAY_REF.point_evaluate(bscar,R)

#%%

data_1D = pd.read_csv('Data/QRD_1D.csv', sep=",", header=None)
data_1D.columns = ["Lp"]

data_ref = pd.read_csv('Data/QRD_ref.csv', sep=",", header=None)
data_ref.columns = ["Lp"]

plt.polar(R.theta, 20*np.log10(np.abs(bq_ps)/2e-5).reshape(len(R.theta),1))
plt.polar(R.theta, data_1D.Lp+3)

plt.show()

plt.polar(R.theta, 20*np.log10(np.abs(bqr_ps)/2e-5).reshape(len(R.theta),1))
plt.polar(R.theta, data_ref.Lp+3)

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