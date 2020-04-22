
import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
from bemder.bem_api_new import ExteriorBEM
import bemder.bem_api_new as BEM
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl
from matplotlib import style
style.use("seaborn-talk")
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 200,2500,500) #Defines frequencies of analysis

AC.freq = [500]# [500,1000,1500,2000,2500]
#% Load mesh

qrd = 'QRD_1D.msh'

qrd = bempp.api.import_grid('Mshs/Diffusers/QRD_Study/'+qrd)


#Defining Sources and Receivers


S = sources.Source("spherical",coord=[200,0,0])
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()


R = receivers.Receiver()
R.arc_receivers(20,90,[-90,90],axis = "z")
R.coord


plane = 'z'
d = 0

grid_size = [2,2]

n_grid_pts = 600

#%%

QRD = ExteriorBEM(qrd,AC,AP,S,R)

#%%

bq = QRD.impedance_bemsolve()
QRD.bem_save("Mshs/Diffusers/QRD_Study/sl_test")


#%%
QRD, bD = BEM.bem_load('Mshs/Diffusers/QRD_Study/sl_test')# ExteriorBEM(qrd,AC,AP,S,R)

# bql = QRD.bem_load("QRD")

bq_pt,bq_ps = QRD.point_evaluate(bD,R=R) 
gfun = QRD.combined_grid_evaluate(bD,0, plane,d,grid_size,n_grid_pts)

#%%

bqr = QRD_REF.bem_load("QRD_REF")
bqa = QRD_ARRAY.bem_load("QRD_ARRAY")
bqar = QRD_ARRAY_REF.bem_load("QRD_ARRAY_REF")

bsc = SC.bem_load("SC")
bscr = SC_REF.bem_load("SC_REF")
bsca = SC_ARRAY.bem_load("SC_ARRAY")
bscar = SC_ARRAY_REF.bem_load("SC_ARRAY_REF")

#%%

bq_pt,bq_ps = QRD.point_evaluate(bq,R) 
bqr_pt, bqr_ps = QRD_REF.point_evaluate(bqr,R) 
# bqa_pt, bqa_ps = QRD_ARRAY.point_evaluate(bqa,R) 
# bqra_pt, bqra_ps = QRD_ARRAY_REF.point_evaluate(bqar,R)



# bsc_pt, bsc_ps = SC.point_evaluate(bsc,R)
# bscr_pt, bscr_ps = SC_REF.point_evaluate(bscr,R)
# bsca_pt, bsca_ps = SC_ARRAY.point_evaluate(bsca,R)
# bscar_pt, bscar_ps = SC_ARRAY_REF.point_evaluate(bscar,R)

#%%

data_1D = pd.read_csv('Data/QRD_1D.csv', sep=",", header=None)
data_1D.columns = ["Lp"]

data_ref = pd.read_csv('Data/QRD_ref.csv', sep=",", header=None)
data_ref.columns = ["Lp"]

plt.polar(R.theta, 20*np.log10(np.abs(bq_ps)/2e-5).reshape(len(R.theta),1))
plt.polar(R.theta, data_1D.Lp+3)

plt.show()

# plt.polar(R.theta, 20*np.log10(np.abs(bqr_ps)/2e-5).reshape(len(R.theta),1))
# plt.polar(R.theta, data_ref.Lp+3)

# plt.show()
#%%

