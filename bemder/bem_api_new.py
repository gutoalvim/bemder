#%%
import time
import warnings
import bempp.api
import numpy as np
import numba
from bemder import controlsair as ctrl
from bemder import sources
from bemder import receivers
from bemder import helpers
from matplotlib import pylab as plt
import cloudpickle
import collections
bempp.api.PLOT_BACKEND = "gmsh"

warnings.filterwarnings('ignore')

 

#bempp.api.GLOBAL_PARAMETERS.assembly.dense


#%%
def bem_load_legacy(filename,ext='.pickle'):
    
        import pickle
        
        infile = open(filename + ext, 'rb')
        simulation_data = pickle.load(infile)
        # simulation_data = ['simulation_data']
        infile.close()
        # Loading simulation data
        gridpack = bempp.api.grid.Grid(simulation_data['grid']['vertices'],
                    simulation_data['grid']['elements'],
                    simulation_data['grid']['domain_indices'])
        # self.simulation = Simulation(simulation_data['frequencies'], self.admittance, grid)
        #     self.set_SR()
        AP = simulation_data['AP']
        # pts = simulation_data['receivers']
        AC = simulation_data['AC']
        S = simulation_data["S"]
        R = simulation_data["R"]
        mu = simulation_data["mu"]
        # self.set_status = True
        BC = simulation_data["BC"]
        EoI = simulation_data["EoI"]
        IS = simulation_data["IS"]
        v = simulation_data["v"]
        # IS=0
        
        # if simulation_data["BC"] == "robin" or "dirichlet": 
        #     self.space = bempp.api.function_space(self.grid, "DP", 0)
        # elif simulation_data["BC"] == "neumann":
        #     self.space = bempp.api.function_space(self.grid, "P", 1)
        # self.simulation._incident_traces = []
        # self.simulation._incident_fields = []
        space = bempp.api.function_space(gridpack, "DP", 0)
        # bData = {}
        bbd = {}
        if IS==1:
            for sol in range(len(AC.freq)):
                i=0
                bData = {}

                for i in range(len(S.coord)):
                
                    u = bempp.api.GridFunction(space, coefficients=simulation_data['bbd'][sol][i][0])
                    un = bempp.api.GridFunction(space, coefficients=simulation_data['bbd'][sol][i][1])
                    # incident_traces = GridFunction(self.simulation._space, coefficients=simulation_data['incident_traces'][sol])
                    bData[i] = [u, un]
                bbd[sol] = bData
        elif IS==0:
            for sol in range(len(AC.freq)):
                u = bempp.api.GridFunction(space, coefficients=simulation_data['bd'][sol][0])
                un = bempp.api.GridFunction(space, coefficients=simulation_data['bd'][sol][1])
                # incident_traces = GridFunction(self.simulation._space, coefficients=simulation_data['incident_traces'][sol])
                bData[sol] = [u, un]
        

            # self.simulation._incident_traces.append(incident_traces)
        # for frequency, mu in zip(self.simulation._frequencies, self.simulation._admittance_factors):
        #     wavenumber = 2 * np.pi * frequency / 343
        #     _, uinc = self.simulation._compute_incident_trace_and_field(wavenumber, mu, grid=False, incident=True)
        #     self.simulation._incident_fields.append(uinc)
        if EoI ==0:
            print('\tBEM loaded successfully.')
            if IS==1:
                return ExteriorBEM(gridpack,AC,AP,S,R,mu,v=v,IS=IS), bbd
            if IS==0:
                return ExteriorBEM(gridpack,AC,AP,S,R,mu,v=v,IS=IS), bData
          
        if EoI ==1:
            return RoomBEM(gridpack,AC,AP,S,R,mu),bData
        
def bem_load(filename,ext='.pickle'):
    
        import pickle
        
        infile = open(filename + ext, 'rb')
        simulation_data = pickle.load(infile)
        # simulation_data = ['simulation_data']
        infile.close()
        # Loading simulation data
        gridpack = bempp.api.grid.Grid(simulation_data['grid']['vertices'],
                    simulation_data['grid']['elements'],
                    simulation_data['grid']['domain_indices'])
        # self.simulation = Simulation(simulation_data['frequencies'], self.admittance, grid)
        #     self.set_SR()
        AP = simulation_data['AP']
        # pts = simulation_data['receivers']
        sAC = simulation_data['AC']
        S = simulation_data["S"]
        R = simulation_data["R"]
        mu = simulation_data["mu"]
        # self.set_status = True
        BC = simulation_data["BC"]
        EoI = simulation_data["EoI"]
        IS = simulation_data["IS"]
        
        # if simulation_data["BC"] == "robin" or "dirichlet": 
        #     self.space = bempp.api.function_space(self.grid, "DP", 0)
        # elif simulation_data["BC"] == "neumann":
        #     self.space = bempp.api.function_space(self.grid, "P", 1)
        # self.simulation._incident_traces = []
        # self.simulation._incident_fields = []
        space = bempp.api.function_space(gridpack, "DP", 0)
        
        bbd = {}
        if IS==1:
            for sol in range(len(sAC.freq)):
                i=0
                bData = {}
                
                for i in range(len(S.coord)):
                
                    u = bempp.api.GridFunction(space, coefficients=simulation_data['bbd'][sol][i][0])
                    un = bempp.api.GridFunction(space, coefficients=simulation_data['bbd'][sol][i][1])
                    # incident_traces = GridFunction(self.simulation._space, coefficients=simulation_data['incident_traces'][sol])
                    bData[i] = [u, un]
                bbd[sol] = bData
        elif IS==0:
            bData = {}

            for sol in range(len(sAC.freq)):
                u = bempp.api.GridFunction(space, coefficients=simulation_data['bd'][sol][0])
                un = bempp.api.GridFunction(space, coefficients=simulation_data['bd'][sol][1])
                # incident_traces = GridFunction(self.simulation._space, coefficients=simulation_data['incident_traces'][sol])
                bData[sol] = [u, un]
        

            # self.simulation._incident_traces.append(incident_traces)
        # for frequency, mu in zip(self.simulation._frequencies, self.simulation._admittance_factors):
        #     wavenumber = 2 * np.pi * frequency / 343
        #     _, uinc = self.simulation._compute_incident_trace_and_field(wavenumber, mu, grid=False, incident=True)
        #     self.simulation._incident_fields.append(uinc)
        if EoI ==0:
            print('\tBEM loaded successfully.')
            if IS==1:
                return ExteriorBEM(gridpack,sAC,AP,S,R,mu,IS=IS), bbd
            if IS==0:
                return ExteriorBEM(gridpack,sAC,AP,S,R,mu,IS=IS), bData
          
        if EoI ==1:
            return RoomBEM(gridpack,AC,AP,S,R,mu),bData

class ExteriorBEM:
    bempp.api.DEVICE_PRECISION_CPU = 'single'    
    """
    Hi, this class contains some tools to solve the interior acoustic problem with monopole point sources. First, you gotta 
    give some inputs:
        
    Inputs:
        
        space = bempp.api.function_space(grid, "DP", 0) || grid = bempp.api.import_grid('#YOURMESH.msh')
        
        f_range = array with frequencies of analysis. eg:   f1= 20
                                                            f2 = 150
                                                            df = 2
                                                            f_range = np.arange(f1,f2+df,df) 
        
        c0 = speed of sound
        
        r0 = dict[0:numSources] with source positions. eg:  r0 = {}
                                                            r0[0] =  np.array([1.4,0.7,-0.35])
                                                            r0[1] = np.array([1.4,-0.7,-0.35])
                                                            
        q = dict[0:numSources] with constant source strenght S. eg: q = {}
                                                                    q[0] = 1
                                                                    q[1] = 1
        
        mu = dict[physical_group_id]| A dictionary containing f_range sized arrays with admittance values. 
        The key (index) to the dictionary must be the physical group ID defined in Gmsh. If needed, check out
        the bemder.porous functions :). 
                                        eg: zsd1 = porous.delany(5000,0.1,f_range)
                                            zsd2 = porous.delany(10000,0.2,f_range)
                                            zsd3 = porous.delany(15000,0.3,f_range)
                                            mud1 = np.complex128(rho0*c0/np.conj(zsd1))
                                            mud2 = np.complex128(rho0*c0/np.conj(zsd2))
                                            mud3 = np.complex128(rho0*c0/np.conj(zsd3))
                                            
                                            mu = {}
                                            mu[1] = mud2
                                            mu[2] = mud2
                                            mu[3] = mud3
        
        

    """
    #then = time.time()
    
    AP_init = ctrl.AirProperties()
    AC_init = ctrl.AlgControls(AP_init.c0, 1000,1000,10)
    S_init = sources.Source("spherical",coord=[2,0,0])
    R_init = receivers.Receiver(coord=[1.5,0,0])
    grid_init = bempp.api.shapes.regular_sphere(2)
    
    def __init__(self,grid=grid_init,AC=AC_init,AP=AP_init,S=S_init,R=R_init,mu=None,v=None,assembler = 'numba'):
        self.grid = grid
        self.f_range = AC.freq
        self.wavetype = S.wavetype
        # self.r0 = S.coord.reshape(len(S.coord),-1)
        self.r0 = S.coord.T
        # print(self.r0)
        self.q = S.q
        self.mu = mu
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        self.AP = AP
        self.AC = AC
        self.S = S
        self.R = R
        self.BC = 'neumann'
        self.EoI = 0
        self.v = v
        self.IS = 0
        self.assembler = assembler
        
        
        self.mu = collections.OrderedDict(sorted(self.mu.items()))
        conv = []
        for i in range(len(self.f_range)):
            conv.append(
                np.array(
                    [self.mu[key][i]
                     for key in self.mu.keys()],dtype="complex128"))
        self.mu = conv
        # print(self.mu[0])
        # self.mu = np.array([self.mu[i] for i in self.mu.keys()])
        if self.BC == "robin" or "dirichlet": 
            self.space = bempp.api.function_space(self.grid, "DP", 0)
        elif self.BC == "neumann":
            self.space = bempp.api.function_space(self.grid, "P", 1)
    def soft_bemsolve(self):
        """
        Computes the bempp gridFunctions for the interior acoustic problem.
        
        Outputs: 
            
            boundP = grid_function for boundary pressure
            
            boundU = grid_function for boundary velocity
        
        """
        
        self.boundData = {}
        self.space = bempp.api.function_space(self.grid, "DP", 0)
        for fi in range(np.size(self.f_range)):
            

            
            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0 # Calculate wave number

            identity = bempp.api.operators.boundary.sparse.identity(
                self.space, self.space, self.space)
            adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.space, self.space, self.space, k)
            slp = bempp.api.operators.boundary.helmholtz.single_layer(
                self.space, self.space, self.space, k)
            
            ni = 1j*k
            if self.wavetype == "plane":
                
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                # result[0] = 1j * k * np.exp(1j * k * (r[0]*self.r0[0][0]+r[1]*self.r0[0][1]+r[2]*self.r0[0][2]))*(n[0]*self.r0[0][0])+(n[1]*self.r0[0][1])+(n[2]*self.r0[0][2])
                    result[0] = 1j * k * np.exp(1j * k * r[0])*(n[0]-1)

                # for i in range(len(q_fi)):
                    # pos =  np.dot(r,self.r0[i])/np.linalg.norm(self.r0[i])
                    # result[0] +=  1j * k *np.exp(1j * k * pos) * np.dot(self.r0[i],n)/np.linalg.norm(self.r0[i])
            elif self.wavetype == "spherical":
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                    result[0]=0
                    for i in range(len(self.r0.T)): 

                        pos  = np.linalg.norm(r-self.r0[:,i].reshape(1,3),axis=1)
                        val  = self.q.flat[i]*np.exp(1j*k*pos)/(pos)
                        result[0] += val/(pos**2) * (1j*k*pos-1)* np.dot(r-self.r0[:,i],n) - (ni*val)  
            else:
                raise TypeError("Wavetype must be plane or spherical")
            
            monopole_fun = bempp.api.GridFunction(self.space, fun=combined_data)
            
            A = 0.5*identity + adlp - 1j*k*slp               
            Ar = (monopole_fun)
            
            lhs = A
            rhs = Ar

            boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)
        
            # boundU = 1j*(mu_op_r+1j*mu_op_i)*k*boundP + monopole_fun# + 1j*self.rho0*self.c0*v_fun
            
            self.boundData[fi] = boundP
            # u[fi] = boundU
            
            
            
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
        return self.boundData
    
    def hard_bemsolve(self,device="cpu"):
        """
        Computes the bempp gridFunctions for the interior acoustic problem.
        
        Outputs: 
            
            boundP = grid_function for boundary pressure
            
            boundU = grid_function for boundary velocity
        
        """
        if device == "cpu":     
            helpers.set_cpu()
        if device == "gpu":
            helpers.set_cpu()
        self.boundData = {}
        
        # r0_fi = np.array([self.r0[i] for i in self.r0.keys()])
        # q_fi = np.array([self.q[i] for i in self.q.keys()])
        # self.space = bempp.api.function_space(self.grid, "P", 1)

        for fi in range(np.size(self.f_range)):
            

            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0 # Calculate wave number
            
            print(" \n Assembling Layer Potentials")
            identity = bempp.api.operators.boundary.sparse.identity(
                self.space, self.space, self.space)
            dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                self.space, self.space, self.space, k)
            hyp = bempp.api.operators.boundary.helmholtz.hypersingular(self.space, self.space, self.space, k)
            print("Layer Potentials Assembled Succesfuly")
            
            ni = 1j*k # Coupling Parameter
            if self.wavetype == "plane":
                
                @bempp.api.complex_callable
                def combined_data(x, n, domain_index, result):
                    with numba.objmode():
                        for i in range(len(self.r0.T)):
                            pos = (x[0]*self.r0[0,i]+x[1]*self.r0[1,i]+x[2]*self.r0[2,i])
                            nm = (((n[0]-1)*self.r0[0,i])+((n[1]-1)*self.r0[1,i])+((n[2]-1)*self.r0[2,i]))
                            result[0] = 1j * k * np.exp(1j * k * pos/np.linalg.norm(self.r0) )*nm/np.linalg.norm(self.r0[:,i]) #- (ni* np.exp(1j * k * pos/np.linalg.norm(self.r0) ))
                if fi==0:
                    print("Incident Plane Pressure Field Computed")
                    
                    
            elif self.wavetype == "spherical":
                
                @bempp.api.complex_callable
                def combined_data(x, n, domain_index, result):
                    with numba.objmode():
                        result[0]=0
                        for i in range(len(self.r0.T)): 
    
                            pos  = np.linalg.norm(x-self.r0[:,i].reshape(1,3),axis=1)
                            val  = self.q.flat[i]*np.exp(1j*k*pos)/(pos)
                            result[0] += val/(pos**2) * (1j*k*pos-1)* np.dot(x-self.r0[:,i],n) - (ni*val)
                        
                print("Incident Spherical Pressure Field Computed")
               
            else:
                raise TypeError("Wavetype must be plane or spherical")
                
            monopole_fun = bempp.api.GridFunction(self.space, fun=combined_data)
            if fi==0:
                print("Assembling the System of Equations")
            A =  -hyp + ni*(0.5*identity - dlp)            
            Ar =-(monopole_fun)    
            
            lhs = A
            rhs = Ar
            if fi==0:
                print("Solving the System. Obs: This might take a long time for big meshes and/or slow computers. Be patient, breathe.")
            
            then = time.time()
            boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5,use_strong_form=False)
            if fi==0:
                print("It took: ", time.time()-then, " seconds")
                if info == 0:
                    print("System Solved Succesfully. Use the method point_evaluate or grid_evaluate to view results. You can also plot boundary pressure using the plot() method for a specific frequency")
                
            
            boundU = 0*boundP
            
            self.boundData[fi] = [boundP, boundU]
            # u[fi] = boundU
            
            self.IS = 0
            self.BC = "neumann"
            
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return self.boundData
    
    def impedance_bemsolve(self,device="cpu",individual_sources=False):
        """
        Computes the bempp gridFunctions for the interior acoustic problem.
        
        Outputs: 
            
            boundP = grid_function for boundary pressure
            
            boundU = grid_function for boundary velocity
        
        """
        # if device == "cpu":     
        #     helpers.set_cpu()
        # if device == "gpu":
        #     helpers.set_cpu()
        self.bD={}
        if self.mu == None:
            self.mu = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.mu[i] = np.zeros_like(self.f_range)

        if self.v == None:
            self.v = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.v[i] = np.zeros_like(self.f_range)        
        # mu_fi = np.array([self.mu[i] for i in self.mu.keys()])
        self.space = bempp.api.function_space(self.grid, "DP", 0)
        if individual_sources==True:
            for fi in range(np.size(self.f_range)):
                mu_f = self.mu[fi]
                self.boundData = {}  
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0 # Calculate wave number
                
                @bempp.api.callable(complex=True,jit=True,parameterized=True)
                def mu_fun(x,n,domain_index,result,mu_f):
                        result[0]=np.conj(mu_f[domain_index])
                        # print(mu_f[domain_index],domain_index)
                    
                
                mu_op = bempp.api.MultiplicationOperator(
                    bempp.api.GridFunction(self.space,fun=mu_fun,function_parameters=mu_f)
                    ,self.space,self.space,self.space)
                    
                # @bempp.api.real_callable
                # def v_data(x, n, domain_index, result):
                #     with numba.objmode():
                #         result[0] = self.v[domain_index][fi]
                
                identity = bempp.api.operators.boundary.sparse.identity(
                    self.space, self.space, self.space)
                dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                    self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)
                slp = bempp.api.operators.boundary.helmholtz.single_layer(
                    self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)
                
                a = 1j*k*self.c0*self.rho0
                icc=0
                for icc in range(len(self.r0.T)):
                    # self.ir0 = self.r0[:,i].T
        
                    if self.wavetype == "plane":    
                        @bempp.api.callable(complex=True,jit=True,parameterized=True)
                        def combined_data(r, n, domain_index, result,parameters):
                            
                            r0 = np.real(parameters[:3])
                            k = parameters[3]
                            q = parameters[4]
                                
                            ap = np.linalg.norm(r0) 
                            pos = (r[0]*r0[0]+r[1]*r0[1]+r[2]*r0[2])
                            # nm = (((n[0]-1)*r0[0,i]/ap)+((n[1]-1)*r0[1,i]/ap)+((n[2]-1)*r0[2,i]/ap))
                            result[0] = q*(np.exp(1j * k * pos/ap))
                    elif self.wavetype == "spherical":    
                        @bempp.api.callable(complex=True,jit=True,parameterized=True)
                        def combined_data(r, n, domain_index, result,parameters):
                            
                            r0 = np.real(parameters[:3])
                            k = parameters[3]
                            q = parameters[4]
        
                            pos  = np.linalg.norm(r-r0)
                            # print(pos)
                            result[0]  = q*np.exp(1j*k*pos)/(pos)
                        
                    else:
                        raise TypeError("Wavetype must be plane or spherical") 
                        
                    mnp_fun = bempp.api.GridFunction.from_zeros(self.space)
                    function_parameters_mnp = np.zeros(5,dtype = 'float64')     
                   
                    function_parameters_mnp[:3] = self.r0[:,icc]
                    function_parameters_mnp[3] = k
                    function_parameters_mnp[4] = self.q.flat[icc]
                    
                    mnp_fun = bempp.api.GridFunction(self.space,fun=combined_data,
                                                      function_parameters=function_parameters_mnp)    
                    # v_fun = bempp.api.GridFunction(self.space, fun=v_data)
                        
                    monopole_fun = mnp_fun
             
                    Y = a*(mu_op)# + monopole_fun
           
                    lhs = (0.5*identity-dlp) - slp*Y
                    rhs = monopole_fun #- slp*a*v_fun
                
                    boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)#, use_strong_form=True)
                
                    boundU = -Y*boundP #+ a*v_fun#- monopole_fun
                    
                    self.boundData[icc] = [boundP, boundU]
                    # u[fi] = boundU
                    
                    self.BC = "robin"
                    self.IS = 1
                    
                    print('{} Hz - Source {}/{}'.format(self.f_range[fi],icc+1,len(self.r0.T)))
        
                    
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                    
                self.bD[fi] = self.boundData 
                
            return self.bD
        
        elif individual_sources==False:
            self.boundData = {}  
           
            for fi in range(np.size(self.f_range)):
    
                mu_f = self.mu[fi]
                # print(mu_f[0])
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0 # Calculate wave number
                
                @bempp.api.callable(complex=True,jit=True,parameterized=True)
                def mu_fun(x,n,domain_index,result,mu_f):
                        result[0]=np.conj(mu_f[domain_index])
                        # print(mu_f[domain_index],domain_index)
                    
                
                mu_op = bempp.api.MultiplicationOperator(
                    bempp.api.GridFunction(self.space,fun=mu_fun,function_parameters=mu_f)
                    ,self.space,self.space,self.space)
                
                
                # @bempp.api.callable(complex=True,jit=False)
                # def v_data(x, n, domain_index, result):
                #     with numba.objmode():
                #         result[0] = self.v[domain_index][fi]
            
                identity = bempp.api.operators.boundary.sparse.identity(
                    self.space, self.space, self.space)
                dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                    self.space, self.space, self.space, k, assembler="dense", device_interface=self.assembler)
                slp = bempp.api.operators.boundary.helmholtz.single_layer(
                    self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)
                
                # ni = (1j/k)
                a = 1j*k*self.c0*self.rho0

                

                if self.wavetype == "plane":    
                    @bempp.api.callable(complex=True,jit=True,parameterized=True)
                    def combined_data(r, n, domain_index, result,parameters):
                        
                        r0 = np.real(parameters[:3])
                        k = parameters[3]
                        q = parameters[4]
                            
                        ap = np.linalg.norm(r0) 
                        pos = (r[0]*r0[0]+r[1]*r0[1]+r[2]*r0[2])
                        # nm = (((n[0]-1)*r0[0,i]/ap)+((n[1]-1)*r0[1,i]/ap)+((n[2]-1)*r0[2,i]/ap))
                        result[0] = q*(np.exp(1j * k * pos/ap))
                elif self.wavetype == "spherical":    
                    @bempp.api.callable(complex=True,jit=True,parameterized=True)
                    def combined_data(r, n, domain_index, result,parameters):
                        
                        r0 = np.real(parameters[:3])
                        k = parameters[3]
                        q = parameters[4]
    
                        pos  = np.linalg.norm(r-r0)
                        # print(pos)
                        result[0]  = q*np.exp(1j*k*pos)/(pos)
           
    
                # v_fun = bempp.api.GridFunction(self.space, fun=v_data)
                mnp_fun = bempp.api.GridFunction.from_zeros(self.space)
                function_parameters_mnp = np.zeros(5,dtype = 'complex128')     
                for i in range(len(self.r0.T)):
                    function_parameters_mnp[:3] = self.r0[:,i]
                    function_parameters_mnp[3] = k
                    function_parameters_mnp[4] = self.q.flat[i]
                    
                    mnp_fun += bempp.api.GridFunction(self.space,fun=combined_data,
                                                      function_parameters=function_parameters_mnp)
                monopole_fun = mnp_fun
                Y = a*(mu_op)# + monopole_fun
       
                lhs = (0.5*identity-dlp) - slp*Y
                rhs = monopole_fun #- slp*a*v_fun
            
                boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)#, use_strong_form=True)
            
                boundU = -Y*boundP #+ a*v_fun#- monopole_fun
                
                self.boundData[fi] = [boundP, boundU]
                # u[fi] = boundU
                
                self.BC = "robin"
                self.IS = 0
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                
            return self.boundData
    
    def kirchhoff_bemsolve(self,device="cpu",individual_sources=False):
        """
        Computes de Bempp gridFunctions on the surface using the Kirchhoff approximation.
        To set a physical group surface to have pressure=0, attribute the domain_index=213 in Gmsh.

        Parameters
        ----------
        device : TYPE, optional
            DESCRIPTION. The default is "cpu".
        individual_sources : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if device == "cpu":     
            helpers.set_cpu()
        if device == "gpu":
            helpers.set_cpu()
        self.bD={}
        if self.mu == None:
            self.mu = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.mu[i] = np.zeros_like(self.f_range)

        if self.v == None:
            self.v = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.v[i] = np.zeros_like(self.f_range)        
        # mu_fi = np.array([self.mu[i] for i in self.mu.keys()])
        self.space = bempp.api.function_space(self.grid, "DP", 0)
        if individual_sources==True:
            for fi in range(np.size(self.f_range)):
                self.boundData = {}  
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0 # Calculate wave number
                
                @bempp.api.real_callable
                def mu_fun_r(x,n,domain_index,result):
                    with numba.objmode():
                        result[0]=np.real(self.mu[domain_index][fi])
                    
                @bempp.api.real_callable
                def mu_fun_i(x,n,domain_index,result):
                    with numba.objmode():
                        result[0]=-np.imag(self.mu[domain_index][fi])
                
                mu_op_r = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_r),self.space,self.space,self.space)
                mu_op_i = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_i),self.space,self.space,self.space)
                
                @bempp.api.real_callable
                def v_data(x, n, domain_index, result):
                    with numba.objmode():
                        result[0] = self.v[domain_index][fi]
                
                ni = (1j/k)
                a = 1j*k*self.c0*self.rho0
                icc=0
                for icc in range(len(self.r0.T)):
                    # self.ir0 = self.r0[:,i].T

                    if self.wavetype == "plane":
                        
                        @bempp.api.complex_callable
                        def combined_data(r, n, domain_index, result):
                            with numba.objmode():
                                result[0] = 0
                                ap = np.linalg.norm(self.r0) 
                                pos = (r[0]*self.r0[0]+r[1]*self.r0[1]+r[2]*self.r0[2])
                                nm = (((n[0]-1)*self.r0[0]/ap)+((n[1]-1)*self.r0[1]/ap)+((n[2]-1)*self.r0[2]/ap))
                                result[0] = self.q.flat[i]*(np.exp(1j * k * pos/ap))
        
                    elif self.wavetype == "spherical":
                    
                        @bempp.api.complex_callable
                        def combined_data(r, n, domain_index, result):
                            with numba.objmode():
                                    pos  = np.linalg.norm(r-self.r0[:,icc].reshape(1,3),axis=1)
                                    val  = self.q.flat[icc]*np.exp(1j*k*pos)/(pos)
                                    result[0] = 2*(val)#/(pos**2) * (1j*k*pos-1)* np.dot(r-self.r0[:,i],n-1) + 1j*self.mu[domain_index][fi]*k*val))
                                    # print(val)
                    
                    else:
                        raise TypeError("Wavetype must be plane or spherical")      
        
                    v_fun = bempp.api.GridFunction(self.space, fun=v_data)
                        
                    monopole_fun = bempp.api.GridFunction(self.space, fun=combined_data)
                    
                    Y = a*(mu_op_r+1j*mu_op_i)
                    boundP = monopole_fun
                
                    boundU = -Y*boundP + a*v_fun#- monopole_fun
                    
                    self.boundData[icc] = [boundP, boundU]
                    # u[fi] = boundU
                    
                    self.BC = "robin"
                    self.IS = 1
                    
                    print('{} Hz - Source {}/{}'.format(self.f_range[fi],icc+1,len(self.r0.T)))

                    
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                    
                self.bD[fi] = self.boundData 
                
            return self.bD
        
        if individual_sources==False:
            self.boundData = {}  
           
            for fi in range(np.size(self.f_range)):
    
    
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0 # Calculate wave number
                
                
                a = 1j*k*self.c0*self.rho0
                
                if self.wavetype == "plane":
                    
                    @bempp.api.callable(complex=True,jit=True,parametrized=True)
                    def combined_data(r, n, domain_index, result,parameters):
                        result[0] = 0
                        for i in range(len(self.r0.T)):
                            ap = np.linalg.norm(self.r0[:,i]) 
                            pos = (r[0]*self.r0[0,i]+r[1]*self.r0[1,i]+r[2]*self.r0[2,i])
                            nm = (((n[0]-1)*self.r0[0,i]/ap)+((n[1]-1)*self.r0[1,i]/ap)+((n[2]-1)*self.r0[2,i]/ap))
                            # result[0] += -(ni*1j * k *np.exp(1j * k * pos/np.linalg.norm(self.r0[:,i]))*nm
                            #               + a*self.mu[domain_index][fi]*np.exp(1j * k * pos/np.linalg.norm(self.r0[:,i])))
                            result[0] += self.q.flat[i]*(np.exp(1j * k * pos/ap))
    
                elif self.wavetype == "spherical":
                    
                    @bempp.api.callable(complex=True,jit=True,parametrized=True)
                    def combined_data(r, n, domain_index, result,parameters):
                        result[0]=0
                        
                        for i in range(len(self.r0.T)): 
                            if domain_index==213:
                                result[0]=0
                            else:
                                pos  = np.linalg.norm(r-self.r0[:,i].reshape(1,3),axis=1)
                                val  = self.q.flat[i]*np.exp(1j*k*pos)/(pos)
                                result[0] += (2*(val))#/(pos**2) * (1j*k*pos-1)* np.dot(r-self.r0[:,i],n-1) + 1j*self.mu[domain_index][fi]*k*val))
                            # print(val)
                
                else:
                    raise TypeError("Wavetype must be plane or spherical")      
    
                    
                monopole_fun = bempp.api.GridFunction(self.space, fun=combined_data)
                boundP = -monopole_fun
                boundU = 0*boundP
                self.boundData[fi] = [boundP, boundU]
                # u[fi] = boundU
                
                self.BC = "robin"
                self.IS = 0
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                
            return self.boundData
    def monopole(self,fi,pts,ir):
        
        pInc = np.zeros(pts.shape[0], dtype='complex128')
        if self.IS==1:
            pos = np.linalg.norm(pts-self.r0[:,ir])
            pInc = self.q.flat[ir]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(pos)
        else:
            for i in range(len(self.r0.T)): 
                pos = np.linalg.norm(pts-self.r0[:,i].reshape(1,3),axis=1)
                pInc += self.q.flat[i]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(pos)
            
        return pInc
    
    def planewave(self,fi,pts,ir):
        pInc = np.zeros(pts.shape[0], dtype='complex128')
        if self.IS==1:
            pos = pts[:,0]*self.r0[0,ir]+pts[:,1]*self.r0[1,ir]+pts[:,2]*self.r0[2,ir]
            pInc = self.q.flat[ir]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos/np.linalg.norm(self.r0[:,ir]))
        else:
            for i in range(len(self.r0.T)): 
            # pos = np.dot(pts,self.r0[i])/np.linalg.norm(self.r0[fi])
                pos = pts[:,0]*self.r0[0,i]+pts[:,1]*self.r0[1,i]+pts[:,2]*self.r0[2,i]
                pInc += self.q.flat[i]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos/np.linalg.norm(self.r0[:,i]))
            
        return (pInc)
    
    def point_evaluate(self,boundD,R=R_init):
        
        """
        Evaluates the solution (pressure) for a point.
        
        Inputs:
            points = dict[0:numPoints] containing np arrays with receiver positions 
            
            boundData = output from bemsolve()

            
        Output:
            
           pT =  Total Pressure Field
           
           pS = Scattered Pressure Field
           
        """
        pT = {}
        pS = {}
        ppt={}
        pps={}
        pts = R.coord.reshape(len(R.coord),3)
        if self.IS==1:
            for fi in range(np.size(self.f_range)):
            
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0
                
                
                

                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                
                for i in range(len(self.r0.T)):
                    # self.ir0 = self.r0[:,i]
                    pScat =  dlp_pot.evaluate(boundD[fi][i][0])-slp_pot.evaluate(boundD[fi][i][1])
                
                    if self.wavetype == "plane":
                        pInc = self.planewave(fi,pts,i)
                        
                    elif self.wavetype == "spherical":
                        pInc = self.monopole(fi,pts,i)
                    
                    pT[i] = pInc+pScat
                    pS[i] = pScat
        
                    # print(20*np.log10(np.abs(pT[fi])/2e-5))
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                ppt[fi] = np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord))
                pps[fi] = np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))
            return ppt,pps
       
        else:
            
            for fi in range(np.size(self.f_range)):
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0
                
                
                
                # if self.BC == "neumann":
                #     dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                #         self.space, pts.T, k)
                #     pScat =  dlp_pot.evaluate(boundD[fi][0])
                    
                # elif self.BC == "dirichlet":
                #     slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                #         self.space, pts.T, k)
                #     pScat =  -slp_pot.evaluate(boundD[fi][0])
                    
                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                pScat =  dlp_pot.evaluate(boundD[fi][0])-slp_pot.evaluate(boundD[fi][1])
                    
                if self.wavetype == "plane":
                    pInc = self.planewave(fi,pts,ir=0)
                    
                elif self.wavetype == "spherical":
                    pInc = self.monopole(fi,pts,ir=0)
                
                pT[fi] = pInc+pScat
                pS[fi] = pScat
    
                print(20*np.log10(np.abs(pT[fi])/2e-5))
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                
            return  np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord)),np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))
       
    def velocity_evaluate(self,boundD,R):
        
        bempp.api.set_default_device(1,0)
        print('\nSelected device:', bempp.api.default_device().name) 
        pT = {}
        pS = {}
        vx = {}
        vy = {}
        vz = {}
        pTX1 = {}
        pTX2 = {}
        pTY1 = {}
        pTY2 = {}
        pTZ1 = {}
        pTZ2 = {}
        pts = R.coord.reshape(len(R.coord),3)

        for fi in range(np.size(self.f_range)):
            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0
            
            # ptsX1 = pts[0,:] + 0.25*self.f_range[fi]*self.c0
            # ptsX1 = [ptsX1,pts[1,:],pts[2,:]]
            # ptsX2 = pts[0,:] - 0.25*self.f_range[fi]*self.c0
            # ptsX1 = [ptsX1,pts[1,:],pts[2,:]]
            # ptsX1 = [ptsX2,pts[1,:],pts[2,:]]


            # ptsY1 = pts[1,:] + 0.25*self.f_range[fi]*self.c0
            # ptsY2 = pts[1,:] - 0.25*self.f_range[fi]*self.c0
            # ptsY1 = [pts[0,:],ptsY1,pts[2,:]]
            # ptsY2 = [pts[0,:],ptsY2,pts[2,:]]
                
            # ptsZ1 = pts[3,:] + 0.25*self.f_range[fi]*self.c0
            # ptsZ2 = pts[3,:] - 0.25*self.f_range[fi]*self.c0
            # ptsZ1 = [pts[0,:],pts[1,:],ptsZ1]
            # ptsZ2 = [pts[0,:],pts[1,:],ptsZ2]
            
            ptsX1 = pts
            ptsX1[0,:] += 0.25*self.f_range[fi]*self.c0
            ptsX2 = pts
            ptsX2[0,:] +=  - 0.25*self.f_range[fi]*self.c0
            
            ptsY1 = pts
            ptsY1[1,:] += 0.25*self.f_range[fi]*self.c0
            ptsY2 = pts
            ptsY2[1,:] +=  - 0.25*self.f_range[fi]*self.c0
            
            ptsZ1 = pts
            ptsZ1[2,:] += 0.25*self.f_range[fi]*self.c0
            ptsZ2 = pts
            ptsZ2[2,:] +=  - 0.25*self.f_range[fi]*self.c0

            ptsT = np.vstack((pts,ptsX1,ptsX2,ptsY2,ptsY2,ptsZ1,ptsZ2))
            # ptsT = ptsT.reshape(len(R.coord)*7,3)
            
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, pts.T, k)
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, pts.T, k)
            pScat =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pInc = self.monopole(fi,pts)
            
            pT[fi] = np.conj(pInc+pScat)
            
            slp_potX1 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsX1.T, k)
            dlp_potX1 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsX1.T, k)
            pScatx1 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncx1 = self.monopole(fi,ptsX1)
            
            pTX1[fi] = np.conj(pInc+pScat)

            slp_potY1 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsY1.T, k)
            dlp_potY1 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsY1.T, k)
            pScatY1 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncY1 = self.monopole(fi,ptsY1)
            
            pTY1[fi] = np.conj(pInc+pScat)

            slp_potx1 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsZ1.T, k)
            dlp_potZ1 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsZ1.T, k)
            pScatZ1 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncZ1 = self.monopole(fi,ptsZ1)
            
            pTZ1[fi] = np.conj(pInc+pScat)

            slp_potX2 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsX2.T, k)
            dlp_potX2 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsX2.T, k)
            pScatx2 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncx2 = self.monopole(fi,ptsX2)
            
            pTX2[fi] = np.conj(pInc+pScat)

            slp_potY2 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsY2.T, k)
            dlp_potY2 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsY2.T, k)
            pScatY2 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncY2 = self.monopole(fi,ptsY2)
            
            pTY2[fi] = np.conj(pInc+pScat)

            slp_potx2 = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, ptsZ2.T, k)
            dlp_potZ2 = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, ptsZ2.T, k)
            pScatZ2 =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])            
            pIncZ2 = self.monopole(fi,ptsZ2)
            
            pTZ2[fi] = np.conj(pInc+pScat)
             
            
            vx[fi] = np.gradient([pTX1[fi],pTX2[fi]])
            vy[fi] = np.gradient([pTY1[fi],pTY2[fi]])
            vz[fi] = np.gradient([pTZ1[fi],pTZ2[fi]])

            print(20*np.log10(np.abs(pT[fi])/2e-5))
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return  np.array([vx[i] for i in vx.keys()]).reshape(len(vx),len(R.coord)),np.array([vy[i] for i in vy.keys()]).reshape(len(vy),len(R.coord)),np.array([vz[i] for i in vz.keys()]).reshape(len(vz),len(R.coord))    


    def grid_evaluate(self,boundData,fi=0,plane="z",d=0,grid_size=[4,4],n_grid_pts=600,device='cpu'):
        
        """
        Evaluates and plots the SPL in symmetrical grid for a mesh centered at [0,0,0].
        
        Inputs:
            
            fi = frequency index of array f_range
            
            plane = string containg axis to plot. eg: 'xy'
            
            d = Posistion of free axis (in relation to center)
            
            grid_size = Size of dimension to plot
            
            n_grid_pts = number of grid points
            
            boundP = output from bemsolve()
            
            boundU = output from bemsolve()
        """
        pT = {}
        pTI = {}
        pTS = {}
        if self.assembler == 'opencl':
            if device == "cpu":     
                helpers.set_cpu()
            if device == "gpu":
                helpers.set_cpu()
        k = 2*np.pi*self.f_range[fi]/self.c0
        if plane == 'z':
            
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((plot_grid[0].ravel()+100,plot_grid[1].ravel(),d+np.zeros(plot_grid[0].size)))
            
        if plane == 'y':
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((plot_grid[0].ravel(),d+np.zeros(plot_grid[0].size),plot_grid[1].ravel()))       
            
        if plane == 'x':
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((d+np.zeros(plot_grid[0].size),plot_grid[0].ravel(),plot_grid[1].ravel()))
            

        dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
            self.space, grid_pts, k, assembler="dense", device_interface=self.assembler)
        slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
            self.space, grid_pts, k, assembler="dense", device_interface=self.assembler)
        pScat =  dlp_pot.evaluate(boundData[fi][0])-slp_pot.evaluate(boundData[fi][1])
            
        
        if self.wavetype == "plane":
            pInc = self.planewave(fi,grid_pts.T,ir=0)
            
        if self.wavetype == "spherical":
            pInc = self.monopole(fi,grid_pts.T,ir=0)
            
        grid_pT = (pScat+pInc)
        grid_pTI = (np.real(pInc))
        grid_pTS = (np.abs(pScat))

        
        pT[fi] = grid_pT.reshape((n_grid_points,n_grid_points))
        


        pTI[fi] = grid_pTI.reshape((n_grid_points,n_grid_points))
        
        plt.imshow(20*np.log10(np.abs(pTI[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Incident Pressure Field - %1.1f Hz' %(self.f_range[fi]))
        plt.show()
        
        pTS[fi] = grid_pTS.reshape((n_grid_points,n_grid_points))
        
        plt.imshow(20*np.log10(np.abs(pTS[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Scattered Pressure Field - %1.1f Hz' %(self.f_range[fi]))

        plt.show()
        
        plt.imshow(20*np.log10(np.abs(pT[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Total Pressure Field - %1.1f Hz' %(self.f_range[fi]))
        
        plt.show()
        return pT[fi], grid_pts
        
    def combined_grid_evaluate_r(self,fi,plane,d,grid_size,n_grid_pts,boundP,boundU):
        
        """
        Evaluates and plots the SPL in symmetrical grid for a mesh centered at [0,0,0].
        
        Inputs:
            
            fi = frequency index of array f_range
            
            plane = string containg axis to plot. eg: 'xy'
            
            d = Posistion of free axis (in relation to center)
            
            grid_size = Size of dimension to plot
            
            n_grid_pts = number of grid points
            
            boundP = output from bemsolve()
            
            boundU = output from bemsolve()
        """
        pT = {}
        pTI = {}
        pTS = {}
        
        k = 2*np.pi*self.f_range[fi]/self.c0

        bempp.api.set_default_device(1,0)
        print('\nSelected device:', bempp.api.default_device().name) 
        if plane == 'xy':
            
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((plot_grid[0].ravel(),d+np.zeros(plot_grid[0].size),plot_grid[1].ravel()))
            
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, grid_pts, k, assembler="dense", device_interface=self.assembler)            
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, grid_pts, k, assembler="dense", device_interface=self.assembler)
            pScat =  dlp_pot.evaluate(boundP[fi]) - slp_pot.evaluate(boundU[fi])
            
            pInc = self.planewave(fi,grid_pts.T)
            
            grid_pT = (pScat+pInc)
            grid_pTI = (np.real (pInc))
            grid_pTS = (np.abs(pScat))

            
            pT[fi] = grid_pT.reshape((n_grid_points,n_grid_points))
            


            pTI[fi] = grid_pTI.reshape((n_grid_points,n_grid_points))
            
            plt.imshow(20*np.log10(np.abs(pTI[fi].T)/2e-5),  cmap='jet')
            plt.colorbar()
            plt.title('Incident Pressure Field')
            plt.show()
            
            pTS[fi] = grid_pTS.reshape((n_grid_points,n_grid_points))
            
            plt.imshow(20*np.log10(np.abs(pTS[fi].T)/2e-5),  cmap='jet')
            plt.colorbar()
            plt.title('Scattered Pressure Field')

            plt.show()
            
            plt.imshow(20*np.log10(np.abs(pT[fi].T)/2e-5),  cmap='jet')
            plt.colorbar()
            plt.title('Total Pressure Field')
            
            plt.show()
            return grid_pT
        

    def bem_save(self, filename="test", ext = ".pickle"):
        # Simulation data
        gridpack = {'vertices': self.grid.vertices,
                'elements': self.grid.elements,
                'volumes': self.grid.volumes,
                'normals': self.grid.normals,
                'jacobians': self.grid.jacobians,
                'jacobian_inverse_transposed': self.grid.jacobian_inverse_transposed,
                'diameters': self.grid.diameters,
                'integration_elements': self.grid.integration_elements,
                'centroids': self.grid.centroids,
                'domain_indices': self.grid.domain_indices}

        
        bd = {}
        bbd= {}
        # incident_traces = []
        if self.IS == 1:
            
            for sol in range(len(self.f_range)):
                i=0
                bda = {}
                for i in range(len(self.r0.T)):
                    bda[i] = [(self.bD[sol][i][0].coefficients),(self.bD[sol][i][1].coefficients)]  
                bbd[sol] = bda
        elif self.IS==0:
            for sol in range(len(self.f_range)):
                bd[sol] = [(self.boundData[sol][0].coefficients),(self.boundData[sol][1].coefficients)]
                # incident_traces.append(self.simulation._incident_traces[0].coefficients)
            
        simulation_data = {'AC': self.AC,
                           "AP": self.AP,
                           'mu': self.mu,
                           'v': self.v,
                           'R': self.R,
                           'S': self.S,
                           'BC': self.BC,
                           'EoI': self.EoI,
                           'IS': self.IS,
                           # 'receivers': self.pts,
                           'grid': gridpack,
                           'bd': bd,
                           'bbd':bbd}
                           # 'incident_traces': incident_traces}

                
        outfile = open(filename + ext, 'wb')
                
        cloudpickle.dump(simulation_data, outfile)
        outfile.close()
        print('BEM saved successfully.')
    
#%%
class RoomBEM:
    
    """
    Hi, this class contains some tools to solve the interior acoustic problem with monopole point sources. First, you gotta 
    give some inputs:
        
    Inputs:
        
        space = bempp.api.function_space(grid, "DP", 0) || grid = bempp.api.import_grid('#YOURMESH.msh')
        
        f_range = array with frequencies of analysis. eg:   f1= 20
                                                            f2 = 150
                                                            df = 2
                                                            f_range = np.arange(f1,f2+df,df) 
        
        c0 = speed of sound
        
        r0 = dict[0:numSources] with source positions. eg:  r0 = {}
                                                            r0[0] =  np.array([1.4,0.7,-0.35])
                                                            r0[1] = np.array([1.4,-0.7,-0.35])
                                                            
        q = dict[0:numSources] with constant source strenght S. eg: q = {}
                                                                    q[0] = 1
                                                                    q[1] = 1
        
        mu = dict[physical_group_id]| A dictionary containing f_range sized arrays with admittance values. 
        The key (index) to the dictionary must be the physical group ID defined in Gmsh. If needed, check out
        the bemder.porous functions :). 
                                        eg: zsd1 = porous.delany(5000,0.1,f_range)
                                            zsd2 = porous.delany(10000,0.2,f_range)
                                            zsd3 = porous.delany(15000,0.3,f_range)
                                            mud1 = np.compleZ128(rho0*c0/np.conj(zsd1))
                                            mud2 = np.complex128(rho0*c0/np.conj(zsd2))
                                            mud3 = np.complex128(rho0*c0/np.conj(zsd3))
                                            
                                            mu = {}
                                            mu[1] = mud2
                                            mu[2] = mud2
                                            mu[3] = mud3
        
        

    """
    AP_init = ctrl.AirProperties()
    AC_init = ctrl.AlgControls(AP_init.c0, 1000,1000,10)
    S_init = sources.Source("plane",coord=[2,0,0])
    R_init = receivers.Receiver(coord=[1.5,0,0])
    grid_init = bempp.api.shapes.regular_sphere(2)
    
    def __init__(self,grid=grid_init,AC=AC_init,AP=AP_init,S=S_init,R=R_init,mu=None):
        self.grid = grid
        self.f_range = AC.freq
        self.wavetype = S.wavetype
        self.r0 = S.coord.T
        self.q = S.q
        self.mu = mu
        self.c0 = AP.c0
        self.rho0 = AP.rho0
        self.AP = AP
        self.AC = AC
        self.S = S
        self.R = R
        self.EoI = 0

  
    def bemsolve(self,device='cpu'):
        """
        Computes the bempp gridFunctions for the interior acoustic problem.
        
        Outputs: 
            
            boundP = grid_function for boundary pressure
            
            boundU = grid_function for boundary velocity
        
        """
        if device == "cpu":     
            helpers.set_cpu()
        if device == "gpu":
            helpers.set_cpu()
        self.boundData = {}
        self.space = bempp.api.function_space(self.grid, "DP", 0)
        
        for fi in range(np.size(self.f_range)):
        
            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0 # Calculate wave number
            @bempp.api.real_callable
            def mu_fun_r(r,n,domain_index,result):
                with numba.objmode():

                    result[0]=np.real((self.mu[domain_index][fi]))
            @bempp.api.real_callable
            def mu_fun_i(r,n,domain_index,result):
                with numba.objmode():
                    result[0]=np.imag((self.mu[domain_index][fi]))
                                        

            mu_op_r = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_r),self.space,self.space,self.space)
            mu_op_i = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_i),self.space,self.space,self.space)
        
            identity = bempp.api.operators.boundary.sparse.identity(
                self.space, self.space, self.space)
            dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                self.space, self.space, self.space, k)
            slp = bempp.api.operators.boundary.helmholtz.single_layer(
                self.space, self.space, self.space, k)
            
            a = 1j*k*(mu_op_r+1j*mu_op_i)
            @bempp.api.complex_callable
            def monopole_data(r, n, domain_index, result):
                with numba.objmode():
                    result[0]=0
                    for i in range(len(self.r0.T)):
                        pos = np.linalg.norm(r-self.r0[:,i])
                        val  = self.q.flat[i]*np.exp(1j*k*pos)/(4*np.pi*pos)
                        result[0] +=  -(1j*(self.mu[domain_index][fi])*k*val - val/(pos*pos) * (1j*k*pos-1)* np.dot(r-self.r0[i],n))
                
            monopole_fun = bempp.api.GridFunction(self.space, fun=monopole_data)
#            v_fun = bempp.api.GridFunction(self.space, fun=v_data)
        
            lhs = (.5 * identity + dlp - a*slp) 
            rhs = -slp*(monopole_fun)
        
        
            boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5, use_strong_form=True)
        
            boundU = a*boundP - monopole_fun
            
            self.boundData[fi] = [boundP, boundU]
            # u[fi] = boundU
            
            self.BC = "robin"
            
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return self.boundData
    
    def monopole(self,fi,pts):
        
        pInc = np.zeros(pts.shape[0], dtype='complex128')
        
        for i in range(len(self.r0.T)): 
            pos = np.linalg.norm(pts-self.r0[:,i].reshape(1,3),axis=1)
            pInc += self.q.flat[i]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(4*np.pi*pos)
            
        return pInc  
    
    def point_evaluate(self,R, boundD):
        
        """
        Evaluates the solution (pressure) for a point.
        
        Inputs:
            points = dict[0:numPoints] containing np arrays with receiver positions 
            
            boundP = output from bemsolve()
            
            boundU = output from bemsolve()
            
        Output:
            
           pT =  Total Pressure Field
           
        """
        bempp.api.set_default_device(1,0)
        print('\nSelected device:', bempp.api.default_device().name) 
        pT = {}
        pS = {}
        pts = R.coord.reshape(len(R.coord),3)

        for fi in range(np.size(self.f_range)):
            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0
                
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, pts.T, k,assembler='dense')
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, pts.T, k)
            pScat =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])
            
            pInc = self.monopole(fi,pts)
            
            pT[fi] = np.conj(pInc+pScat)
            pS[fi] = np.conj(pScat)

            print(20*np.log10(np.abs(pT[fi])/2e-5))
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return  np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord)),np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))
    
    # def velocity_evaluate(self,R,bD):
        
    #     bempp.api.set_default_device(1,0)
    #     print('\nSelected device:', bempp.api.default_device().name) 
    #     pT = {}
    #     pS = {}
    #     vx = {}
    #     vy = {}
    #     vz = {}
    #     pts = R.coord.reshape(len(R.coord),3)

    #     for fi in range(np.size(self.f_range)):
    #         f = self.f_range[fi] #Convert index to frequency
    #         k = 2*np.pi*f/self.c0
            
    #         ptsX1 = pts[0,:] + 0.25*self.f_range[fi]*self.c0
    #         ptsX2 = pts[0,:] - 0.25*self.f_range[fi]*self.c0

    #         ptsY1 = pts[1,:] + 0.25*self.f_range[fi]*self.c0
    #         ptsY2 = pts[1,:] - 0.25*self.f_range[fi]*self.c0
                
    #         ptsZ1 = pts[3,:] + 0.25*self.f_range[fi]*self.c0
    #         ptsZ2 = pts[3,:] - 0.25*self.f_range[fi]*self.c0
            
    #         ptsT = np.hstack(pts,ptsX1,ptsX2,ptsY2,ptsY2,ptsZ1,ptsZ2)
            
    #         # ptsT = ptsT.reshape(len(R.coord)*7,3)
            
    #         slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
    #             self.space, ptsT.T, k)
    #         dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
    #             self.space, ptsT.T, k)
    #         pScat =  (slp_pot * boundD[fi][1] - dlp_pot * boundD[fi][0])
            
    #         pInc = self.monopole(fi,pts)
            
    #         pT[fi] = np.conj(pInc+pScat)
    #         pS[fi] = np.conj(pScat)
            
    #         pT[fi] = pT.reshape(7,3)
            
    #         vx[fi] = np.grad(pT[fi][1],pT[fi][2])
    #         vy[fi] = np.grad(pT[fi][3],pT[fi][4])
    #         vz[fi] = np.grad(pT[fi][5],pT[fi][6])

    #         print(20*np.log10(np.abs(pT[fi])/2e-5))
    #         print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
    #     return  np.array([vx[i] for i in vx.keys()]).reshape(len(vx),len(R.coord)),
    # np.array([vy[i] for i in vy.keys()]).reshape(len(vy),len(R.coord)),
    # np.array([vz[i] for i in vy.keys()]).reshape(len(vz),len(R.coord))
            
            

    def combined_grid_evaluate(self,boundData,fi=0,plane="z",d=0,grid_size=[4,4],n_grid_pts=600):
        
        """
        Evaluates and plots the SPL in symmetrical grid for a mesh centered at [0,0,0].
        
        Inputs:
            
            fi = frequency index of array f_range
            
            plane = string containg axis to plot. eg: 'xy'
            
            d = Posistion of free axis (in relation to center)
            
            grid_size = Size of dimension to plot
            
            n_grid_pts = number of grid points
            
            boundP = output from bemsolve()
            
            boundU = output from bemsolve()
        """
        pT = {}
        pTI = {}
        pTS = {}
        
        k = 2*np.pi*self.f_range[fi]/self.c0

        helpers.set_gpu()
        if plane == 'z':
            
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((plot_grid[0].ravel(),plot_grid[1].ravel(),d+np.zeros(plot_grid[0].size)))
            
        if plane == 'y':
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((plot_grid[0].ravel(),d+np.zeros(plot_grid[0].size),plot_grid[1].ravel()))       
            
        if plane == 'x':
            n_grid_points = n_grid_pts
            plot_grid = np.mgrid[-grid_size[0]/2:grid_size[0]/2:n_grid_points*1j, -grid_size[1]/2:grid_size[1]/2:n_grid_points*1j]
            grid_pts = np.vstack((d+np.zeros(plot_grid[0].size),plot_grid[0].ravel(),plot_grid[1].ravel()))
            


        elif self.BC == "robin":
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, grid_pts, k)
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, grid_pts, k)
            pScat =  dlp_pot.evaluate(boundData[fi][0])-slp_pot.evaluate(boundData[fi][1])
            
        
        # if self.wavetype == "plane":
        #     pInc = self.planewave(fi,grid_pts.T)
            
        if self.wavetype == "spherical":
            pInc = self.monopole(fi,grid_pts.T)
            
        grid_pT = np.conj(pScat+pInc)
        grid_pTI = (np.real(pInc))
        grid_pTS = (np.abs(pScat))

        
        pT[fi] = grid_pT.reshape((n_grid_points,n_grid_points))
        


        pTI[fi] = grid_pTI.reshape((n_grid_points,n_grid_points))
        
        plt.imshow(20*np.log10(np.abs(pTI[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Incident Pressure Field - %1.1f Hz' %(self.f_range[fi]))
        plt.show()
        
        pTS[fi] = grid_pTS.reshape((n_grid_points,n_grid_points))
        
        plt.imshow(20*np.log10(np.abs(pTS[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Scattered Pressure Field - %1.1f Hz' %(self.f_range[fi]))

        plt.show()
        
        plt.imshow(20*np.log10(np.abs(pT[fi].T)/2e-5),  cmap='jet')
        plt.colorbar()
        plt.title('Total Pressure Field - %1.1f Hz' %(self.f_range[fi]))
        
        plt.show()
        return pT[fi], grid_pts