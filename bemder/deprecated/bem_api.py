#%%
import time
import bempp.api
import numpy as np
from bemder import controlsair as ctrl
from bemder import sources
from bemder import receivers
from bemder import helpers
from matplotlib import pylab as plt
import cloudpickle
bempp.api.PLOT_BACKEND = "gmsh"


 

#bempp.api.GLOBAL_PARAMETERS.assembly.dense


#%%
class ExteriorBEM:
    # bempp.api.DEVICE_PRECISION_CPU = 'single'    
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
        self.AC = AC,
        self.S = S
        self.R = R
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
                
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                    for i in range(len(self.r0.T)):
                        pos = (r[0]*self.r0[0,i]+r[1]*self.r0[1,i]+r[2]*self.r0[2,i])
                        nm = (((n[0]-1)*self.r0[0,i])+((n[1]-1)*self.r0[1,i])+((n[2]-1)*self.r0[2,i]))
                        result[0] = 1j * k * np.exp(1j * k * pos/np.linalg.norm(self.r0) )*nm/np.linalg.norm(self.r0[:,i]) #- (ni* np.exp(1j * k * pos/np.linalg.norm(self.r0) ))
                if fi==0:
                    print("Incident Plane Pressure Field Computed")
            elif self.wavetype == "spherical":
                
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                    result[0]=0
                    for i in range(len(self.r0.T)): 

                        pos  = np.linalg.norm(r-self.r0[:,i].reshape(1,3),axis=1)
                        val  = self.q.flat[i]*np.exp(1j*k*pos)/(pos)
                        result[0] += val/(pos**2) * (1j*k*pos-1)* np.dot(r-self.r0[:,i],n) - (ni*val)
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
            
            self.BC = "neumann"
            
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return self.boundData
    
    def impedance_bemsolve(self,device="cpu"):
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
        # mu_fi = np.array([self.mu[i] for i in self.mu.keys()])
        self.space = bempp.api.function_space(self.grid, "DP", 0)
        for fi in range(np.size(self.f_range)):
            


            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0 # Calculate wave number
            
            @bempp.api.real_callable(jit=False)
            def mu_fun_r(x,n,domain_index,result):
                result[0]=np.real(self.mu[domain_index][fi])
                
            @bempp.api.real_callable(jit=False)
            def mu_fun_i(x,n,domain_index,result):
                result[0]=np.imag(self.mu[domain_index][fi])

#            @bempp.api.real_callable(jit=False)
#            def v_data(x,n,domain_index,result):
#                result[0]=0
#                result[0] = self.v[domain_index][fi]*(n[0]-1)
            
            mu_op_r = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_r),self.space,self.space,self.space)
            mu_op_i = bempp.api.MultiplicationOperator(bempp.api.GridFunction(self.space,fun=mu_fun_i),self.space,self.space,self.space)
        
            identity = bempp.api.operators.boundary.sparse.identity(
                self.space, self.space, self.space)
            dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                self.space, self.space, self.space, k)
            adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.space, self.space, self.space, k)
            hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
                self.space, self.space, self.space, k)
            slp = bempp.api.operators.boundary.helmholtz.single_layer(
                self.space, self.space, self.space, k)
            
            ni = (1/(1j*k))
            a = 1j*k*self.c0*self.rho0
            if self.wavetype == "plane":
                
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                    result[0] = 0
                    for i in range(len(self.r0.T)):
                        ap = np.linalg.norm(self.r0[:,i]) 
                        pos = (r[0]*self.r0[0,i]+r[1]*self.r0[1,i]+r[2]*self.r0[2,i])
                        nm = (((n[0]-1)*self.r0[0,i]/ap)+((n[1]-1)*self.r0[1,i]/ap)+((n[2]-1)*self.r0[2,i]/ap))
                        result[0] += -(ni*1j * k *np.exp(1j * k * pos/np.linalg.norm(self.r0[:,i]))*nm
                                      + np.exp(1j * k * pos/np.linalg.norm(self.r0[:,i])))

            elif self.wavetype == "spherical":
                
                @bempp.api.complex_callable(jit=False)
                def combined_data(r, n, domain_index, result):
                    result[0]=0
                    for i in range(len(self.r0.T)): 

                        pos  = np.linalg.norm(r-self.r0[:,i].reshape(1,3),axis=1)
                        val  = self.q.flat[i]*np.exp(1j*k*pos)/(pos)
                        result[0] += -((val/(pos**2) * (1j*k*pos-1)* np.dot(r-self.r0[:,i],n-1) + 1j*self.mu[domain_index][fi]*k*val))
                
            else:
                raise TypeError("Wavetype must be plane or spherical")      
                
            monopole_fun = bempp.api.GridFunction(self.space, fun=combined_data)
            # monopole_fun.plot()
#            v_fun = bempp.api.GridFunction(self.space, fun=v_data)

        
            
            Y = a*(mu_op_r+1j*mu_op_i)# + monopole_fun
            # A = 0.5*identity + dlp - ni*slp
            # Bp = hyp + ni*(0.5*identity - adlp)

            # lhs = Bp + ni*Y*A
            # rhs = monopole_fun
            
            
            Ap = 0.5*identity + adlp - 1j*ni*slp
            B = hyp + 1j*ni*(0.5*identity - dlp)
            
            C = B + 1j*Ap*Y
            
            lhs = C
            rhs = Ap*monopole_fun
            
            # Hp = hyp
            # D = a*Y*slp
            # G = a*slp
            # H = dlp
            # Dp = a*Y*adlp
            
            # lhs = 0.5*identity + H + D + ni*(Hp + Dp + 0.5*a*Y)
            # rhs = monopole_fun
        
            boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)#, use_strong_form=True)
        
            boundU = -ni*Y*boundP + monopole_fun
            
            self.boundData[fi] = [boundP, boundU]
            # u[fi] = boundU
            
            self.BC = "robin"
            
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return self.boundData
    

    def monopole(self,fi,pts):
        
        pInc = np.zeros(pts.shape[0], dtype='complex128')
        
        for i in range(len(self.r0.T)): 
            pos = np.linalg.norm(pts-self.r0[:,i].reshape(1,3),axis=1)
            pInc += self.q.flat[i]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(pos)
        
        return pInc
    
    def planewave(self,fi,pts):
        pInc = np.zeros(pts.shape[0], dtype='complex128')
        
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
        pts = R.coord.reshape(len(R.coord),3)
        
        for fi in range(np.size(self.f_range)):
            f = self.f_range[fi] #Convert index to frequency
            k = 2*np.pi*f/self.c0
            
            if self.BC == "neumann":
                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k)
                pScat =  dlp_pot.evaluate(boundD[fi][0])
                    
            elif self.BC == "dirichlet":
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k)
                pScat =  -slp_pot.evaluate(boundD[fi][0])
                
            elif self.BC == "robin":
                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k)
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k)
                pScat =  dlp_pot.evaluate(boundD[fi][0])-slp_pot.evaluate(boundD[fi][1])
                
            if self.wavetype == "plane":
                pInc = self.planewave(fi,pts)
                
            elif self.wavetype == "spherical":
                pInc = self.monopole(fi,pts)
            
            pT[fi] = pInc+pScat
            pS[fi] = pScat

            print(20*np.log10(np.abs(pT[fi])/2e-5))
            print('{} / {}'.format(fi+1,np.size(self.f_range)))
            
        return  np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord)),np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))
    


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
            

        if self.BC == "neumann":
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, grid_pts, k)
            pScat =  dlp_pot.evaluate(boundData[fi][0])
                
        elif self.BC == "dirichlet":
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, grid_pts, k)
            pScat =  -slp_pot.evaluate(boundData[fi][0])
            
        elif self.BC == "robin":
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, grid_pts, k)
            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                self.space, grid_pts, k)
            pScat =  dlp_pot.evaluate(boundData[fi][0])-slp_pot.evaluate(boundData[fi][1])
            
        
        if self.wavetype == "plane":
            pInc = self.planewave(fi,grid_pts.T)
            
        if self.wavetype == "spherical":
            pInc = self.monopole(fi,grid_pts.T)
            
        grid_pT = (pScat+pInc)
        grid_pTI = (np.real(pInc))
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
                self.space, grid_pts, k)            
            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                self.space, grid_pts, k)
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
    
        u = []
        un = []
        # incident_traces = []
    
        for sol in range(len(self.f_range)):
            u.append(self.boundData[sol][0].coefficients)
            un.append(self.boundData[sol][1].coefficients)
            # incident_traces.append(self.simulation._incident_traces[0].coefficients)
    
        simulation_data = {'AC': self.AC,
                           "AP": self.AP,
                           # 'admittance_factors': self.mu,
                           'sources': self.S,
                           'BC': self.BC,
                           # 'receivers': self.pts,
                           'grid': gridpack,
                           'u':  u,
                           'un': un}
                           # 'incident_traces': incident_traces}

                
        outfile = open(filename + ext, 'wb')
                
        cloudpickle.dump(simulation_data, outfile)
        outfile.close()
        print('BEM saved successfully.')
    
    def bem_load(self, filename, ext = ".pickle"):
        # from bempp.api import function_space
        # from bempp.applications.room_acoustic import Simulation
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
        self.AP = simulation_data['AP']
        # self.pts = simulation_data['receivers']
        self.AC = simulation_data['AC']
        self.S = simulation_data["sources"]
        # self.R = simulation_data["R"]
        # self.set_status = True
        self.BC = simulation_data["BC"]
    
        boundData = {}
        if simulation_data["BC"] == "robin" or "dirichlet": 
            self.space = bempp.api.function_space(self.grid, "DP", 0)
        elif simulation_data["BC"] == "neumann":
            self.space = bempp.api.function_space(self.grid, "P", 1)
        # self.simulation._incident_traces = []
        # self.simulation._incident_fields = []
        for sol in range(len(self.f_range)):
            u = bempp.api.GridFunction(self.space, coefficients=simulation_data['u'][sol])
            un = bempp.api.GridFunction(self.space, coefficients=simulation_data['un'][sol])
            # incident_traces = GridFunction(self.simulation._space, coefficients=simulation_data['incident_traces'][sol])
            boundData[sol] = [u, un]
        

            # self.simulation._incident_traces.append(incident_traces)
        # for frequency, mu in zip(self.simulation._frequencies, self.simulation._admittance_factors):
        #     wavenumber = 2 * np.pi * frequency / 343
        #     _, uinc = self.simulation._compute_incident_trace_and_field(wavenumber, mu, grid=False, incident=True)
        #     self.simulation._incident_fields.append(uinc)
    
        print('\tBEM loaded successfully.')
        
        return boundData