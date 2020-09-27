import numpy as np
from bemder import controlsair
class BC():
    def __init__(self,AC,AP):
        self.AP = AP
        self.AC = AC
        self.mu = {}
        self.v = {}
    
    def impedance(self,domain_index, impedance):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        
        for i in domain_index:
            self.mu[i] = np.array(1/impedance[:,i])

    def rigid(self,domain_index):
        """
        

        Parameters
        ----------
        domain_index : int
            Physical group index assigned in gmsh for rigid surfaces


        """
        self.mu[domain_index] = np.zeros_like(self.AC.freq)
                
    def admittance(self,domain_index, admittance):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        
        if type(admittance) == int or float:
            self.mu[domain_index] = np.ones_like(self.AC.freq)*admittance
        
        else:
            
            for i in domain_index:
                self.mu[i] = np.array(admittance[:,i])

            
    def velocity(self,domain_index, velocity):
        """
        

        Parameters
        ----------
        domain_index : TYPE
            Physical group indexes assigned in gmsh for each surface.
        impedance : TYPE
            frequency x domain_index matrix with surface impedance values.

        Returns
        -------
        None.

        """
        
        self.v[domain_index] = np.array(velocity)           
        
    def delany(self,domain_index,RF,d):
    
        """
        This function implements th e Delany-Bazley-Miki model for a single porous layers.
        
        Input:
            RF: Flow Resistivity []
            d: Depth of porous layer [m]
            f_range: Frequency vector [Hz]
        
        Output:
            Zs: Surface Impedance [Pa*s/m]
        """
        f_range = self.AC.freq
        w = 2*np.pi*f_range
    
    
        C1=0.0978
        C2=0.7
        C3=0.189
        C4=0.595
        C5=0.0571
        C6=0.754
        C7=0.087
        C8=0.723
    
        X = f_range*self.AP.rho0/RF
        cc = self.AP.c0/(1+C1*np.power(X,-C2) -1j*C3*np.power(X,-C4))
        rhoc = (self.AP.rho0*self.AP.c0/cc)*(1+C5*np.power(X,-C6)-1j*C7*np.power(X,-C8))
    
        Zs = -1j*rhoc*cc/np.tan((w/cc)*d) 
    
        self.mu[domain_index] = np.array(1/Zs)
            