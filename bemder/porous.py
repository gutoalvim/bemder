import numpy as np
c0 = 343 #Speed ou sound
rho0 = 1.21 #Air density
def delany(RF,d,f_range):
    
    """
    This function implements th e Delany-Bazley-Miki model for a single porous layers.
    
    Input:
        RF: Flow Resistivity []
        d: Depth of porous layer [m]
        f_range: Frequency vector [Hz]
    
    Output:
        Zs: Surface Impedance [Pa*s/m]
    """
    w = 2*np.pi*f_range


    C1=0.0978
    C2=0.7
    C3=0.189
    C4=0.595
    C5=0.0571
    C6=0.754
    C7=0.087
    C8=0.723

    X = f_range*rho0/RF
    cc = c0/(1+C1*np.power(X,-C2) -1j*C3*np.power(X,-C4))
    rhoc = (rho0*c0/cc)*(1+C5*np.power(X,-C6)-1j*C7*np.power(X,-C8))

    Zs = -1j*rhoc*cc/np.tan((w/cc)*d) 

    return Zs

def allard(RF, ai, phi, A, Cp, Cv, d, f_range):
    
    """
    This function implements the Allard model for a single porous layers.
    
    Input:
        RF: Flow Resistivity []
        ai: Tortuosity
        phi: Porosity
        A: Characteristic Viscous Length [m*1e-6]
        d: Depth of porous layer [m]
        f_range: Frequency vector [Hz]
    
    Output:
        Zs: Surface Impedance [Pa*s/m]
    """
    
    w = 2*np.pi*f_range
    ni = 1.84e-5
    B2 = 0.77
    y = Cp/Cv
    
    rhoc = rho0*ai*(1+ (RF*phi/(1j*ai*rho0*w))*np.sqrt((1+ (4*1j*(ai**2)*ni*rho0*w/(RF**2 * A**2 * phi**2)))))
    K = y*P0/(y-(y-1)/(1+ (RF*phi/(1j*ai*rho0*B2*w))*np.sqrt((1+ (4*1j*(ai**2)*ni*rho0*B2*w/(RF**2 * A**2 * phi**2))))))
    
    Zc = np.sqrt(K*rhoc)
    kc = w*np.sqrt(rhoc/K)
    
    Zs = -1j*Zc/np.tan(kc*d)