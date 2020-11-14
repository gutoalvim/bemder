import bemder
import cloudpickle
import pickle
import os
import time
# import bemder.bem_api_new as bem
import matplotlib.pyplot as plt
import numpy as np

def progress_bar(progress):
    """
    Prints progress bar.
        progress is an int from 0 to 1
    """
    import sys

    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    # clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    sys.stdout.write("\r" + text)
    sys.stdout.flush()
    # print(text, end='\r')


def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array. Returns closest value and position index.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def view_object(obj):
    """
    Prints all the Configuration fields.
    """
    from pprint import pprint

    pprint(vars(obj))


def folder_files(path):
    """
    Lists all files in a given path.
    """
    import os

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if not '__init__' in file:
                if not '.png' in file:
                    if not '.ipynb' in file:
                        file = file.split('.')
                        name = file[0]
                        extension = file[1]
                        files.append(name)
    return files


def set_gpu(printDevice=True):
    """
    Set non-Intel GPU as current computation device if available.
    """
    import pyopencl as _cl
    import bempp.api
    from bemder.cl_helpers import Context, set_default_device
    platforms = _cl.get_platforms()
    for platform_index, platform in enumerate(platforms):
        devices = platform.get_devices()
        if 'Intel' not in platform.get_info(_cl.platform_info.NAME):
            for device_index, device in enumerate(devices):
                if device.type == _cl.device_type.GPU:
                    set_default_device(platform_index, device_index)
    if printDevice:
        print('Selected device:', bempp.api.default_device().name)


def set_cpu(printDevice=True):
    """
    Set CPU as current computation device.
    """
    import pyopencl as _cl
    import bempp
    from bemder.cl_helpers import Context, set_default_device
    platforms = _cl.get_platforms()
    for platform_index, platform in enumerate(platforms):
        devices = platform.get_devices()
        for device_index, device in enumerate(devices):
            if device.type == _cl.device_type.CPU:
                set_default_device(platform_index, device_index)
    # if printDevice:
        # print('Selected device:', bempp.api.default_device().name)


def float_range(initVal, finalVal, step, decimals=1):
    """
    Returns a list with values between the initial and final values with the chosen step resolution and precision.
    """
    itemCount = int((finalVal - initVal)/step) + 2
    items = []
    for x in range(itemCount):
        items.append(truncate(initVal, decimals=decimals))
        initVal += step
    for i in range(1, len(items)-1):
        if items[i] == items[i-1]:
            items.remove(items[i])
    return items


def truncate(n, decimals=0):
    """
    Truncates a given value according to the chosen amount of decimals.
    """
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# def pack_bem(filename='Configuration', save=True, folder=None, timestamp=True, ext='.pickle'):
#     if save is True:
#         # timestr = time.strftime("%Y%m%d-%H%M_")
#         # filename = filename+'_'+str(self.config_number)
#         outfile = open(filename + ext, 'wb')

#         packedData = {}

#         try:
#             simulation_data, bem_data = bem.pack()
#             packedData['bem'] = [simulation_data, bem_data]
#         except AttributeError:
#             if save:
#                 print('BEM data not found.')

#     if save is True:
#         cloudpickle.dump(packedData, outfile)
#         outfile.close()
#         print('Saved successfully.')
#     if save is False:
#         return packedData

def diffusion_coef(frequency,pDiffuser,pRef, plot=False):

    f_range = frequency
    Tf = np.zeros([len(f_range),1])
    
    for i in range(len(f_range)):
        T = (np.sum(np.abs(pDiffuser[i,:]))**2 - np.sum(np.abs(pDiffuser[i,:])**2))/((len(pDiffuser.T)-1)*np.sum(np.abs(pDiffuser[i,:])**2))
        T_ref = (np.sum(np.abs(pRef[i,:]))**2 - np.sum(np.abs(pRef[i,:])**2))/((len(pRef.T)-1)*np.sum(np.abs(pRef[i,:])**2))
        
        Tf[i] = (T - T_ref)/(1-T_ref)
            
    if plot == True: 
        
        fig, ax = plt.subplots()
        ax.plot(f_range, Tf)
        ax.set_ylim(0,1)
    return Tf
def random_diffusion_coef(frequency,pDiffuser,pRef, S,plot=False):
    f_range = frequency
    
    n_average=1
    Tf = np.zeros([int(len(f_range)/n_average),1],dtype=float)
    T_ref = np.zeros([len(S.coord),1],dtype=float)
    T = np.zeros([len(S.coord),1],dtype=float)
    for i in range(len(f_range)):
        for ir in range(len(S.coord)):
            T[ir] = (np.sum(np.abs(pDiffuser[i][ir,:]))**2 - np.sum(np.abs(pDiffuser[i][ir,:])**2))/((len(pDiffuser[i][ir,:].T)-1)*np.sum(np.abs(pDiffuser[i][ir,:])**2))
            T_ref[ir] = (np.sum(np.abs(pRef[i][ir,:]))**2 - np.sum(np.abs(pRef[i][ir,:])**2))/((len(pRef[i][ir,:].T)-1)*np.sum(np.abs(pRef[i][ir,:])**2))
        Tf[i] = (np.mean(T) - np.mean(T_ref))/(1-np.mean(T_ref))
          
    return Tf

def r_d_coef(frequency, pDiffuser,pRef,S,n_average=7,s_number=False):
    """
    

    Parameters
    ----------
    frequency : TYPE
        DESCRIPTION.
    pDiffuser : TYPE
        DESCRIPTION.
    pRef : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    n_average : TYPE, optional
        DESCRIPTION. The default is 7.
    s_number : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    d : TYPE
        Normalized Diffusion Coefficient.

    """
    
    
    f_range = frequency
    Tf = np.zeros([int(len(f_range)/n_average),1],dtype=float)
    Tr = np.zeros([len(S.coord),1],dtype=float)
    T = np.zeros([len(S.coord),1],dtype=float)
    ppd = {}
    ppr = {}
    ddp = {}
    rrp = {}
    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = np.zeros_like(pDiffuser[0])
        rp = np.zeros_like(pDiffuser[0])# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            # pD = np.abs(pDiffuser.get(iic))
            # pR = np.abs(pRef.get(iic))
            pD = np.real(np.abs(pDiffuser.get(iic))/np.amax(np.abs(pDiffuser.get(iic))))
            pR = np.real(np.abs(pRef.get(iic))/np.amax(np.abs(pRef.get(iic))))
            rp += pR #np.array([pR[c] for c in pR.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            dp += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            # print(iic)
        # print("STOP")
        ddp[i] = dp/(n_average)
        rrp[i] = rp/(n_average)
        
        ppd[i] = ddp[i]
        ppr[i] = rrp[i]
        a=n_average
    if s_number == 'sum':
        for ic in range(int(len(f_range)/n_average)):
            ir=0
            for ir in range(len(S.coord)):
                T[ir] = (np.sum(np.abs(np.sum(ppd[ic])))**2 - np.sum(np.abs(np.sum(ppd[ic]))**2))/((len(ppd[ic].T)-1)*np.sum(np.abs(np.sum(ppd[ic])**2)))
                Tr[ir] = (np.sum(np.abs(np.sum(ppr[ic])))**2 - np.sum(np.abs(np.sum(ppr[ic]))**2))/((len(ppr[ic].T)-1)*np.sum(np.abs(np.sum(ppr[ic])**2)))
                
            Tf[ic] = (np.mean(T) - np.mean(Tr))/(1-np.mean(Tr))        
        
    
    if type(s_number) == int or type(a) == float:
        ir = s_number
        for ic in range(int(len(f_range)/n_average)):
            T = (np.sum(np.abs(ppd[ic][ir,:]))**2 - np.sum(np.abs(ppd[ic][ir,:])**2))/((len(ppd[ic][ir,:].T)-1)*np.sum(np.abs(ppd[ic][ir,:])**2))
            Tr = (np.sum(np.abs(ppr[ic][ir,:]))**2 - np.sum(np.abs(ppr[ic][ir,:])**2))/((len(ppr[ic][ir,:].T)-1)*np.sum(np.abs(ppr[ic][ir,:])**2))
            Tf[ic] = (T - Tr)/(1-Tr)
        print(Tf)
    elif s_number=='random':
            
        for ic in range(int(len(f_range)/n_average)):
            ir=0
            for ir in range(len(S.coord)):
                T[ir] = (np.sum(np.abs(ppd[ic][ir,:]))**2 - np.sum(np.abs(ppd[ic][ir,:])**2))/((len(ppd[ic][ir,:].T)-1)*np.sum(np.abs(ppd[ic][ir,:])**2))
                Tr[ir] = (np.sum(np.abs(ppr[ic][ir,:]))**2 - np.sum(np.abs(ppr[ic][ir,:])**2))/((len(ppr[ic][ir,:].T)-1)*np.sum(np.abs(ppr[ic][ir,:])**2))
                
            Tf[ic] = (np.mean(T) - np.mean(Tr))/(1-np.mean(Tr))
            # Tf[ic] = np.m'ean(T)
    
    idx = np.where(Tf<0)
    Tf[idx] = 0
    return Tf

def r_d_coef_spl(frequency, pDiffuser,pRef,S,n_average=7,s_number=False):
    """
    

    Parameters
    ----------
    frequency : TYPE
        DESCRIPTION.
    pDiffuser : TYPE
        DESCRIPTION.
    pRef : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    n_average : TYPE, optional
        DESCRIPTION. The default is 7.
    s_number : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    Tf : TYPE
        DESCRIPTION.

    """
    f_range = frequency
    Tf = np.zeros([int(len(f_range)/n_average),1],dtype=float)
    T = np.zeros([int(len(f_range)/n_average),1],dtype=float)

    Tr = np.zeros([len(S.coord),1],dtype=float)
    T = np.zeros([len(S.coord),1],dtype=float)

    ppd = {}
    ppr = {}
    ddp = {}
    rrp = {}
    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = np.zeros_like(pDiffuser[0],dtype=float)
        rp = np.zeros_like(pDiffuser[0],dtype=float)# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            pD = np.real(np.abs(pDiffuser.get(iic))/np.amax(np.abs(pDiffuser.get(iic))))
            pR = np.real(np.abs(pRef.get(iic))/np.amax(np.abs(pRef.get(iic))))
            rp += pR #np.array([pR[c] for c in pR.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            dp += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            print(iic)
        print("STOP")
        ddp[i] = 20*np.log10((dp/(n_average))/2e-5)
        rrp[i] = 20*np.log10((rp/(n_average))/2e-5)
        
        ppd[i] = ddp[i]
        ppr[i] = rrp[i]
        a=n_average
        
    
    if s_number != False:
        ir = s_number
        for ic in range(int(len(f_range)/n_average)):
            T = (np.sum(10**(ppd[ic][ir,:]/10))**2 - np.sum(10**(ppd[ic][ir,:]/10)**2))/((len(ppd[ic][ir,:].T)-1)*np.sum(10**(ppd[ic][ir,:]/10)**2))
            Tr = (np.sum(10**(ppr[ic][ir,:]/10))**2 - np.sum(10**(ppr[ic][ir,:]/10)**2))/((len(ppr[ic][ir,:].T)-1)*np.sum(10**(ppr[ic][ir,:]/10)**2))
    
            Tf[ic] = (T - Tr)/(1-Tr)
    else:
            
        for ic in range(int(len(f_range)/n_average)):
            for ir in range(len(S.coord)):
                T[ir] = ((np.sum(10**(ppd[ic][ir,:]/10))**2) - np.sum((10**(ppd[ic][ir,:]/10)))**2)/((len(ppd[ic][ir,:].T)-1)*np.sum((10**(ppd[ic][ir,:]/10)**2)))
                                                                                                     
                Tr[ir] = ((np.sum(10**(ppr[ic][ir,:]/10))**2) - np.sum((10**(ppr[ic][ir,:]/10)))**2)/((len(ppr[ic][ir,:].T)-1)*np.sum((10**(ppr[ic][ir,:]/10)**2)))
              
                # T[ir] = (np.sum(10**(ppd[ic][ir,:]/10))**2 - np.sum(10**(ppd[ic][ir,:]/10)**2))/((len(ppd[ic][ir,:].T)-1)*np.sum(10**(ppd[ic][ir,:]/10)**2))
                # Tr[ir] = (np.sum(10**(ppr[ic][ir,:]/10))**2 - np.sum(10**(ppr[ic][ir,:]/10)**2))/((len(ppr[ic][ir,:].T)-1)*np.sum(10**(ppr[ic][ir,:]/10)**2))
        
            Tf[ic] = (np.mean(T) - np.mean(Tr))/(1-np.mean(Tr))
    
    return Tf

def theta_d_coef(frequency, pDiffuser,pRef,S,n_average=7,normalized=False):
    f_range = frequency
    Tf = np.zeros([int(len(f_range)/n_average),1],dtype=float)
    Tt = np.zeros([int(len(f_range)/n_average),1],dtype=float)


    ppd = {}
    ppr = {}
    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = [] #np.zeros_like(pDiffuser[0])
        rp = []# np.zeros_like(pDiffuser[0])# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            pD = np.abs(pDiffuser[iic,:])
            pR = np.abs(pRef[iic,:])
            rp.append(pR)# += pR #np.array([pR[c] for c in pR.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            dp.append(pD)# += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            # print(iic)
        # print("STOP")
        ddp = np.mean(dp,axis=0)#/(n_average)
        rrp = np.mean(rp,axis=0)#/(n_average)
        
        ppd[i] = ddp
        ppr[i] = rrp
        a=n_average
        
    for ic in range(int(len(f_range)/n_average)):
        T = (np.sum(np.abs(ppd[ic][:]))**2 - np.sum(np.abs(ppd[ic][:])**2))/((len(ppd[ic][:].T)-1)*np.sum(np.abs(ppd[ic][:])**2))
        Tr = (np.sum(np.abs(ppr[ic][:]))**2 - np.sum(np.abs(ppr[ic][:])**2))/((len(ppr[ic][:].T)-1)*np.sum(np.abs(ppr[ic][:])**2))
        
        Tt[ic]=T
        Tf[ic] = ((T) - (Tr))/(1-(Tr))
        
    if normalized == False:
        return Tt
    else:
        return Tf
def scattering_coef(frequency,pDiffuser,pRef,plot=False):
    f_range = frequency
    s = np.zeros([len(f_range),1])
    for i in range(len(f_range)):
        s[i] = 1 - (np.abs(np.sum(pDiffuser[i,:]*np.conj(pRef[i,:])))**2/(np.sum(np.abs(pDiffuser[i,:])**2)*np.sum(np.abs(pRef[i,:])**2)))
        
    if plot == True: 
        
        fig, ax = plt.subplots()
        ax.plot(f_range, Tf)
        ax.set_ylim(0,1)
        
    return s

def r_s_coef(frequency, pDiffuser,pRef,S,n_average=7,s_number=False):
    f_range = frequency
    sr = np.zeros([len(S.coord),1],dtype=float)
    s =  np.zeros([int(len(f_range)/n_average),1],dtype=float)
    ppd = {}
    ppr = {}
    ddp = {}
    rrp = {}
    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = np.zeros_like(pDiffuser[0])
        rp = np.zeros_like(pDiffuser[0])# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            pD = np.abs(pDiffuser.get(iic))
            pR = np.abs(pRef.get(iic))
            rp += pR #np.array([pR[c] for c in pR.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            dp += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            print(iic)
        print("STOP")
        ddp[i] = dp/(n_average)
        rrp[i] = rp/(n_average)
        
        ppd[i] = ddp[i]
        ppr[i] = rrp[i]
        a=n_average
        
    
    if s_number != False:
        ir = s_number
        for ic in range(int(len(f_range)/n_average)):
            s[ic] = 1 - (np.abs(np.sum(ppd[ic][ir,:]*np.conj(ppr[ic][ir,:])))**2/(np.sum(np.abs(ppd[ic][ir,:])**2)*np.sum(np.abs(ppr[ic][ir,:])**2)))
    else:
            
        for ic in range(int(len(f_range)/n_average)):
            for ir in range(len(S.coord)):
                sr[ir] = 1 - (np.abs(np.sum(ppd[ic][ir,:]*np.conj(ppr[ic][ir,:])))**2/(np.sum(np.abs(ppd[ic][ir,:])**2)*np.sum(np.abs(ppr[ic][ir,:])**2)))
        
            s[ic] = np.mean(sr)
    
    return s


class IR(object):
    """Perform a room impulse response computation."""

    def __init__(self, sampling_rate, duration,
            minimum_frequency, maximum_frequency):
        """
        Setup the room impulse computation.

        Parameters
        ----------
        sampling_rate : integer
            Sampling rate in Hz for the time signal.
        duration: float
            Time in seconds until which to sample the room impulse response.
        minimum_frequency: float
            Minimum sampling frequency
        maximum_frequency: float
            Maximum sampling frequency

        """
        self._number_of_frequencies = int(round(sampling_rate * duration))
        self._sampling_rate = sampling_rate
        self._duration = duration
        self._frequencies = (sampling_rate * np.arange(self._number_of_frequencies) 
                / self._number_of_frequencies)
        self._timesteps = np.arange(self._number_of_frequencies) / sampling_rate

        self._maximum_frequency = maximum_frequency
        self._minimum_frequency = minimum_frequency

        self._frequency_filter_indices = np.flatnonzero(
                (self._frequencies <= self._maximum_frequency) & 
                (self._frequencies >= self._minimum_frequency))

        self._high_pass_frequency = 2 * minimum_frequency
        self._low_pass_frequency = 2 * maximum_frequency

        self._high_pass_order = 4
        self._low_pass_order = 4

        self._alpha = 0.18  # Tukey window alpha

        
    @property
    def number_of_frequencies(self):
        """Return number of frequencies."""
        return self._number_of_frequencies

    @property
    def sampling_rate(self):
        """Return sampling rate."""
        return self._sampling_rate

    @property
    def duration(self):
        """Return duration."""
        return self._duration

    @property
    def timesteps(self):
        """Return time steps."""
        return self._timesteps

    @property
    def frequencies(self):
        """Return frequencies."""
        return self._frequencies

    @property
    def filtered_frequencies(self):
        """Return the filtered frequencies."""
        return self.frequencies[
                self._frequency_filter_indices
                ]

    @property
    def maximum_frequency(self):
        """Return maximum frequency."""
        return self._maximum_frequency

    @property
    def minimum_frequency(self):
        """Return minimum frequency."""
        return self._minimum_frequency

    @property
    def high_pass_frequency(self):
        """Return high pass frequency."""
        return self._high_pass_frequency

    @high_pass_frequency.setter
    def high_pass_frequency(self, freq):
        """Set high pass frequency."""
        self._high_pass_frequency = freq

    @property
    def low_pass_frequency(self):
        """Return low pass frequency."""
        return self._low_pass_frequency

    @low_pass_frequency.setter
    def low_pass_frequency(self, freq):
        """Set low pass frequency."""
        self._low_pass_frequency = freq

    @property
    def high_pass_filter_order(self):
        """Return high pass filter order."""
        return self._high_pass_order

    @high_pass_filter_order.setter
    def high_pass_filter_order(self, order):
        """Set high pass filter order."""
        self._high_pass_order = order

    @property
    def low_pass_filter_order(self):
        """Return low pass filter order."""
        return self._low_pass_order

    @low_pass_filter_order.setter
    def low_pass_filter_order(self, order):
        """Set low pass filter order."""
        self._low_pass_order = order


    def compute_room_impulse_response(
            self, values_at_filtered_frequencies):
        """
        Compute the room impulse response.

        Parameters
        ----------
        values_at_filtered_frequencies : array
            The frequency domain values to be transformed taken
            at the filtered frequencies.

        Output
        ------
        An array of approximate time values at the given time steps.
        
        """
        from scipy.signal import butter, freqz, tukey
        from scipy.fftpack import ifft
        
        b_high, a_high = butter(
                self.high_pass_filter_order,
                self.high_pass_frequency * 2 / self.sampling_rate, 
                'high')

        b_low, a_low = butter(
                self.low_pass_filter_order,
                self.low_pass_frequency * 2 / self.sampling_rate, 
                'low')

        high_pass_values = freqz(
                b_high, a_high, self.filtered_frequencies,
                fs=self.sampling_rate)[1]

        low_pass_values = freqz(
                b_low, a_low, self.filtered_frequencies,
                fs=self.sampling_rate)[1]

        butter_filtered_values = (values_at_filtered_frequencies * 
                np.conj(low_pass_values) * np.conj(high_pass_values))

        # windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies),
        #         min([self.maximum_frequency - self.low_pass_frequency,
        #              self.high_pass_frequency - self.minimum_frequency]) /
        #         (self.maximum_frequency - self.minimum_frequency))

        windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies), alpha=self._alpha)

        full_frequency_values = np.zeros(self.number_of_frequencies, dtype='complex128')
        full_frequency_values[self._frequency_filter_indices] = windowed_values
        full_frequency_values[-self._frequency_filter_indices] = np.conj(windowed_values)

        return ifft((full_frequency_values)) * self.number_of_frequencies

# def compute_ir(fmin = 20,fmax=200,df,p):
#     """
#     Compute the room impulse response.

#     Parameters
#     ----------
#     values_at_filtered_frequencies : array
#         The frequency domain values to be transformed taken
#         at the filtered frequencies.

#     Output
#     ------
#     An array of approximate time values at the given time steps.
    
#     """
    
#     from scipy.signal import butter, freqz, tukey
#     from scipy.fftpack import ifft
    
#     high_pass_filter_order = 2
#     high_pass_frequency = fmin
#     low_pass_frequency = fmax
#     alpha = 4
#     sampling_rate = 44100
#     b_high, a_high = butter(
#             high_pass_filter_order,
#             high_pass_frequency * 2 / sampling_rate, 
#             'high')

#     b_low, a_low = butter(
#             high_pass_filter_order,
#             low_pass_frequency * 2 / sampling_rate, 
#             'low')

#     high_pass_values = freqz(
#             b_high, a_high, filtered_frequencies,
#             fs=sampling_rate)[1]

#     low_pass_values = freqz(
#             b_low, a_low, filtered_frequencies,
#             fs=sampling_rate)[1]

#     butter_filtered_values = (values_at_filtered_frequencies * 
#             _np.conj(low_pass_values) * _np.conj(high_pass_values))

#     # windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies),
#     #         min([self.maximum_frequency - self.low_pass_frequency,
#     #              self.high_pass_frequency - self.minimum_frequency]) /
#     #         (self.maximum_frequency - self.minimum_frequency))

#     windowed_values = butter_filtered_values * tukey(len(filtered_frequencies), alpha=alpha)

#     full_frequency_values = _np.zeros(self.number_of_frequencies, dtype='complex128')
#     full_frequency_values[self._frequency_filter_indices] = windowed_values
#     full_frequency_values[-self._frequency_filter_indices] = _np.conj(windowed_values)

#     return ifft(_np.conj(full_frequency_values)) * self.number_of_frequencies