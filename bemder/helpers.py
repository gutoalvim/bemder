import bemder
import cloudpickle
import pickle
import os
import time
import bemder.bem_api as bem


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
    from bempp.core.cl_helpers import Context, set_default_device
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
    from bempp.core.cl_helpers import Context, set_default_device
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


def pack_bem(filename='Configuration', save=True, folder=None, timestamp=True, ext='.pickle'):
    if save is True:
        # timestr = time.strftime("%Y%m%d-%H%M_")
        # filename = filename+'_'+str(self.config_number)
        outfile = open(filename + ext, 'wb')

        packedData = {}

        try:
            simulation_data, bem_data = bem.pack()
            packedData['bem'] = [simulation_data, bem_data]
        except AttributeError:
            if save:
                print('BEM data not found.')

    if save is True:
        cloudpickle.dump(packedData, outfile)
        outfile.close()
        print('Saved successfully.')
    if save is False:
        return packedData
