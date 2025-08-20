import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Process, Queue, Pool
import emcee
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rc('font', size=24)

# Here you need to input your paths to the Escape library and the AGAMA_Zv_calibration folder
path_to_function_libraries='/your_path_to/Escape_Library/Function_Libraries/'
path_to_Zv_calibration='/your_path_to/AGAMA_Zv_calibration'

sys.path.insert(0, path_to_function_libraries)
from escape_theory_functions import dehnen_nfwM200_errors, rho_crit_z
from escape_analysis_functions import main

#Helper function for our escape library
def cosmology(cosmology):
    case = cosmology.name
    if case == 'Flatw0waCDM':
        return [cosmology.Om0, cosmology.w0, cosmology.wa, cosmology.h]
    
    elif case == 'FlatwCDM':
        return [cosmology.Om0, cosmology.w0, cosmology.h]

    elif case == 'wCDM':
        return [cosmology.Om0, cosmology.Ode0, cosmology.w0,cosmology.h]
        
    elif case == 'LambdaCDM':
        return [cosmology.Om0, cosmology.Ode0, cosmology.h]

    elif case == 'FlatLambdaCDM':
        return [cosmology.Om0, cosmology.h]

# set our desired cosmology
Omega_m = 0.3
Omega_L = 1-Omega_m
h0 = 0.7

cosmo_name = 'FlatLambdaCDM'
cosmo = FlatLambdaCDM(H0=h0*100.0,Om0=Omega_m,name = cosmo_name)
cosmo_params = cosmology(cosmo)

# In this example we assume the cluster we observe is A7. We use Rines 2013 and 2016 HeCS and HeCS-SZ data, and a starting estimate of M200 from Herbonnet 2020:

# Load galaxy data
# galaxy_data is assumed to have this format: RAh, RAm, RAs, DEd, DEm, DEs, redshift
path_to_galaxy_data = '/your_path_to/Escape_Library/Example/'
galaxy_positional_data = np.genfromtxt(os.path.join(path_to_galaxy_data, 'Rines_galaxy_data.txt'))


# Load cluster data
#cl_ra: Cluster right ascension in decimal degrees
#cl_dec: Cluster declination in decimal degrees  
#cl_z: Cluster redshift
cluster_positional_data = (2.9385416666666666, 32.41569444444444, 0.106)
M200, M200_err_up, M200_err_down=(440000000000000.06, 0.15588787296739426, 0.24551266781415038) #input starting esimate to bin/count galaxies, measured in solar masses and dex respectively
main(path_to_Zv_calibration,galaxy_positional_data,cluster_positional_data,M200,M200_err_up,M200_err_down,cosmo_params,cosmo_name,nwalkers=250,nsteps=1000)
