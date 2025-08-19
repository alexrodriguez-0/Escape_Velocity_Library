# Escape_Velocity_Library
All tools and functions required for escape velocity analysis of galaxy clusters.
Escape Velocity Mass Estimation Library
A Python library for estimating galaxy cluster masses using the escape velocity technique, as described in Rodriguez et al. 2025 (arXiv:2507.20938).
Overview
This library implements a novel approach to measure galaxy cluster masses by identifying and modeling the escape velocity edge in projected phase-space data. Unlike traditional caustic techniques, this method accounts for the statistical suppression of the observed edge due to sparse sampling through a calibrated suppression function (Zv).
Key Features

Accurate mass estimation: Achieves excellent agreement with weak lensing masses (correlation coefficient ~0.68)
Minimal bias: ~2% systematic bias when proper cosmology is assumed
Robust to systematics: Handles non-equilibrium clusters, asphericity, and interlopers
MCMC-based inference: Full posterior distributions for mass estimates

Installation
Prerequisites
bashpip install numpy pandas matplotlib astropy emcee scipy
Required Libraries

numpy >= 1.19
pandas >= 1.0
matplotlib >= 3.0
astropy >= 4.0
emcee >= 3.0
scipy >= 1.5
multiprocessing (standard library)

Repository Structure
Escape_Velocity_Library/
├── AGAMA_Zv_calibration/       # Pre-computed suppression function calibrations
│   └── Zv_fits_z_*.pkl         # Calibration files for z∈[0,0.7], M200∈[10^14,10^15.6] M⊙
├── Example/
│   ├── run_example.py           # Example script for Abell 7 cluster
│   └── Rines_galaxy_data.txt   # Galaxy spectroscopic data from HeCS/HeCS-SZ
└── Function_Libraries/
    ├── escape_analysis_functions.py  # Main analysis pipeline
    └── escape_theory_functions.py    # Theoretical models and cosmology
Methodology
Physical Model
The escape velocity profile is derived from the effective potential in an accelerating universe:
v_esc²(r) = -2[Ψ(r) - Ψ(r_eq)] - q(z)H²(z)(r² - r_eq²)
where:

Ψ(r) is the gravitational potential (modeled using a Dehnen profile fitted to NFW)
r_eq is the equilibrium radius where gravitational and cosmological accelerations balance
q(z) is the deceleration parameter
H(z) is the Hubble parameter

Key Modeling Assumptions

Density Profile: Dehnen profile parametrization mapped to NFW with mass-concentration relation (Duffy et al. 2008)
Cosmology: Flat ΛCDM (Ωm = 0.3, h = 0.7 by default, customizable)
Suppression Function: Skewed-normal distribution calibrated using AGAMA simulations
Velocity Anisotropy: Constant β = 0.25 (validated against simulations)
Interloper Removal: Shifting-gapper algorithm with optimized parameters
Edge Identification: Maximum absolute velocity in radial bins with enforced monotonicity

Suppression Function (Zv)
The observed escape edge is suppressed relative to the true 3D escape velocity:
⟨v_esc,observed⟩(r⊥) = ⟨v_esc,3D⟩(r⊥) / Zv(N)
where Zv follows a skewed-normal distribution with parameters dependent on the phase-space sampling N (number of galaxies between 0.2-1.0 r200).
Usage
Basic Example
pythonimport numpy as np
from escape_analysis_functions import main
from escape_theory_functions import cosmology
from astropy.cosmology import FlatLambdaCDM

# Set cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')
cosmo_params = [0.3, 0.7]  # [Omega_m, h]

# Load galaxy data (format: RAh, RAm, RAs, DEd, DEm, DEs, redshift)
galaxy_data = np.genfromtxt('Rines_galaxy_data.txt')

# Cluster parameters
cluster_coords = (2.939, 32.416, 0.106)  # (RA[deg], Dec[deg], z)
M200_initial = 4.4e14  # Initial mass estimate [M⊙]
M200_err_up = 0.156    # Upper error in dex
M200_err_down = 0.246  # Lower error in dex

# Run mass estimation
path_to_calibration = '/path/to/AGAMA_Zv_calibration'
main(path_to_calibration, galaxy_data, cluster_coords, 
     M200_initial, M200_err_up, M200_err_down,
     cosmo_params, 'FlatLambdaCDM', 
     nwalkers=250, nsteps=1000)
Input Data Format
Galaxy Data: ASCII file with columns:

Columns 1-3: Right Ascension (hours, minutes, seconds)
Columns 4-6: Declination (degrees, arcminutes, arcseconds)
Column 7: Redshift (as cz in km/s)

Cluster Data: Tuple containing:

RA in decimal degrees
Dec in decimal degrees
Cluster redshift

Key Parameters

nwalkers: Number of MCMC walkers (default: 250)
nsteps: Number of MCMC steps per walker (default: 1000)
n_processes: Number of parallel processes (default: 30)

Calibration Data
The AGAMA_Zv_calibration/ directory contains pre-computed suppression functions covering:

Redshift range: 0.00 ≤ z ≤ 0.70 (steps of 0.05)
Mass range: 14.0 ≤ log₁₀(M200/M⊙) ≤ 15.6 (steps of 0.1 dex)
Sampling range: 50 ≤ N ≤ 1200 galaxies

Each file contains skewed-normal parameters (location, scale, skewness) as linear functions of N for 5 radial bins.
Output
The pipeline produces:

Mass posterior: MCMC chains with escape velocity mass estimates
Diagnostics: Convergence statistics and uncertainty estimates
Visualization: Phase-space diagram showing:

Galaxy distribution
Identified edge profile
Dynamical mass fit
Comparison with initial mass estimate



Systematic Uncertainties
Accounted for in the method:

Line-of-sight projection effects (via Zv calibration)
Sparse sampling variations
Interloper contamination
Galaxy redshift errors (~30 km/s)
Non-equilibrium dynamics
Cluster asphericity

Dominant systematic:

Cosmological parameters (particularly H₀): ~10% effect on masses

Citation
If you use this code in your research, please cite:
bibtex@article{Rodriguez2025,
    author = {Rodriguez, Alexander and Miller, Christopher J.},
    title = {The Concordance of Weak Lensing and Escape Velocity Mass Estimates for Galaxy Clusters},
    journal = {arXiv e-prints},
    year = {2025},
    eprint = {2507.20938},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.CO}
}

Contact
For questions or issues, please contact:

Alexander Rodriguez (alexcrod@umich.edu)
Christopher J. Miller (christoq@umich.edu)
